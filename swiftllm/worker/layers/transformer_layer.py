"""
Transformer layer utilities in the Llama model
"""

import time
import torch
import torch.distributed as dist
#import vllm_flash_attn_2_cuda as flash_attn_cuda
import flash_attn_2_cuda as flash_attn_cuda
# import vllm_flash_attn
import cupy as cp

# pylint: disable=no-name-in-module
from swiftllm_c import \
    fused_add_rmsnorm_inplace, \
    silu_and_mul_inplace, \
    rotary_embedding_inplace, \
    store_kvcache#, paged_attention
    # linear, \

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.weight import LlamaTransformerLayerWeight
from swiftllm.worker.block_swapper import Swapper
from swiftllm.structs import SubBatch

from swiftllm.worker.kernels.linear import linear
# from swiftllm.worker.kernels.rotary_emb import rotary_embedding_inplace
# from swiftllm.worker.kernels.kvcache_mgmt import store_kvcache
# from swiftllm.worker.kernels.silu_and_mul import silu_and_mul_inplace
# from swiftllm.worker.kernels.rmsnorm import fused_add_rmsnorm_inplace
from swiftllm.worker.kernels.paged_attn import paged_attention
from swiftllm.worker.kernels.prefill_attn import prefill_attention

class TransformerEvents:
    """
    Performance monitoring & synchronization events for a transformer layer
    """
    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        self.stage_s = torch.cuda.Event(enable_timing=True)
        self.linr_e = torch.cuda.Event(enable_timing=True)
        self.pref_e = torch.cuda.Event(enable_timing=True)
        self.gdec_e = torch.cuda.Event(enable_timing=True)
        self.qkvtr_e = torch.cuda.Event()
        self.lnch_s = 0.0
        self.lnch_m = 0.0
        self.cdec_s = 0.0
        self.cdec_e = 0.0
        self.lnch_e = 0.0
    
    @property
    def linr_time(self) -> float:
        """
        Time for linear operations + other small operations other than attention
        """
        return self.stage_s.elapsed_time(self.linr_e)

    @property
    def pref_time(self) -> float:
        """
        Time for prefilling
        """
        return self.linr_e.elapsed_time(self.pref_e)

    @property
    def gdec_time(self) -> float:
        """
        Time for GPU decoding
        """
        return self.pref_e.elapsed_time(self.gdec_e)

    @property
    def cdec_time(self) -> float:
        """
        Time for CPU decoding
        """
        return self.cdec_e - self.cdec_s

    @property
    def lnch_time(self) -> float:
        """
        Time for kernel launch & other Python overheads other than CPU decoding
        """
        return self.lnch_e - self.cdec_e + self.lnch_m - self.lnch_s

    def pf_record(self, name: str):
        """
        Record the event if performance monitoring is enabled
        """
        if self.engine_config.monitor_performance:
            getattr(self, name).record()

    def pf_time(self, name: str):
        """
        Record the time if performance monitoring is enabled
        """
        if self.engine_config.monitor_performance:
            setattr(self, name, time.perf_counter() * 1e3) # ms

    def pf_time_nocpu(self):
        """
        If performance monitoring is enabled but there is no CPU decoding sequences, set CPU time stamps to the current time
        """
        if self.engine_config.monitor_performance:
            self.lnch_m = self.cdec_s = self.cdec_e = time.perf_counter()

class LlamaTransformerLayer:
    """
    A transformer layer in the Llama model, may contain weights of the next layer
    """
    def __init__(
        self,
        model_config: LlamaModelConfig,
        engine_config: EngineConfig,
        weight: LlamaTransformerLayerWeight,
        next_layer_weight: LlamaTransformerLayerWeight | None,
        Gpu_communication_stream: torch.cuda.Stream,
        
        G_stream: torch.cuda.Stream,
        g2G_stream: torch.cuda.Stream,

        layer_id: int

    ):
        self.model_config = model_config
        self.engine_config = engine_config
        self.weight = weight
        self.next_layer_weight = next_layer_weight
        self.Gpu_communication_stream = Gpu_communication_stream
        self.layer_id = layer_id

        self.events = [TransformerEvents(engine_config) for _ in range(2)]

        # Set after KV cache initialization
        self.swapper = None


       
        self.G_stream = G_stream
        self.g2G_stream = g2G_stream

        self.g_event = torch.cuda.Event()
        self.g_o_event = torch.cuda.Event()
        with torch.cuda.device('cuda:1'):
            self.g2G_event = torch.cuda.Event()
            self.G_event = torch.cuda.Event()
            self.g2G_block_event = torch.cuda.Event()

        
        


    def set_swapper(self, swapper: Swapper):
        """
        Set the swapper for the layer
        """
        self.swapper = swapper


    def _comm_wait_compute(self):
        """
        Do communication after necessary computation
        """
        # self.Gpu_communication_stream.wait_stream(torch.cuda.default_stream())

        # torch.cuda.synchronize(torch.device('cuda:1'))

        start = time.perf_counter()
        self.Gpu_communication_stream.wait_stream(torch.cuda.default_stream())
        end = time.perf_counter()

        print(f"wait compute Synchronization took {end - start:.6f} seconds")

        
        # event = torch.cuda.Event()
        # event.synchronize()

    def _compute_wait_comm_2(self):
        torch.cuda.default_stream().wait_stream(self.Gpu_communication_stream)
    
    def _compute_wait_comm_swap(self):
        self.g2G_block_event.wait(torch.cuda.default_stream())

    def _compute_wait_comm(self):
        """
        Do computation after necessary communication
        """
        self.g_event.wait(torch.cuda.default_stream())
        # self.g2G_block_event.wait(torch.cuda.default_stream())
        # torch.cuda.default_stream().wait_stream(self.Gpu_communication_stream)



        # start = time.perf_counter()
        # self.g_event.synchronize()
        # end = time.perf_counter()

        # print(f"wait commm Synchronization took {end - start:.6f} seconds")





    def _maybe_allreduce(self, x: torch.Tensor):
        if self.model_config.world_size > 1:
            dist.all_reduce(x)


    def _transfer_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch: SubBatch,
        cur_stage: int = 0
    ):
        """
        Initiate transfer of QKV to CPU buffers
        """
        # self._comm_wait_compute() # pre proj for cuda:0

        torch.cuda.nvtx.range_push("transfer_kv")

        if batch.num_Gdecs > 0:
            # with torch.cuda.device('cuda:1'):
            #     self.g_event.wait(self.g2G_stream)
            
            

            with torch.cuda.device('cuda:1'), torch.cuda.stream(self.g2G_stream):
                self.g_event.wait(self.g2G_stream)

                cp.cuda.Device(1).use()
                stream = cp.cuda.Stream(non_blocking=True)
                self.g_cp_event = cp.cuda.Event()
                with stream:
                    

                    qc = self.swapper.q_Gpu[:batch.num_Gdecs]
                    kc = self.swapper.k_Gpu[:batch.num_Gdecs]
                    vc = self.swapper.v_Gpu[:batch.num_Gdecs]

                    element_size = 2  # float16 = 2 bytes
                    num_elements = batch.num_Gdecs
                    offset_elements = q.numel() - num_elements
                    # offset_bytes = offset_elements * element_size
                    # copy_bytes = num_elements * element_size

                    # src_ptr = q.data_ptr() + offset_bytes
                    # dst_ptr = qc.data_ptr()

                    # cp.cuda.runtime.memcpyAsync(
                    #     qc.data_ptr(),
                    #     q.data_ptr() + offset_elements * element_size,
                    #     num_elements * element_size,  # float16 = 2 bytes
                    #     cp.cuda.runtime.memcpyDeviceToDevice,
                    #     stream.ptr
                    # )
                    # cp.cuda.runtime.memcpyAsync(
                    #     kc.data_ptr(),
                    #     k.data_ptr() + offset_elements * element_size,
                    #     num_elements * element_size,  # float16 = 2 bytes
                    #     cp.cuda.runtime.memcpyDeviceToDevice,
                    #     stream.ptr
                    # )
                    # cp.cuda.runtime.memcpyAsync(
                    #     vc.data_ptr(),
                    #     v.data_ptr() + offset_elements * element_size,
                    #     num_elements * element_size,  # float16 = 2 bytes
                    #     cp.cuda.runtime.memcpyDeviceToDevice,
                    #     stream.ptr
                    # )

                    cp.cuda.runtime.memcpyPeerAsync(
                        qc.data_ptr(), 1,  # destination pointer and device
                        q.data_ptr() + offset_elements * element_size, 0,  # source pointer and device
                        num_elements * element_size,
                        stream.ptr
                    )
                    cp.cuda.runtime.memcpyPeerAsync(
                        kc.data_ptr(), 1,  # destination pointer and device
                        k.data_ptr() + offset_elements * element_size, 0,  # source pointer and device
                        num_elements * element_size,
                        stream.ptr
                    )
                    cp.cuda.runtime.memcpyPeerAsync(
                        vc.data_ptr(), 1,  # destination pointer and device
                        v.data_ptr() + offset_elements * element_size, 0,  # source pointer and device
                        num_elements * element_size,
                        stream.ptr
                    )
                    self.g_cp_event.record(stream)


                # qc.copy_(q[-batch.num_Gdecs:], non_blocking=True)
                # kc.copy_(k[-batch.num_Gdecs:], non_blocking=True)
                # vc.copy_(v[-batch.num_Gdecs:], non_blocking=True)


                # qc.copy_(q[-batch.num_Gdecs:].to(qc.device))# , non_blocking=True) # update .to(device)
                # kc.copy_(k[-batch.num_Gdecs:].to(kc.device))#, non_blocking=True)
                # vc.copy_(v[-batch.num_Gdecs:].to(vc.device))#, non_blocking=True)
                # self.events[cur_stage].qkvtr_e.record()
                self.g2G_event.record(self.g2G_stream)
        torch.cuda.nvtx.range_pop()


    def _swap_out_blocks(
        self,
        batch: SubBatch,
    ):
        """
        Swap blocks from GPU to CPU, assume that new prefilled KVs are ready in the last stage
        """
        torch.cuda.nvtx.range_push("_swap_out_blocks")
        if batch.num_Gprfs > 0:
            # with torch.cuda.device("cuda:1"):
            self.g_o_event.wait(self.g2G_stream)
            #     with torch.cuda.stream(self.g2G_stream):
            with torch.cuda.stream(self.g2G_stream):
                self.swapper.swap_blocks(
                    batch.src_blk_ids, 
                    batch.dst_blk_ids, 
                    is_swap_out=True, 
                    gpu_layer=self.model_config.num_layers if self.engine_config.extra_layer_for_cprf else self.layer_id,
                    Gpu_layer=self.layer_id
                )
                self.g2G_block_event.record()
        torch.cuda.nvtx.range_pop()


    def _preproj(
        self,
        embeddings: torch.Tensor,
        batch: SubBatch,
        layer_off: int = 0
    ) -> tuple[torch.Tensor]:
        """
        Perform pre-projection, including RMSNorm, QKV calculation, and rotary embedding
        """
        torch.cuda.nvtx.range_push("_preproj")

        weight = self.weight if not layer_off else self.next_layer_weight

        self._maybe_allreduce(embeddings)
        fused_add_rmsnorm_inplace(
            embeddings,
            batch.residual_buf,
            weight.attn_norm,
            self.model_config.rms_norm_eps
        )

        # Calculate QKV
        q = linear(embeddings, weight.q_proj)		# [iter_width, hidden_size / ws]
        k = linear(embeddings, weight.k_proj)		# [iter_width, num_kv_heads / ws * head_dim]
        v = linear(embeddings, weight.v_proj)		# [iter_width, num_kv_heads / ws * head_dim]
        q = q.view(batch.iter_width, -1, self.model_config.head_dim)
        k = k.view(batch.iter_width, -1, self.model_config.head_dim)
        v = v.view(batch.iter_width, -1, self.model_config.head_dim)

        # Rotary emb
        rotary_embedding_inplace(
            q,
            k,
            batch.position_sin,
            batch.position_cos
        )

        # Here we only store k, v for prefilling, the kernel won't store decoding KVs
        if batch.num_prefs > 0 and self.swapper is not None:
            gpu_layer = (self.layer_id + layer_off) % self.model_config.num_layers
            itm_layer = self.model_config.num_layers if self.engine_config.extra_layer_for_cprf else gpu_layer
            # NOTE: if swapping is too slow, there's risk that we writes to what's being swapped out
            self._compute_wait_comm_swap()
            store_kvcache(
                k,
                v,
                self.swapper.k_cache, 
                self.swapper.v_cache,
                self.swapper.gpu_block_table,
                batch.prgd_seq_ids[:batch.num_prefs],
                batch.pref_st_locs_we,
                batch.prgd_seq_lens[:batch.num_prefs],
                itm_layer,
                gpu_layer,
                batch.num_Gprfs,
                batch.max_pref_toks
            )
        self.g_event.record(torch.cuda.default_stream(device='cuda:1'))
        torch.cuda.nvtx.range_pop()
        return q, k, v


    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch: SubBatch,
        cur_stage: int = 0 # also as the offset of layer_id
    ):
        """
        Stores attention output of current batch into buffer o
        """

        torch.cuda.nvtx.range_push("attention_local")

        events = self.events[cur_stage]
        o = batch.attn_out_buf.view(batch.iter_width, -1, self.model_config.head_dim)
        cur_layer_id = (self.layer_id + cur_stage) % self.model_config.num_layers

        if batch.num_prefs > 0:
            # If we are using Ampere GPU or above, we can use the flash attention kernel.
            # Otherwise, we use a customized prefilling kernel.
            if torch.cuda.get_device_properties(0).major >= 8:
                # flash_attn.forward(
                # pylint: disable=c-extension-no-member
                flash_attn_cuda.varlen_fwd(
                    q[:batch.sum_pref_toks],
                    k[:batch.sum_pref_toks],
                    v[:batch.sum_pref_toks],
                    o[:batch.sum_pref_toks],
                    batch.pref_st_locs_we,
                    batch.pref_st_locs_we,
                    None,
                    None,  # block table
                    None,  # alibi slopes
                    None,       # add to align with flash_attn_cuda call
                    batch.max_pref_toks,
                    batch.max_pref_toks,
                    0.0,
                    self.model_config.softmax_scale,
                    False,
                    True,  # causal
                    -1,    # window size 0
                    -1,    # window size 1
                    0.0,   # softcap
                    False, # return softmax
                    None
                )
            else:
                prefill_attention(
                    q, k, v, o[:batch.sum_pref_toks],
                    self.model_config, self.engine_config, batch
                )
        events.pf_record("pref_e")

        # Actually we can further separate KV-cache storing for prefilling and decoding,
        # but since the kernel is fast enough, we put all to decoding stream for simplicity
        if batch.num_gdecs > 0:
            # with torch.cuda.stream(self.decoding_piggyback_stream):
            #     torch.cuda.current_stream().wait_event(self.events[cur_stage].stage_s)
            paged_attention(
                q[batch.sum_pref_toks:batch.sum_prgd_toks],
                k[batch.sum_pref_toks:batch.sum_prgd_toks],
                v[batch.sum_pref_toks:batch.sum_prgd_toks],
                o[batch.sum_pref_toks:batch.sum_prgd_toks],
                self.swapper.k_cache, 
                self.swapper.v_cache,
                self.model_config.softmax_scale,
                self.swapper.gpu_block_table,
                batch.prgd_seq_ids[batch.num_prefs:],
                batch.prgd_seq_lens[batch.num_prefs:],
                cur_layer_id,
                batch.seq_block_size,
                batch.num_seq_blocks,
            )
        events.pf_record("gdec_e")

        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("attention_remote")

                
        if batch.num_Gdecs > 0:
            # oc = self.swapper.o_Gpu[:batch.num_Gdecs]
            # events.pf_time("lnch_m")
            # self.events[cur_stage].qkvtr_e.synchronize()
            # events.pf_time("cdec_s")
            # torch.ops.pacpu.paged_attention_cpu(
            #     cur_layer_id,
            #     self.model_config.softmax_scale,
            #     batch.seq_ids_list[batch.num_prgds:],
            #     batch.seq_lens_list[batch.num_prgds:],

            #     self.swapper.q_Gpu[:batch.num_Gdecs],
            #     self.swapper.k_Gpu[:batch.num_Gdecs],
            #     self.swapper.v_Gpu[:batch.num_Gdecs],
            #     self.swapper.k_swap,
            #     self.swapper.v_swap,
            #     self.swapper.Gpu_block_table,
            #     oc
            # )
            # events.pf_time("cdec_e")
            # with torch.cuda.stream(self.Gpu_communication_stream):
            #     o[-batch.num_Gdecs:, :].copy_(oc, non_blocking=True)
            # with torch.cuda.device('cuda:1'):
            #     self.g2G_event.wait(self.G_stream)


            self.g_cp_event.synchronize()

            with torch.cuda.device('cuda:1'), torch.cuda.stream(self.G_stream):
                self.g2G_event.wait(self.G_stream)

                # cp.cuda.Device(1).use()
                # other_stream = cp.cuda.Stream(non_blocking=True)
                # other_stream.wait_event(self.g_cp_event)

                # self.g_pa_event = cp.cuda.Event()
                # with other_stream:

                oc = self.swapper.o_Gpu[:batch.num_Gdecs]
                events.pf_time("lnch_m")
                # self.g2G_stream.wait_event(self.g2G_event)
                # self.events[cur_stage].qkvtr_e.synchronize()
                events.pf_time("cdec_s")
                paged_attention(
                    self.swapper.q_Gpu[:batch.num_Gdecs],
                    self.swapper.k_Gpu[:batch.num_Gdecs],
                    self.swapper.v_Gpu[:batch.num_Gdecs],
                    oc,
                    self.swapper.k_swap, 
                    self.swapper.v_swap,
                    self.model_config.softmax_scale,
                    self.swapper.Gpu_block_table,
                    # torch.tensor(batch.seq_ids_list[batch.num_prgds:], dtype=torch.int32, device="cuda:1"),
                    # torch.tensor(batch.seq_lens_list[batch.num_prgds], dtype=torch.int32, device="cuda:1"),
                    batch.prgd_seq_ids_post,
                    batch.prgd_seq_lens_post,
                    # batch.prgd_seq_ids[batch.num_prefs:],
                    # batch.prgd_seq_lens[batch.num_prefs:],
                    cur_layer_id,
                    batch.seq_block_size,
                    batch.num_seq_blocks,
                )
                # self.g_pa_event.record(other_stream)
                events.pf_time("cdec_e")
                self.G_event.record(self.G_stream)
                # with torch.cuda.stream(self.Gpu_communication_stream):
            self.G_event.wait(self.Gpu_communication_stream)
            with torch.cuda.stream(self.Gpu_communication_stream):
                # self.g_pa_event.synchronize()

                # self.g2G_stream.wait_event(self.G_event)
                o[-batch.num_Gdecs:, :].copy_(oc, non_blocking=True)
                self.g_o_event.record()
        else:
            events.pf_time_nocpu()
        self._compute_wait_comm_2() # Wait for CPU decoding to finish
        
        torch.cuda.nvtx.range_pop()

    def _postproj(
        self,
        batch: SubBatch        
    ) -> torch.Tensor:

        torch.cuda.nvtx.range_push("_postproj")

        o = linear(batch.attn_out_buf, self.weight.o_proj)
        self._maybe_allreduce(o)
        fused_add_rmsnorm_inplace(o, batch.residual_buf, self.weight.ffn_norm, self.model_config.rms_norm_eps)
        ug = linear(o, self.weight.up_gate_proj)
        del o
        silu_and_mul_inplace(ug)
        embeddings = linear(ug[:, :ug.shape[1] // 2], self.weight.down_proj)
        del ug

        torch.cuda.nvtx.range_pop()

        return embeddings


    def forward(self, batch: SubBatch, embeddings: torch.Tensor) -> torch.Tensor:
        """
        (fused) Add last layer's residual, and perform RMSNorm
        Before: input_embds is the output of the last FFN block, and residual_buf
                is the residual to be added to input_embds
        After: input_embds will be RMSNorm(input_embds + residual_buf), and
               residual_buf will be input_embds + residual_buf (which will be
               used as the residual after the attention block)
        """
        self.events[0].pf_record("stage_s")
        self.events[0].pf_time("lnch_s")
        q, k, v = self._preproj(embeddings, batch)
        self.events[0].pf_record("linr_e")
        self._transfer_qkv(q, k, v, batch)
        self._attention(q, k, v, batch)
        del q, k, v
        self.events[1].pf_record("stage_s")
        self._swap_out_blocks(batch)
        embeddings = self._postproj(batch)
        self.events[0].pf_time("lnch_e")
        self.events[1].pf_record("linr_e")
        return embeddings


    def _forward_pipeline_stage(
        self,
        q1: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
        k1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        batches: list[SubBatch],
        cur_stage: int,
    ) -> tuple[torch.Tensor]:
        """
        Do 1 pipeline stage:

            batch 0 : o0   |=> post-projection[i] -> pre-projection[i+1] |=> qkv0
            batch 1 : qkv1 |=>      attention[i + attn_layer_id_offs]    |=> [o1]

        buffer of o1 is given as input

        Note that batch-0 here may actually be batch-1 in the forward pass, should reverse the list
        in the second stage
        """
        self.events[cur_stage].pf_record("stage_s")
        self.events[cur_stage].pf_time("lnch_s")
        self._transfer_qkv(q1, k1, v1, batches[cur_stage^1], cur_stage=cur_stage)
        self._swap_out_blocks(batches[cur_stage])
        e0 = self._postproj(batches[cur_stage])
        q0, k0, v0 = self._preproj(e0, batches[cur_stage], layer_off=1)
        del e0
        self.events[cur_stage].pf_record("linr_e")
        self._attention(q1, k1, v1, batches[cur_stage^1], cur_stage=cur_stage)
        self.events[cur_stage].pf_time("lnch_e")

        return q0, k0, v0


    def forward_double(
        self,
        q1: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
        k1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        batches: list[SubBatch]
    ) -> tuple[torch.Tensor]:
        """
        Do all jobs for 1 transformer layer for 2 batches

        Note that the weights of pre-projection need to be of the next layer compared to the post-projection

            batch 0 : o0   |=>  post-projection[i] -> pre-projection[i+1]  |        attention[i+1]                     |=> [o0']
            batch 1 : qkv1 |=>       attention[i]                          | post-projection[i] -> pre-projection[i+1] |=> qkv1'
        """
        q0, k0, v0 = self._forward_pipeline_stage(q1, k1, v1, batches, cur_stage=0)
        q1, k1, v1 = self._forward_pipeline_stage(q0, k0, v0, batches, cur_stage=1)

        return q1, k1, v1


    def forward_first_stage(
        self,
        embeddings: torch.Tensor,
        batches: list[SubBatch]
    ) -> tuple[torch.Tensor]:
        """
        Do the first stage of the pipeline for 2 batches

        batch0 : embeddings0 |=> pre-projection -> attention       |=> [o0]
        batch1 : embeddings1 |=>                  pre-projection   |=> q1, k1, v1
        """
        embeddings = torch.split(embeddings, [batch.iter_width for batch in batches])
        q0, k0, v0 = self._preproj(embeddings[0], batches[0], layer_off=1)
        self._compute_wait_comm_swap() # Here we must make sure all swaps are done before the first attention

        self.events[1].pf_record("stage_s")
        self.events[1].pf_time("lnch_s")
        self._transfer_qkv(q0, k0, v0, batches[0], cur_stage=1)
        q1, k1, v1 = self._preproj(embeddings[1], batches[1], layer_off=1)
        self.events[1].pf_record("linr_e")
        self._attention(q0, k0, v0, batches[0], cur_stage=1)
        self.events[1].pf_time("lnch_e")

        return q1, k1, v1


    def forward_last_stage(
        self,
        q1: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
        k1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        batches: list[SubBatch]
    ) -> torch.Tensor:
        """
        Do the last stage of the pipeline for 2 batches, return the concatenated output

        batch0 : o0   |=> post-projection              |=> [f0]
        batch1 : qkv1 |=> attention -> post-projection |=> [f1]
        """
        self.events[0].pf_record("stage_s")
        self.events[0].pf_time("lnch_s")
        self._transfer_qkv(q1, k1, v1, batches[1], cur_stage=0)
        self._swap_out_blocks(batches[0])
        e0 = self._postproj(batches[0])
        self.events[0].pf_record("linr_e")
        self._attention(q1, k1, v1, batches[1], cur_stage=0)
        self.events[0].pf_time("lnch_e")

        e1 = self._postproj(batches[1])

        return torch.cat((e0, e1))
 
