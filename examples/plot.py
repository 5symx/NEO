import matplotlib.pyplot as plt

# X-axis labels
configs = ['6/0', '4/2', '2/4']

# Y-axis values: average E2E times
gpu_latency = [20.7400, 34.5232, 34.3835]
cpu_latency = [17.5078, 33.8515, 44.9266]

# Standard deviations
gpu_std = [0.4179, 0.6963, 0.7447]
cpu_std = [0.3653, 1.8206, 3.3308]

# Plotting
plt.figure(figsize=(8, 5))
plt.errorbar(configs, gpu_latency, yerr=gpu_std, label='GPU Offload', fmt='o-', capsize=5)
plt.errorbar(configs, cpu_latency, yerr=cpu_std, label='CPU Offload', fmt='s--', capsize=5)

# Labels and title
plt.xlabel('decoding batch configuratioin (non-offload / offload)')
plt.ylabel('Average E2E Latency (ms)')
plt.title('E2E Latency Comparison: GPU vs CPU Offload')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save to file
plt.savefig('plot.png', dpi=300)
# plt.show()


