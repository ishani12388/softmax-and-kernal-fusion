import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

# Graph 1: execution time
plt.figure(figsize=(8, 5))
plt.bar(df["method"], df["time_ms"])
plt.title("Softmax Execution Time")
plt.xlabel("Implementation")
plt.ylabel("Time (ms)")
plt.tight_layout()
plt.savefig("softmax_execution_time.png", dpi=150)
plt.close()

# Graph 2: speedup relative to CPU
plt.figure(figsize=(8, 5))
plt.bar(df["method"], df["speedup"])
plt.title("Softmax Speedup Relative to CPU")
plt.xlabel("Implementation")
plt.ylabel("Speedup (x)")
plt.tight_layout()
plt.savefig("softmax_speedup.png", dpi=150)
plt.close()

print("Saved softmax_execution_time.png and softmax_speedup.png")