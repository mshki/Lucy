#!/usr/bin/env python3
"""Plot v0.5 GPU CFR throughput numbers from the sweep logs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Sweep 1 — batch size scaling at 1000 iters fixed.
batch  = [256,        1024,        4096,         16384,        65536]
gpu_traj_s = [689633, 3.22861e+06, 1.20944e+07,  3.07972e+07,  5.58637e+07]

# Sweep 2 — iterations fixed batch=8192.
iters    = [100, 1000, 10000, 50000]
gpu_iter_traj_s = [1.80251e+07, 2.06277e+07, 2.19119e+07, 2.19734e+07]

# CPU baseline points.
cpu_outcome  = {"iters": [100, 1000, 10000], "traj_s": [1333, 11111, 15385]}
cpu_external = {"iters": [1000], "wallclock": [0.41]}  # 1000 iter ≈ 2000 traj

# Reference: best CPU outcome-sampling traj/s.
best_cpu = max(cpu_outcome["traj_s"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.loglog(batch, gpu_traj_s, "o-", color="#0c8", label="GPU (NVIDIA A16, 1000 iters)", linewidth=2, markersize=8)
ax.axhline(best_cpu, color="#d33", linestyle="--", linewidth=1.5, label=f"Best CPU outcome-sampling ({best_cpu:,} traj/s)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Trajectories per kernel launch (batch size)")
ax.set_ylabel("Throughput (trajectories / second)")
ax.set_title("GPU throughput vs batch size\nLucy device-side outcome-sampling MCCFR, HU NLHE FCPA")
ax.grid(True, which="both", linestyle=":", alpha=0.4)
ax.legend(loc="lower right")
# Annotate the speedup at each point.
for b, t in zip(batch, gpu_traj_s):
    speedup = t / best_cpu
    ax.annotate(f"{speedup:,.0f}×", (b, t), textcoords="offset points",
                xytext=(8, -3), fontsize=9, color="#0c8")

ax = axes[1]
labels = ["CPU outcome\n(15.4K traj/s)", "GPU B=4096\n(12M traj/s)",
          "GPU B=8192\n(22M traj/s)", "GPU B=65536\n(56M traj/s)"]
values = [15385, 1.20944e+07, 2.19119e+07, 5.58637e+07]
colors = ["#d33", "#5b9", "#3a8", "#1a6"]
bars = ax.bar(labels, values, color=colors)
ax.set_yscale("log")
ax.set_ylabel("Throughput (trajectories / second)")
ax.set_title("CPU vs GPU throughput on HU NLHE FCPA")
for bar, v in zip(bars, values):
    sp = v / 15385
    ax.text(bar.get_x() + bar.get_width()/2, v * 1.3,
            f"{sp:,.0f}× CPU", ha="center", fontsize=10)
ax.grid(True, axis="y", linestyle=":", alpha=0.4)

fig.tight_layout()
fig.savefig("v05_gpu_throughput.png", dpi=150)
print("saved v05_gpu_throughput.png")
