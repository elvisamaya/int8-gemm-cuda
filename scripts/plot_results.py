#!/usr/bin/env python3
"""Charts from results/results.csv and results/device.json.

  python3 scripts/plot_results.py results

Writes:
  results/gops_by_kernel.png   throughput of every kernel at the largest size
  results/gops_vs_size.png     how each kernel scales with matrix size
  results/roofline.png         each kernel against the GPU's compute and memory limits
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["naive", "coalesced", "smem_tiled", "reg_tiled", "dp4a", "tensor_core", "cublas"]
COLORS = {
    "naive": "#9aa5b1", "coalesced": "#7b8794", "smem_tiled": "#4c9aff", "reg_tiled": "#2f74d0",
    "dp4a": "#1c4f9c", "tensor_core": "#e8833a", "cublas": "#3d3d3d",
}

# Published NVIDIA T4 peak for int8 on tensor cores (T4 datasheet). Used only
# when the device is a T4; other GPUs get the CUDA-core ceilings only.
T4_TENSOR_INT8_TOPS = 130.0


def load(results_dir):
    rows = []
    with open(results_dir / "results.csv") as f:
        for r in csv.DictReader(f):
            if r["wrong"] == "error" or not r["gops"]:
                continue
            rows.append({"kernel": r["kernel"], "M": int(r["M"]), "N": int(r["N"]), "K": int(r["K"]),
                         "ms": float(r["ms"]), "gops": float(r["gops"]), "wrong": int(r["wrong"])})
    device = json.loads((results_dir / "device.json").read_text())
    return rows, device


def bar_chart(rows, device, out):
    size = max(r["M"] for r in rows)
    data = {r["kernel"]: r["gops"] for r in rows if r["M"] == size}
    names = [k for k in ORDER if k in data]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(names, [data[k] / 1000 for k in names], color=[COLORS[k] for k in names])
    for b, k in zip(bars, names):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{data[k] / 1000:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("TOPS (int8 ops / s, x10^12)")
    ax.set_title(f"int8 GEMM, {size}x{size}x{size} on {device['name']}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def scaling_chart(rows, device, out):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for k in ORDER:
        pts = sorted((r["M"], r["gops"]) for r in rows if r["kernel"] == k)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] / 1000 for p in pts], marker="o", label=k, color=COLORS[k])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("matrix size (M = N = K)")
    ax.set_ylabel("TOPS")
    ax.set_title(f"Throughput vs size on {device['name']}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def roofline(rows, device, out):
    bw = device["memory_bandwidth_gbs"]                                  # GB/s
    cores = device["sm_count"] * 64                                      # 64 int32 lanes per SM (Turing+)
    clock_ghz = device["sm_clock_mhz"] / 1000
    int32_peak = cores * clock_ghz * 2                                   # GOPS: one multiply-add = 2 ops
    dp4a_peak = int32_peak * 4                                           # 4 int8 MACs per dp4a
    ceilings = [("int32 multiply-add (CUDA cores)", int32_peak, "#7b8794"),
                ("dp4a (4 int8 MACs / instr)", dp4a_peak, "#1c4f9c")]
    if "T4" in device["name"]:
        ceilings.append(("tensor cores, int8 (published)", T4_TENSOR_INT8_TOPS * 1000, "#e8833a"))

    size = max(r["M"] for r in rows)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    xs = [2 ** i for i in range(-2, 12)]
    top = max(c[1] for c in ceilings)
    ax.plot(xs, [min(top, bw * x) for x in xs], color="black", lw=1)
    ax.text(0.3, bw * 0.3 * 1.25, f"memory {bw:.0f} GB/s", rotation=0, fontsize=8)
    for label, peak, color in ceilings:
        ax.axhline(peak, color=color, ls="--", lw=1)
        ax.text(xs[-1], peak * 1.06, f"{label}: {peak / 1000:.1f} TOPS", ha="right", fontsize=8, color=color)
    for r in rows:
        if r["M"] != size:
            continue
        M, N, K = r["M"], r["N"], r["K"]
        intensity = 2 * M * N * K / (M * K + K * N + 4 * M * N)          # ops per byte of compulsory traffic
        ax.scatter(intensity, r["gops"], color=COLORS[r["kernel"]], zorder=3, label=r["kernel"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (int8 ops / byte of DRAM traffic)")
    ax.set_ylabel("GOPS")
    ax.set_title(f"Roofline at {size}^3 on {device['name']}")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows, device = load(results_dir)
    if not rows:
        sys.exit("no successful rows in results.csv")
    bad = [r for r in rows if r["wrong"]]
    if bad:
        print("WARNING: some results were wrong:", {(r["kernel"], r["M"]) for r in bad})
    bar_chart(rows, device, results_dir / "gops_by_kernel.png")
    scaling_chart(rows, device, results_dir / "gops_vs_size.png")
    roofline(rows, device, results_dir / "roofline.png")
    print("wrote", ", ".join(str(results_dir / p) for p in
                             ["gops_by_kernel.png", "gops_vs_size.png", "roofline.png"]))


if __name__ == "__main__":
    main()
