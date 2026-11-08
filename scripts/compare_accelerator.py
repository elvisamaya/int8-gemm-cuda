#!/usr/bin/env python3
"""Put the GPU numbers next to my 4x4 systolic-array accelerator and the CPU.

  python3 scripts/compare_accelerator.py results

The accelerator figures come from github.com/elvisamaya/systolic-accelerator
(sky130 synthesis and OpenLane layout); its throughput is the theoretical peak
(16 MACs per cycle), not a measurement. GPU and CPU rows are measured here.
Writes results/comparison.md.
"""
import csv
import json
import sys
from pathlib import Path

ACCEL_MACS_PER_CYCLE = 16              # 4x4 array of int8 MACs
ACCEL_CLOCKS_MHZ = {"pre-layout estimate": 263, "post-layout": 100}
ACCEL_DIE_UM2 = 573_361                # OpenLane die area (sky130)


def best(path, size):
    rows = []
    if path.exists():
        with open(path) as f:
            rows = [r for r in csv.DictReader(f)
                    if r["gops"] and r["wrong"] != "error" and int(r["M"]) == size]
    return max(rows, key=lambda r: float(r["gops"])) if rows else None


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    device = json.loads((d / "device.json").read_text())
    sizes = sorted({int(r["M"]) for r in csv.DictReader(open(d / "results.csv")) if r["gops"]})
    size = max(s for s in sizes if s <= 2048) if any(s <= 2048 for s in sizes) else sizes[-1]

    lines = [f"| Platform | Implementation | Throughput (GOPS) | Notes |",
             "|---|---|---:|---|"]
    gpu_rows = {}
    with open(d / "results.csv") as f:
        for r in csv.DictReader(f):
            if r["gops"] and r["wrong"] != "error" and int(r["M"]) == size:
                gpu_rows[r["kernel"]] = float(r["gops"])
    for k in ("naive", "dp4a", "tensor_core", "cublas"):
        if k in gpu_rows:
            lines.append(f"| {device['name']} | {k} | {gpu_rows[k]:,.0f} | measured, {size}^3 |")
    cpu = best(d / "cpu_baseline.csv", size)
    if cpu:
        lines.append(f"| Colab CPU | {cpu['kernel']} | {float(cpu['gops']):,.1f} | measured, {size}^3 |")
    for label, mhz in ACCEL_CLOCKS_MHZ.items():
        gops = ACCEL_MACS_PER_CYCLE * 2 * mhz / 1000
        lines.append(f"| 4x4 systolic array (sky130) | {label}, {mhz} MHz | {gops:.1f} | "
                     f"peak, {ACCEL_DIE_UM2 / 1e6:.2f} mm^2 die |")

    text = (f"# GPU vs CPU vs my accelerator\n\n" + "\n".join(lines) + "\n\n"
            "1 multiply-accumulate = 2 ops. The accelerator is a 16-MAC teaching design on an "
            "open 130 nm process; the GPU has thousands of MAC units on a 12 nm process, so the "
            "gap is expected. The interesting comparison is how each one keeps its MACs fed.\n")
    (d / "comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
