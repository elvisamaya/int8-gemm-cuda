#!/usr/bin/env python3
"""CPU baseline: the same matrix multiply with NumPy on this machine's CPU.

  python3 scripts/cpu_baseline.py results

NumPy has no fast int8 matmul, so two numbers are recorded:
  numpy_int32    integers widened to int32 (exact, but no BLAS - slow)
  numpy_float32  float32 through the BLAS library (fast, the fair CPU number)
Writes results/cpu_baseline.csv with the same columns as results.csv.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np


def bench(fn, budget_s=1.0):
    fn()                                         # warm-up
    t0, n = time.perf_counter(), 0
    while True:
        fn()
        n += 1
        dt = time.perf_counter() - t0
        if dt > budget_s or n >= 20:
            return dt / n


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rows = ["kernel,M,N,K,ms,gops,wrong"]
    print(f"CPU threads available: {os.cpu_count()}")
    for S in (512, 1024, 2048):
        A = rng.integers(-128, 128, (S, S), dtype=np.int8)
        B = rng.integers(-128, 128, (S, S), dtype=np.int8)
        A32, B32 = A.astype(np.int32), B.astype(np.int32)
        Af, Bf = A.astype(np.float32), B.astype(np.float32)
        tests = [("numpy_float32", lambda: Af @ Bf)]
        if S <= 1024:                            # int32 matmul is too slow beyond this
            tests.insert(0, ("numpy_int32", lambda: A32 @ B32))
        for name, fn in tests:
            s = bench(fn)
            gops = 2 * S ** 3 / s / 1e9
            print(f"{name:14s} {S:5d}  {s * 1e3:9.2f} ms  {gops:8.1f} GOPS")
            rows.append(f"{name},{S},{S},{S},{s * 1e3:.4f},{gops:.2f},0")
    (out / "cpu_baseline.csv").write_text("\n".join(rows) + "\n")
    print("wrote", out / "cpu_baseline.csv")


if __name__ == "__main__":
    main()
