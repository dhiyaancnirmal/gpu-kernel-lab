"""Run every kernel stage on the GPU, check it against PyTorch, time it, and write a results file.

    python -m gkl.run --out results/a10.json          # full run
    python -m gkl.run --out /tmp/smoke.json --quick   # small sizes only
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import torch

from gkl import metrics, ops
from gkl.bench import cuda_clock, measure

MATMUL_SIZES = [1024, 2048, 4096]
SOFTMAX_SHAPES = [(4096, 256), (4096, 1024), (4096, 4096), (4096, 16384)]
QUICK_MATMUL_SIZES = [512]
QUICK_SOFTMAX_SHAPES = [(1024, 1024)]

# Norm-wise relative error, ||C - ref|| / ||ref||, against a float64 reference.
FP32_TOL = 1e-5
TF32_TOL = 5e-3


def rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.double() - ref).norm() / ref.norm()).item()


def env_info() -> dict:
    props = torch.cuda.get_device_properties(0)
    info = ops.device_info()
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.strip()
    return {
        "gpu": props.name,
        "sm_count": info["sm_count"],
        "clock_mhz": info["clock_khz"] / 1e3,
        "compute_capability": f"{info['cc_major']}.{info['cc_minor']}",
        "l2_bytes": props.L2_cache_size,
        "memory_bytes": props.total_memory,
        "driver": driver,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
    }


def bench_matmul(sizes: list[int], clock) -> list[dict]:
    rows = []
    for n in sizes:
        gen = torch.Generator(device="cuda").manual_seed(n)
        a = torch.randn(n, n, device="cuda", generator=gen)
        b = torch.randn(n, n, device="cuda", generator=gen)
        ref = a.double() @ b.double()
        out = torch.empty(n, n, device="cuda")
        flops = metrics.matmul_flops(n, n, n)

        cases = [(name, s, (lambda s=s: ops.matmul(a, b, s, out)), TF32_TOL if s == 5 else FP32_TOL)
                 for s, name in ops.MATMUL_STAGES.items()]
        for name, tf32 in (("cublas_fp32", False), ("cublas_tf32", True)):
            def run(tf32=tf32):
                torch.backends.cuda.matmul.allow_tf32 = tf32
                torch.matmul(a, b, out=out)
            cases.append((name, None, run, TF32_TOL if tf32 else FP32_TOL))

        for name, stage, fn, tol in cases:
            out.zero_()
            fn()
            torch.cuda.synchronize()
            err = rel_err(out, ref)
            t = measure(fn, clock)
            rows.append({
                "kernel": name, "stage": stage, "m": n, "n": n, "k": n,
                "median_ms": t["median_ms"], "min_ms": t["min_ms"], "reps": t["reps"],
                "tflops": flops / (t["median_ms"] * 1e-3) / 1e12,
                "rel_err": err, "tolerance": tol, "correct": err <= tol,
            })
            print(f"matmul {n:5d} {name:18s} {t['median_ms']:9.3f} ms {rows[-1]['tflops']:7.2f} TFLOP/s err {err:.1e}")
        torch.backends.cuda.matmul.allow_tf32 = False
    return rows


def bench_softmax(shapes: list[tuple[int, int]], clock) -> list[dict]:
    rows = []
    for r, c in shapes:
        gen = torch.Generator(device="cuda").manual_seed(r * c)
        x = torch.randn(r, c, device="cuda", generator=gen) * 4
        ref = torch.softmax(x.double(), dim=-1)
        out = torch.empty_like(x)
        nbytes = metrics.softmax_bytes(r, c)
        cases = [(name, s, (lambda s=s: ops.softmax(x, s, out))) for s, name in ops.SOFTMAX_STAGES.items()]
        cases.append(("torch", None, lambda: torch.softmax(x, dim=-1, out=out)))
        for name, stage, fn in cases:
            out.zero_()
            fn()
            torch.cuda.synchronize()
            err = (out.double() - ref).abs().max().item()
            t = measure(fn, clock)
            rows.append({
                "kernel": name, "stage": stage, "rows": r, "cols": c,
                "median_ms": t["median_ms"], "min_ms": t["min_ms"], "reps": t["reps"],
                "gbps": nbytes / (t["median_ms"] * 1e-3) / 1e9,
                "tflops": metrics.softmax_flops(r, c) / (t["median_ms"] * 1e-3) / 1e12,
                "max_abs_err": err, "tolerance": 1e-6, "correct": err <= 1e-6,
            })
            print(f"softmax {r}x{c:<6d} {name:13s} {t['median_ms']:8.3f} ms {rows[-1]['gbps']:7.1f} GB/s err {err:.1e}")
    return rows


def ceilings(env: dict, clock) -> dict:
    n = 64 * 1024 * 1024  # 256 MiB per buffer, far beyond any L2
    src = torch.randn(n, device="cuda")
    dst = torch.empty_like(src)
    t = measure(lambda: ops.copy(src, dst), clock)
    if not torch.equal(src, dst):
        raise RuntimeError("copy kernel produced wrong output")
    major, minor = (int(v) for v in env["compute_capability"].split("."))
    return {
        "dram_gbps_measured": 2 * n * 4 / (t["median_ms"] * 1e-3) / 1e9,
        "fp32_tflops_peak": metrics.fp32_peak_flops(env["sm_count"], int(env["clock_mhz"] * 1e3), (major, minor)) / 1e12,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    env = env_info()
    clock = cuda_clock(flush_bytes=4 * env["l2_bytes"])
    result = {
        "env": env,
        "ceilings": ceilings(env, clock),
        "matmul": bench_matmul(QUICK_MATMUL_SIZES if args.quick else MATMUL_SIZES, clock),
        "softmax": bench_softmax(QUICK_SOFTMAX_SHAPES if args.quick else SOFTMAX_SHAPES, clock),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    bad = [r["kernel"] for r in result["matmul"] + result["softmax"] if not r["correct"]]
    print(f"wrote {args.out}; incorrect: {bad or 'none'}")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
