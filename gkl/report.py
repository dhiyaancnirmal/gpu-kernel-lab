"""Turn a results file into the README tables and the roofline chart.

    python -m gkl.report results/a10.json            # rewrite README tables and chart
    python -m gkl.report results/a10.json --check    # fail if README is out of date
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from gkl import metrics

START, END = "<!-- results:start -->", "<!-- results:end -->"

MATMUL_LABELS = {
    "naive": "0. Naive",
    "coalesced": "1. Coalesced",
    "smem_tiled": "2. Shared-memory tiles",
    "register_blocked": "3. Register blocking (8x8 per thread)",
    "vectorized": "4. Vectorized float4 loads",
    "tensor_core_tf32": "5. Tensor cores (TF32 WMMA)",
    "cublas_fp32": "cuBLAS FP32 (`torch.matmul`)",
    "cublas_tf32": "cuBLAS TF32 (`torch.matmul`)",
}
SOFTMAX_LABELS = {
    "naive": "0. Naive (thread per row)",
    "block_reduce": "1. Block per row, warp shuffles",
    "online": "2. Online softmax, float4",
    "torch": "`torch.softmax`",
}


def render(results: dict) -> str:
    env, ceil = results["env"], results["ceilings"]
    out = [
        f"Measured on an NVIDIA {env['gpu']} ({env['sm_count']} SMs, compute capability "
        f"{env['compute_capability']}), driver {env['driver']}, PyTorch {env['torch']} (CUDA {env['torch_cuda']}). "
        f"Times are medians of CUDA-event-timed runs with L2 flushed before each run.",
        "",
        f"Ceilings: {ceil['dram_gbps_measured']:.0f} GB/s DRAM bandwidth (measured with a float4 copy kernel), "
        f"{ceil['fp32_tflops_peak']:.1f} TFLOP/s FP32 peak (SMs x FP32 lanes x 2 x {env['clock_mhz']:.0f} MHz).",
        "",
    ]

    mm = results["matmul"]
    sizes = list(dict.fromkeys(r["m"] for r in mm))
    big = max(sizes)
    by = {(r["kernel"], r["m"]): r for r in mm}
    ref = by[("cublas_fp32", big)]["tflops"]
    out.append(f"**Matmul, square fp32, TFLOP/s** (higher is better; last column relative to cuBLAS FP32 at {big})")
    out.append("")
    out.append("| Kernel | " + " | ".join(f"{s}" for s in sizes) + f" | vs cuBLAS FP32 @ {big} |")
    out.append("|---|" + "---:|" * (len(sizes) + 1))
    for k in dict.fromkeys(r["kernel"] for r in mm):
        cells = [f"{by[(k, s)]['tflops']:.2f}" for s in sizes]
        out.append(f"| {MATMUL_LABELS.get(k, k)} | " + " | ".join(cells) + f" | {100 * by[(k, big)]['tflops'] / ref:.0f}% |")
    out.append("")

    sm = results["softmax"]
    shapes = list(dict.fromkeys((r["rows"], r["cols"]) for r in sm))
    by = {(r["kernel"], r["rows"], r["cols"]): r for r in sm}
    out.append(f"**Softmax over rows, fp32, effective GB/s** (one read + one write; higher is better)")
    out.append("")
    out.append("| Kernel | " + " | ".join(f"{r}x{c}" for r, c in shapes) + " |")
    out.append("|---|" + "---:|" * len(shapes))
    for k in dict.fromkeys(r["kernel"] for r in sm):
        cells = [f"{by[(k, r, c)]['gbps']:.0f}" for r, c in shapes]
        out.append(f"| {SOFTMAX_LABELS.get(k, k)} | " + " | ".join(cells) + " |")
    out.append("")
    ok = all(r["correct"] for r in mm + sm)
    out.append(f"Correctness against a float64 PyTorch reference: {'all kernels pass' if ok else 'FAILURES, see results file'}.")
    return "\n".join(out)


def splice(readme: str, table: str) -> str:
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(readme):
        raise ValueError("README has no results markers")
    return pattern.sub(lambda _: f"{START}\n{table}\n{END}", readme)


def roofline(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ceil = results["ceilings"]
    bw, peak = ceil["dram_gbps_measured"] * 1e9, ceil["fp32_tflops_peak"] * 1e12
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = [2 ** (i / 4) for i in range(-12, 45)]
    ax.plot(xs, [metrics.attainable(x, peak, bw) / 1e12 for x in xs], "k-", lw=2,
            label=f"Roofline: {bw / 1e9:.0f} GB/s, {peak / 1e12:.1f} TFLOP/s FP32")

    big = max(r["m"] for r in results["matmul"])
    for r in results["matmul"]:
        if r["m"] != big:
            continue
        x = metrics.matmul_flops(r["m"], r["n"], r["k"]) / metrics.matmul_bytes(r["m"], r["n"], r["k"])
        ax.scatter(x, r["tflops"], s=50, marker="s" if r["stage"] is None else "o", zorder=3)
        ax.annotate(MATMUL_LABELS.get(r["kernel"], r["kernel"]).replace("`", ""), (x, r["tflops"]),
                    textcoords="offset points", xytext=(6, -3), fontsize=7)
    widest = max(r["cols"] for r in results["softmax"])
    for r in results["softmax"]:
        if r["cols"] != widest:
            continue
        x = metrics.SOFTMAX_FLOPS_PER_ELEMENT / (2 * metrics.FP32_BYTES)
        ax.scatter(x, r["tflops"], s=50, marker="s" if r["stage"] is None else "^", zorder=3)
        ax.annotate("softmax " + SOFTMAX_LABELS.get(r["kernel"], r["kernel"]).replace("`", ""), (x, r["tflops"]),
                    textcoords="offset points", xytext=(6, -3), fontsize=7)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP / byte of compulsory DRAM traffic)")
    ax.set_ylabel("Achieved TFLOP/s")
    ax.set_title(f"Roofline, {results['env']['gpu']}: matmul {big}^3, softmax 4096x{widest}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    p.add_argument("--chart", type=Path, default=Path("results/roofline.png"))
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    results = json.loads(args.results.read_text())
    readme = args.readme.read_text()
    updated = splice(readme, render(results))
    if args.check:
        if updated != readme:
            raise SystemExit(f"{args.readme} tables differ from {args.results}; run python -m gkl.report {args.results}")
        return
    args.readme.write_text(updated)
    roofline(results, args.chart)


if __name__ == "__main__":
    main()
