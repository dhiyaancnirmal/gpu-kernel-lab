"""Work and traffic models for each kernel, and the roofline bound.

Arithmetic intensity uses compulsory traffic: each input read once and each
output written once. Real kernels move more bytes than that (the naive matmul
far more), so a point's x position is the best case for its problem shape and
its y position is the FLOP/s it actually achieved.
"""

from __future__ import annotations

FP32_BYTES = 4
# FLOPs counted per softmax element: compare (max), subtract, exp, add (sum), multiply.
SOFTMAX_FLOPS_PER_ELEMENT = 5

# FP32 FMA lanes per SM, from the CUDA C Programming Guide arithmetic throughput table.
_FP32_LANES_PER_SM = {(7, 5): 64, (8, 0): 64, (8, 6): 128, (8, 7): 128, (8, 9): 128, (9, 0): 128}


def matmul_flops(m: int, n: int, k: int) -> int:
    return 2 * m * n * k


def matmul_bytes(m: int, n: int, k: int) -> int:
    return FP32_BYTES * (m * k + k * n + m * n)


def softmax_flops(rows: int, cols: int) -> int:
    return SOFTMAX_FLOPS_PER_ELEMENT * rows * cols


def softmax_bytes(rows: int, cols: int) -> int:
    return 2 * FP32_BYTES * rows * cols


def fp32_peak_flops(sm_count: int, clock_khz: int, cc: tuple[int, int]) -> float:
    """Peak FP32 FLOP/s on CUDA cores: SMs x lanes per SM x 2 (an FMA is 2 FLOPs) x clock."""
    lanes = _FP32_LANES_PER_SM.get(cc)
    if lanes is None:
        raise ValueError(f"no FP32 lane count for compute capability {cc}")
    return sm_count * lanes * 2 * clock_khz * 1e3


def attainable(intensity: float, peak_flops: float, bandwidth: float) -> float:
    """Roofline: a kernel with `intensity` FLOP/byte cannot beat min(peak, intensity x bandwidth)."""
    return min(peak_flops, intensity * bandwidth)
