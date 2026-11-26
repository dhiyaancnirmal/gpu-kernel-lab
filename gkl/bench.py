"""Timing loop shared by every benchmark."""

from __future__ import annotations

import statistics
from typing import Callable

# One timed run of the kernel under test, returning milliseconds.
Clock = Callable[[Callable[[], None]], float]


def measure(fn: Callable[[], None], clock: Clock, warmup: int = 2, budget_ms: float = 1000.0,
            min_reps: int = 5, max_reps: int = 100) -> dict[str, float]:
    """Warm up, then time `fn` repeatedly and report median and min.

    The rep count fits roughly `budget_ms` of kernel time, so slow stages
    (naive matmul at 4096) do not dominate GPU minutes.
    """
    for _ in range(warmup):
        clock(fn)
    first = clock(fn)
    reps = int(budget_ms / first) if first > 0 else max_reps
    reps = max(min_reps, min(max_reps, reps))
    times = [first] + [clock(fn) for _ in range(reps - 1)]
    return {"median_ms": statistics.median(times), "min_ms": min(times), "reps": len(times)}


def cuda_clock(flush_bytes: int) -> Clock:
    """CUDA-event clock that evicts L2 before each run, so memory-bound kernels hit DRAM."""
    import torch

    scratch = torch.empty(flush_bytes // 4, dtype=torch.float32, device="cuda")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def clock(fn: Callable[[], None]) -> float:
        scratch.zero_()
        start.record()
        fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    return clock
