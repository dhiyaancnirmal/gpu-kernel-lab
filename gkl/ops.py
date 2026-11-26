"""Python entry points for the CUDA kernels in libgkl.so, called through ctypes."""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import torch

LIB_PATH = Path(os.environ.get("GKL_LIB", Path(__file__).resolve().parent.parent / "build" / "libgkl.so"))

MATMUL_STAGES = {
    0: "naive",
    1: "coalesced",
    2: "smem_tiled",
    3: "register_blocked",
    4: "vectorized",
    5: "tensor_core_tf32",
}
SOFTMAX_STAGES = {0: "naive", 1: "block_reduce", 2: "online"}

_ERRORS = {-1: "unknown stage", -2: "unsupported on this GPU or shape"}


class KernelError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _lib() -> ctypes.CDLL:
    if not LIB_PATH.exists():
        raise FileNotFoundError(f"{LIB_PATH} not found; run `make` first")
    lib = ctypes.CDLL(str(LIB_PATH))
    p, i, ll = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
    lib.gkl_matmul.argtypes = [i, i, i, i, p, p, p, p]
    lib.gkl_softmax.argtypes = [i, i, i, p, p, p]
    lib.gkl_copy.argtypes = [p, p, ll, p]
    ip = ctypes.POINTER(ctypes.c_int)
    lib.gkl_device_info.argtypes = [ip, ip, ip, ip]
    for f in (lib.gkl_matmul, lib.gkl_softmax, lib.gkl_copy, lib.gkl_device_info):
        f.restype = ctypes.c_int
    return lib


def _check(code: int) -> None:
    if code != 0:
        raise KernelError(_ERRORS.get(code, f"CUDA error {code}"))


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def _require(t: torch.Tensor, ndim: int) -> None:
    if not (t.is_cuda and t.dtype == torch.float32 and t.dim() == ndim and t.is_contiguous()):
        raise ValueError(f"expected a contiguous CUDA float32 tensor with {ndim} dims")


def matmul(a: torch.Tensor, b: torch.Tensor, stage: int, out: torch.Tensor | None = None) -> torch.Tensor:
    _require(a, 2)
    _require(b, 2)
    (m, k), (k2, n) = a.shape, b.shape
    if k != k2:
        raise ValueError(f"inner dimensions differ: {a.shape} @ {b.shape}")
    if out is None:
        out = torch.empty(m, n, device=a.device, dtype=a.dtype)
    _check(_lib().gkl_matmul(stage, m, n, k, a.data_ptr(), b.data_ptr(), out.data_ptr(), _stream()))
    return out


def softmax(x: torch.Tensor, stage: int, out: torch.Tensor | None = None) -> torch.Tensor:
    _require(x, 2)
    rows, cols = x.shape
    if out is None:
        out = torch.empty_like(x)
    _check(_lib().gkl_softmax(stage, rows, cols, x.data_ptr(), out.data_ptr(), _stream()))
    return out


def copy(src: torch.Tensor, dst: torch.Tensor) -> None:
    _check(_lib().gkl_copy(src.data_ptr(), dst.data_ptr(), src.numel(), _stream()))


def device_info() -> dict[str, int]:
    vals = [ctypes.c_int() for _ in range(4)]
    _check(_lib().gkl_device_info(*[ctypes.byref(v) for v in vals]))
    sm, clock_khz, major, minor = (v.value for v in vals)
    return {"sm_count": sm, "clock_khz": clock_khz, "cc_major": major, "cc_minor": minor}
