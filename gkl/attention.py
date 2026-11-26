"""Fused attention forward pass in Triton (FlashAttention-2 style).

softmax(Q K^T / sqrt(d)) V without ever writing the N x N score matrix to
memory: each program owns BLOCK_M query rows, streams K and V through in
BLOCK_N chunks, and keeps a running max, running sum and output accumulator
in registers, rescaling them with the online-softmax update from
csrc/softmax.cu each time the max grows.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _attn_fwd(Q, K, V, O, sm_scale_log2, N,
              stride_qh, stride_qn, stride_kh, stride_kn, stride_vh, stride_vn, stride_oh, stride_on,
              HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)  # batch and head flattened; inputs are contiguous (B*H, N, D)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + bh * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :],
                mask=offs_m[:, None] < N, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    hi = tl.minimum((start_m + 1) * BLOCK_M, N) if CAUSAL else N
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = offs_n[:, None] < N
        k = tl.load(K + bh * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :], mask=kv_mask, other=0.0)
        v = tl.load(V + bh * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :], mask=kv_mask, other=0.0)
        # Scores in log2 space: exp(x) == exp2(x * log2(e)), and exp2 is the native instruction.
        s = tl.dot(q, tl.trans(k)) * sm_scale_log2
        valid = offs_n[None, :] < N
        if CAUSAL:
            valid = valid & (offs_n[None, :] <= offs_m[:, None])
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)  # rescales everything accumulated under the old max
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + bh * stride_oh + offs_m[:, None] * stride_on + offs_d[None, :], acc.to(O.dtype.element_ty),
             mask=offs_m[:, None] < N)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False,
              block_m: int = 64, block_n: int = 64) -> torch.Tensor:
    """q, k, v: (batch, heads, seq, head_dim), contiguous, head_dim a power of two >= 16."""
    b, h, n, d = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k and v must have the same shape")
    if d < 16 or d & (d - 1):
        raise ValueError("head_dim must be a power of two >= 16")
    q, k, v = (t.contiguous().view(b * h, n, d) for t in (q, k, v))
    o = torch.empty_like(q)
    grid = (triton.cdiv(n, block_m), b * h)
    _attn_fwd[grid](q, k, v, o, LOG2E / math.sqrt(d), n,
                    q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                    o.stride(0), o.stride(1),
                    HEAD_DIM=d, BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=causal, num_warps=4, num_stages=2)
    return o.view(b, h, n, d)


def attention_flops(b: int, h: int, n: int, d: int, causal: bool) -> int:
    """Two N x N x D matmuls per head (Q K^T and P V); causal does about half."""
    flops = 4 * b * h * n * n * d
    return flops // 2 if causal else flops
