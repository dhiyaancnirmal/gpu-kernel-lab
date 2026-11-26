"""Fused attention against PyTorch's reference. Runs on a GPU (fp16) or in
Triton's CPU interpreter (fp32, small shapes)."""

import pytest
import torch

pytest.importorskip("triton")
from gkl.attention import attention  # noqa: E402

CUDA = torch.cuda.is_available()
DEVICE, DTYPE, TOL = ("cuda", torch.float16, 2e-3) if CUDA else ("cpu", torch.float32, 1e-5)

# Sequence lengths that are and are not block multiples, including shorter than one block.
SHAPES = [(1, 1, 64, 16), (2, 3, 100, 32), (1, 2, 13, 64), (1, 1, 300, 16)] + ([(2, 8, 1031, 64), (1, 4, 2048, 128)] if CUDA else [])


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("b,h,n,d", SHAPES)
def test_attention_matches_reference(b, h, n, d, causal):
    gen = torch.Generator(device=DEVICE).manual_seed(n * d)
    q, k, v = (torch.randn(b, h, n, d, device=DEVICE, dtype=DTYPE, generator=gen) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)
    torch.testing.assert_close(attention(q, k, v, causal=causal).float(), ref, rtol=TOL, atol=TOL)


def test_attention_rejects_bad_head_dim():
    q = torch.zeros(1, 1, 8, 24, device=DEVICE, dtype=DTYPE)
    with pytest.raises(ValueError):
        attention(q, q, q)
