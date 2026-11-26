"""Every kernel stage against a float64 PyTorch reference. Needs a CUDA GPU and build/libgkl.so."""

import math

import pytest
import torch

from gkl import ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# Square, non-square, sizes that are not tile multiples, and K or N not divisible
# by 4 (forces the vectorized stage onto its scalar path).
MATMUL_SHAPES = [(128, 128, 128), (1, 1, 1), (257, 129, 63), (100, 300, 37), (64, 1000, 8), (513, 515, 1027)]


@pytest.mark.parametrize("stage", sorted(ops.MATMUL_STAGES))
@pytest.mark.parametrize("m,n,k", MATMUL_SHAPES)
def test_matmul_matches_reference(stage, m, n, k):
    if stage == 5 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("TF32 tensor cores need sm_80+")
    gen = torch.Generator(device="cuda").manual_seed(m * n + k)
    a = torch.randn(m, k, device="cuda", generator=gen)
    b = torch.randn(k, n, device="cuda", generator=gen)
    out = ops.matmul(a, b, stage)
    ref = a.double() @ b.double()
    err = ((out.double() - ref).norm() / ref.norm()).item()
    assert err < (5e-3 if stage == 5 else 1e-5)


def test_matmul_rejects_unknown_stage():
    a = torch.ones(4, 4, device="cuda")
    with pytest.raises(ops.KernelError):
        ops.matmul(a, a, 99)


SOFTMAX_SHAPES = [(1, 1), (7, 3), (33, 1000), (128, 1024), (5, 4099), (2, 50000)]


@pytest.mark.parametrize("stage", sorted(ops.SOFTMAX_STAGES))
@pytest.mark.parametrize("rows,cols", SOFTMAX_SHAPES)
def test_softmax_matches_reference(stage, rows, cols):
    gen = torch.Generator(device="cuda").manual_seed(rows * cols)
    x = torch.randn(rows, cols, device="cuda", generator=gen) * 5
    torch.testing.assert_close(ops.softmax(x, stage), torch.softmax(x.double(), -1).float(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("stage", sorted(ops.SOFTMAX_STAGES))
def test_softmax_extreme_logits(stage):
    """Huge logits must not overflow, -inf entries get probability 0, an all -inf row is NaN like PyTorch."""
    x = torch.randn(4, 1024, device="cuda") * 1e4
    x[1, ::3] = -math.inf
    x[2, :1023] = -math.inf  # one finite value: probability 1 there
    x[3] = -math.inf
    out = ops.softmax(x, stage)
    ref = torch.softmax(x.double(), -1).float()
    torch.testing.assert_close(out[:3], ref[:3], rtol=1e-5, atol=1e-6)
    assert out[3].isnan().all() and ref[3].isnan().all()


def test_copy_probe_copies():
    src = torch.randn(1 << 20, device="cuda")
    dst = torch.empty_like(src)
    ops.copy(src, dst)
    assert torch.equal(src, dst)
