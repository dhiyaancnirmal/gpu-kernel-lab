from gkl import metrics


def test_fp32_peak_matches_a10_datasheet():
    # NVIDIA A10 datasheet: 31.2 TFLOPS FP32, 72 SMs, 1695 MHz boost.
    assert abs(metrics.fp32_peak_flops(72, 1_695_000, (8, 6)) / 1e12 - 31.2) < 0.1


def test_matmul_intensity_grows_with_size():
    # 2n^3 FLOPs over 3 n^2 fp32 matrices = n / 6 FLOP per byte.
    n = 4096
    assert metrics.matmul_flops(n, n, n) / metrics.matmul_bytes(n, n, n) == n / 6


def test_roofline_is_bandwidth_bound_below_ridge_and_flat_above():
    peak, bw = 30e12, 600e9  # ridge at 50 FLOP/byte
    assert metrics.attainable(1, peak, bw) == 600e9
    assert metrics.attainable(500, peak, bw) == peak
