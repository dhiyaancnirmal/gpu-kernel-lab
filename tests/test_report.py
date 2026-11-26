"""The README's numbers must come from the committed results file."""

import json
from pathlib import Path

import pytest

from gkl import report

ROOT = Path(__file__).resolve().parent.parent
RESULTS = sorted((ROOT / "results").glob("*.json"))


@pytest.mark.skipif(not RESULTS, reason="no GPU results committed yet")
def test_readme_tables_match_committed_results():
    readme = (ROOT / "README.md").read_text()
    assert report.splice(readme, report.render(json.loads(RESULTS[0].read_text()))) == readme


def _row(kernel, stage, size, tflops):
    return {"kernel": kernel, "stage": stage, "m": size, "n": size, "k": size, "tflops": tflops, "correct": True}


def test_render_reports_share_of_cublas_at_largest_size():
    results = {
        "env": {"gpu": "Test GPU", "sm_count": 1, "compute_capability": "8.6", "driver": "x", "torch": "x",
                "torch_cuda": "x", "clock_mhz": 1000},
        "ceilings": {"dram_gbps_measured": 500, "fp32_tflops_peak": 30},
        "matmul": [_row("vectorized", 4, 512, 1.0), _row("vectorized", 4, 1024, 10.0),
                   _row("cublas_fp32", None, 512, 2.0), _row("cublas_fp32", None, 1024, 20.0)],
        "softmax": [{"kernel": "online", "stage": 2, "rows": 8, "cols": 8, "gbps": 400.0, "correct": True}],
    }
    table = report.render(results)
    assert "| 4. Vectorized float4 loads | 1.00 | 10.00 | 50% |" in table
    assert "| 2. Online softmax, float4 | 400 |" in table
    assert "all kernels pass" in table


def test_render_flags_incorrect_kernels():
    results = {
        "env": {"gpu": "g", "sm_count": 1, "compute_capability": "8.6", "driver": "x", "torch": "x",
                "torch_cuda": "x", "clock_mhz": 1000},
        "ceilings": {"dram_gbps_measured": 1, "fp32_tflops_peak": 1},
        "matmul": [_row("cublas_fp32", None, 8, 1.0)],
        "softmax": [{"kernel": "online", "stage": 2, "rows": 8, "cols": 8, "gbps": 1.0, "correct": False}],
    }
    assert "FAILURES" in report.render(results)
