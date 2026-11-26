"""Compiles the CUDA sources with nvcc (no GPU needed), treating warnings and
register spills as errors, and checks the C ABI that gkl.ops loads."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NVCC = os.environ.get("NVCC") or shutil.which("nvcc")
ABI = {"gkl_matmul", "gkl_softmax", "gkl_copy", "gkl_device_info"}


@pytest.mark.skipif(NVCC is None, reason="nvcc not found; set NVCC")
def test_library_builds_without_warnings_and_exports_abi(tmp_path):
    out = tmp_path / "libgkl.so"
    flags = (os.environ.get("EXTRA_NVCCFLAGS", "") + " -Werror all-warnings -Xptxas -Werror").strip()
    subprocess.run(["make", f"NVCC={NVCC}", f"OUT={out}", f"EXTRA_NVCCFLAGS={flags}"], cwd=ROOT, check=True)
    symbols = subprocess.run(["nm", "-D", "--defined-only", str(out)], capture_output=True, text=True, check=True).stdout
    exported = {line.split()[-1] for line in symbols.splitlines()}
    assert ABI <= exported
