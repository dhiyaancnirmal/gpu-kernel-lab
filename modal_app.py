"""Build, test and benchmark on a Modal GPU, then write the results file locally.

    modal run modal_app.py --quick      # smoke run, small sizes (about 2 GPU minutes)
    modal run modal_app.py              # full run, writes results/a10.json
"""

from pathlib import Path

import modal

GPU = "A10"
ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("make", "binutils")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy", "matplotlib", "pytest")
    .add_local_dir(ROOT, "/lab", ignore=[".venv", "build", ".git", "**/__pycache__", "results"])
)
app = modal.App("gpu-kernel-lab", image=image)


@app.function(gpu=GPU, timeout=15 * 60)
def run(quick: bool) -> dict:
    import json
    import subprocess

    def sh(*cmd: str) -> str:
        p = subprocess.run(cmd, cwd="/lab", capture_output=True, text=True)
        print(p.stdout[-4000:], p.stderr[-4000:])
        if p.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed with {p.returncode}")
        return p.stdout

    sh("nvidia-smi")
    sh("make", "ARCHS=-gencode arch=compute_86,code=sm_86")
    tests = sh("python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests")
    sh("python", "-m", "gkl.run", "--out", "/tmp/results.json", *(["--quick"] if quick else []))
    results = json.loads(Path("/tmp/results.json").read_text())
    results["tests"] = tests.strip().splitlines()[-1]
    return results


@app.local_entrypoint()
def main(quick: bool = False, out: str = "results/a10.json"):
    import json

    results = run.remote(quick)
    path = ROOT / (out if not quick else "build/smoke.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {path}; tests: {results['tests']}")
