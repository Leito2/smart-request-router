"""Run the quality-loop pipeline locally (KFP SubprocessRunner, no cluster).

Usage: python pipelines/run_local.py [--model laya] [--degrade 0.0]
Each step runs as a subprocess of this interpreter; the gate and promote steps call the repo's own CLIs.
Known limit: on Windows KFP's local runner breaks while passing backslash paths to its executor (JSON decode
error), so run it on Linux/macOS/WSL or on a cluster. The promote/rollback logic was exercised directly
(training/promote.py with the MLflow registry) and the pipeline compiles to pipelines/quality_loop.yaml.
"""
import argparse
import sys
from pathlib import Path

from kfp import local

ROOT = Path(__file__).resolve().parent.parent


def shell_safe(path: str) -> str:
    """KFP's local runner builds an `sh` script with unquoted paths: use 8.3 short names with forward
    slashes on Windows so a repo under a folder with spaces still runs."""
    if sys.platform != "win32":
        return Path(path).as_posix()
    import ctypes

    buffer = ctypes.create_unicode_buffer(1024)
    ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, 1024)
    return Path(buffer.value or path).as_posix()
sys.path.insert(0, str(ROOT / "pipelines"))

from quality_loop import quality_loop  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="laya")
    parser.add_argument("--degrade", type=float, default=0.0)
    parser.add_argument("--version", default="1")
    args = parser.parse_args()
    sys.executable = shell_safe(sys.executable)
    local.init(runner=local.SubprocessRunner(use_venv=False))
    run = quality_loop(repo=shell_safe(str(ROOT)), model=args.model, degrade=args.degrade, version=args.version)
    print("pipeline finished:", run)


if __name__ == "__main__":
    main()
