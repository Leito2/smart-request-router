"""Kubeflow Pipelines `quality-loop` (PLAN §12).

evaluate -> gate -> (regression? retrain -> gate again) -> promote. Every step shells out to the repo's own
CLIs, so the pipeline is a thin orchestration layer and the logic stays testable.
Compile: python pipelines/quality_loop.py     Run locally: python pipelines/run_local.py
"""
from kfp import compiler, dsl


@dsl.component(base_image="python:3.12-slim")
def gate_step(repo: str, model: str, degrade: float) -> str:
    """Run the regression gate (eval/ci_gate.py); returns 'pass' or 'fail'."""
    import os
    import subprocess
    import sys

    command = [sys.executable, f"{repo}/eval/ci_gate.py", "--model", model, "--degrade", str(degrade)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([repo, f"{repo}/eval", f"{repo}/training"])}
    result = subprocess.run(command, capture_output=True, text=True, cwd=repo, env=env)
    print(result.stdout[-2000:], result.stderr[-1000:])
    return "pass" if result.returncode == 0 else "fail"


@dsl.component(base_image="python:3.12-slim")
def retrain_step(repo: str, train_cmd: str) -> str:
    """Retrain (feedback merge + the train command)."""
    import subprocess

    result = subprocess.run(train_cmd, shell=True, cwd=repo, capture_output=True, text=True)
    print(result.stdout[-2000:], result.stderr[-1000:])
    return "ok" if result.returncode == 0 else "failed"


@dsl.component(base_image="python:3.12-slim")
def promote_step(repo: str, version: str) -> str:
    """Promote the candidate to `production` (the old one is kept as `previous`)."""
    import subprocess
    import sys

    command = [sys.executable, f"{repo}/training/promote.py", "promote", "--version", version, "--gate-passed"]
    result = subprocess.run(command, capture_output=True, text=True, cwd=repo)
    print(result.stdout[-500:], result.stderr[-500:])
    return "promoted" if result.returncode == 0 else "promotion_failed"


@dsl.pipeline(name="quality-loop", description="Gate a candidate; on regression retrain and gate again")
def quality_loop(repo: str, model: str = "rules", degrade: float = 0.0, version: str = "1",
                 train_cmd: str = "echo retraining"):
    first = gate_step(repo=repo, model=model, degrade=degrade)
    with dsl.If(first.output == "pass"):
        promote_step(repo=repo, version=version)
    with dsl.If(first.output == "fail"):
        retrained = retrain_step(repo=repo, train_cmd=train_cmd)
        second = gate_step(repo=repo, model=model, degrade=0.0).after(retrained)
        with dsl.If(second.output == "pass"):
            promote_step(repo=repo, version=version)


if __name__ == "__main__":
    compiler.Compiler().compile(quality_loop, "pipelines/quality_loop.yaml")
    print("compiled pipelines/quality_loop.yaml")
