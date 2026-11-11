import sys
from pathlib import Path

import pytest

pytest.importorskip("kfp")                      # in the `mlops` group, not installed by CI


def test_quality_loop_compiles_to_a_pipeline_spec(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from kfp import compiler

    from pipelines.quality_loop import quality_loop

    out = tmp_path / "quality_loop.yaml"
    compiler.Compiler().compile(quality_loop, str(out))
    text = out.read_text(encoding="utf-8")
    assert "gate-step" in text and "retrain-step" in text and "promote-step" in text
