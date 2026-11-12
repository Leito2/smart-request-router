"""Export Laya to ONNX, quantize to int8 and measure what that costs (PLAN §4.6).

Usage: python training/onnx_parity.py [--checkpoint models/laya_ft] [--limit 300]
1. export fp32 + int8 with Laya's own exporter (training/vendor/laya_export_onnx.py) into models/onnx/
2. top-class agreement of each ONNX graph with the PyTorch model on validation rows (gate: >= 99.5%)
3. CPU latency per message at batch sizes 1, 8 and 16 (p50 / p95)
Writes docs/results/onnx_parity.json. Laya warns that int8 can damage calibrated probabilities: when agreement
falls below the gate, the fp32 ONNX graph is the one to ship, and the report says so.
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE = 0.995


def agreement(reference: list[str], other: list[str]) -> float:
    return sum(a == b for a, b in zip(reference, other, strict=True)) / len(reference)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def top(result: dict, question: str) -> str:
    probabilities = result["answers"][question]["probabilities"]
    return max(probabilities, key=probabilities.__getitem__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="convaiinnovations/laya-multilingual")
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()
    out_dir = ROOT / "models" / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32, int8 = out_dir / "laya.onnx", out_dir / "laya.int8.onnx"

    exporter = str(ROOT / "training/vendor/laya_export_onnx.py")
    subprocess.run([sys.executable, exporter, "--model", args.checkpoint, "--output", str(fp32),
                    "--quantize", "--no-verify"], check=True)  # the agreement gate below is the verdict

    import laya
    from laya.onnx_agent import ONNXAgent

    from routercore.adapters.laya_model import build_questions
    from routercore.adapters.route_map_config import load_route_map

    questions = build_questions(load_route_map(ROOT / "config" / "route_map.yaml").routes)
    rows = [json.loads(line) for line in (ROOT / "training/data/dataset.jsonl").read_text(
        encoding="utf-8").splitlines() if line]
    texts = [r["text"] for r in random.Random(0).sample([r for r in rows if r["split"] == "val"], args.limit)]

    torch_agent = laya.load(args.checkpoint, device="cpu")
    reference = torch_agent.predict_batch(texts, questions, batch_size=16)
    report: dict = {"checkpoint": args.checkpoint, "rows": len(texts), "gate": GATE}
    for name, path in (("fp32", fp32), ("int8", int8)):
        agent = ONNXAgent(args.checkpoint, onnx_path=str(path))
        results = agent.predict_batch(texts, questions, batch_size=16)
        entry = {q: round(agreement([top(r, q) for r in reference], [top(r, q) for r in results]), 4)
                 for q in ("route", "urgency")}
        entry["passes_gate"] = all(entry[q] >= GATE for q in ("route", "urgency"))
        weights = path.with_name(path.name + ".data")  # fp32 graph keeps its weights beside the .onnx
        total = path.stat().st_size + (weights.stat().st_size if weights.exists() else 0)
        entry["size_mb"] = round(total / 1e6, 1)
        for batch in (1, 8, 16):
            samples = []
            for start in range(0, min(len(texts), 64), batch):
                chunk = texts[start:start + batch]
                began = time.perf_counter()
                agent.predict_batch(chunk, questions, batch_size=batch)
                samples.append((time.perf_counter() - began) * 1000 / len(chunk))
            entry[f"batch{batch}_ms_per_msg"] = {"p50": round(percentile(samples, 0.5), 1),
                                                 "p95": round(percentile(samples, 0.95), 1)}
        report[name] = entry
    report["recommendation"] = "int8" if report["int8"]["passes_gate"] else "fp32"
    out = ROOT / "docs" / "results" / "onnx_parity.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
