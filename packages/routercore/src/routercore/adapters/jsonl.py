"""One reader for the JSONL files (dataset, golden set, decisions) that scripts and services share."""
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
