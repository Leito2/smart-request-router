"""Downloads Banking77 (CC BY 4.0, PolyAI) from the Hugging Face Hub into training/data/raw/*.jsonl.

Two small parquet files from the Hub CDN (the datasets-server rows API was too flaky). Needs the `data`
dependency group: `uv sync --group data`. Output rows: origin_id, text, intent.
Intent names come from the ClassLabel stored in the parquet schema metadata.
"""
import json
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

REPO = "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/data"
RAW = Path(__file__).parent / "data" / "raw"


def _download(url: str, target: Path, tries: int = 6) -> None:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                target.write_bytes(response.read())
            return
        except OSError:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}")


def _intent_names(parquet: Path) -> list[str]:
    meta = json.loads(pq.read_schema(parquet).metadata[b"huggingface"])
    return meta["info"]["features"]["label"]["names"]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        parquet = RAW / f"{split}.parquet"
        _download(f"{REPO}/{split}-00000-of-00001.parquet", parquet)
        names = _intent_names(parquet)
        table = pq.read_table(parquet).to_pylist()
        rows = [{"origin_id": f"{split[:2]}-{i}", "text": r["text"], "intent": names[r["label"]]}
                for i, r in enumerate(table)]
        (RAW / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        print(split, len(rows))


if __name__ == "__main__":
    main()
