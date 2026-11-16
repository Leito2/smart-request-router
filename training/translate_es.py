"""Translates a stratified sample of Banking77 into Spanish with a local Ollama model ($0, PLAN §4.2).

Resumable: finished origins are appended to training/data/translations.jsonl and skipped on restart.
Usage: python training/translate_es.py [--model gemma3:4b] [--per-intent-train 20] [--per-intent-test 3]
"""
import argparse
import json
import random
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from routercore.adapters.jsonl import read_jsonl

DATA = Path(__file__).parent / "data"
OLLAMA = "http://localhost:11434/api/chat"
SYSTEM = (
    "You translate bank customer messages from English to natural Latin American Spanish. Translate exactly: "
    "do not add, remove or soften anything, no extra questions, no exclamations unless the original has "
    "them. Output only the translation."
)
SHOTS = [("I am still waiting on my card?", "¿Sigo esperando mi tarjeta?"),
         ("my card was not stolen, I just want to change my PIN",
          "mi tarjeta no fue robada, solo quiero cambiar mi PIN")]


def clean_translation(raw: str, source: str) -> str | None:
    """Return the usable translation, or None if the model answered with something unusable."""
    text = raw.strip().strip('"').strip()
    if not text or "\n" in text or text.casefold() == source.casefold():
        return None
    if not 0.4 <= len(text) / max(len(source), 1) <= 3.0:
        return None
    return text


def sample(rows: list[dict], per_intent: int, seed: int) -> list[dict]:
    by_intent = defaultdict(list)
    for r in rows:
        by_intent[r["intent"]].append(r)
    rng = random.Random(seed)
    picked = []
    for intent in sorted(by_intent):
        rows_for_intent = by_intent[intent]
        picked += rng.sample(rows_for_intent, min(per_intent, len(rows_for_intent)))
    return picked


def load_jsonl(path: Path) -> list[dict]:
    return read_jsonl(path)


def translate(text: str, model: str, tries: int = 3) -> str | None:
    messages = [{"role": "system", "content": SYSTEM}]
    for en, es in SHOTS:
        messages += [{"role": "user", "content": en}, {"role": "assistant", "content": es}]
    messages.append({"role": "user", "content": text})
    body = json.dumps({"model": model, "stream": False, "think": False, "messages": messages,
                       "options": {"temperature": 0, "num_ctx": 512, "num_predict": 80}}).encode()
    request = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return clean_translation(json.load(response)["message"]["content"], text)
        except OSError:
            time.sleep(5 * (attempt + 1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--per-intent-train", type=int, default=20)
    parser.add_argument("--per-intent-test", type=int, default=3)
    args = parser.parse_args()

    train, test = load_jsonl(DATA / "raw" / "train.jsonl"), load_jsonl(DATA / "raw" / "test.jsonl")
    todo = sample(train, args.per_intent_train, 0) + sample(test, args.per_intent_test, 0)
    out = DATA / "translations.jsonl"
    done = {r["origin_id"] for r in load_jsonl(out)} if out.exists() else set()
    started = time.time()
    with out.open("a", encoding="utf-8") as sink:
        for i, row in enumerate(todo):
            if row["origin_id"] in done:
                continue
            es = translate(row["text"], args.model)
            if es:
                sink.write(json.dumps({"origin_id": row["origin_id"], "es": es}, ensure_ascii=False) + "\n")
                sink.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)} {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
