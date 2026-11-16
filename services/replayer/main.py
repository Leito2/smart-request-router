"""Replayer (PLAN §3.1): emits dataset messages to `inbound-messages` at a controlled, open-loop rate.

Usage: python -m services.replayer.main --rate 20 --duration 60 --lang-mix es:0.6,en:0.4 [--set test|hard]
"""
import argparse
import json
import random
import time
import uuid
from pathlib import Path

from routercore.adapters.jsonl import read_jsonl

ROOT = Path(__file__).resolve().parent.parent.parent
SETS = {"test": ROOT / "training/data/dataset.jsonl", "hard": ROOT / "training/golden_set/hard_cases.jsonl"}
CHANNELS = ("chat", "email", "app")


def load_rows(name: str) -> list[dict]:
    rows = read_jsonl(SETS[name])
    return [r for r in rows if r.get("split", "test") == "test"]


def pick_rows(rows: list[dict], count: int, lang_mix: dict[str, float], seed: int) -> list[dict]:
    rng = random.Random(seed)
    picked: list[dict] = []
    for lang, share in lang_mix.items():
        pool = [r for r in rows if r["lang"] == lang]
        picked += [rng.choice(pool) for _ in range(round(count * share))]
    rng.shuffle(picked)
    return picked


def parse_surge(spec: str) -> tuple[str, float, float, float]:
    """'route:factor:at_s:duration_s' -> e.g. payments x5 starting at 120 s for 180 s."""
    route, factor, at_s, duration_s = spec.split(":")
    return route, float(factor), float(at_s), float(duration_s)


def surge_rows(rows: list[dict], route: str, *, factor: float, base_rate: float, at_s: float,
               duration_s: float, seed: int) -> list[tuple[float, dict]]:
    """Extra (offset_s, row) events so `route` runs at `factor` x its normal rate during the window."""
    pool = [r for r in rows if r.get("route") == route]
    rng = random.Random(seed)
    count = int(base_rate * (factor - 1) * duration_s)
    schedule = ((at_s + duration_s * i / count, rng.choice(pool)) for i in range(count))
    return sorted(schedule, key=lambda x: x[0])


def build_events(rows: list[dict], *, customers: int, start_ns: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    events = []
    for i, row in enumerate(rows):
        events.append({"message_id": str(uuid.UUID(int=rng.getrandbits(128))),
                       "customer_id": f"c_{rng.randrange(customers):04d}", "channel": rng.choice(CHANNELS),
                       "lang": row["lang"], "text": row["text"], "ts_sent_ns": start_ns + i})
    return events


def main() -> None:
    from confluent_kafka import Producer

    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=10, help="messages per second (open loop)")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--lang-mix", default="es:0.5,en:0.5")
    parser.add_argument("--set", dest="dataset", choices=SETS, default="test")
    parser.add_argument("--customers", type=int, default=200)
    parser.add_argument("--broker", default="localhost:9092")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--surge", help="route:factor:at_s:duration_s, e.g. payments:5:120:180")
    args = parser.parse_args()

    mix = {k: float(v) for k, v in (part.split(":") for part in args.lang_mix.split(","))}
    total = int(args.rate * args.duration)
    rows = pick_rows(load_rows(args.dataset), total, mix, args.seed)
    schedule = [(i / args.rate, row) for i, row in enumerate(rows)]
    if args.surge:
        route, factor, at_s, duration_s = parse_surge(args.surge)
        schedule = sorted(schedule + surge_rows(load_rows(args.dataset), route, factor=factor,
                                                base_rate=args.rate, at_s=at_s, duration_s=duration_s,
                                                seed=args.seed), key=lambda x: x[0])
    events = build_events([row for _, row in schedule], customers=args.customers, start_ns=time.time_ns(),
                          seed=args.seed)
    producer = Producer({"bootstrap.servers": args.broker})
    began = time.perf_counter()
    for (offset, _), event in zip(schedule, events, strict=True):
        wait = began + offset - time.perf_counter()            # open loop: never waits for replies
        if wait > 0:
            time.sleep(wait)
        event["ts_sent_ns"] = time.time_ns()
        producer.produce("inbound-messages", key=event["customer_id"], value=json.dumps(event))
        producer.poll(0)
    producer.flush()
    print(f"sent {len(events)} messages in {time.perf_counter() - began:.1f}s")


if __name__ == "__main__":
    main()
