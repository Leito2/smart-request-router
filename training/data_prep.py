"""Builds the router dataset from Banking77 rows (PLAN §4.2).

Pure function over rows already in memory: downloading and translating live in other scripts, so the
split and labelling rules can be tested without network or models.
"""
import hashlib
import random

from routercore.domain.routes import RouteMap

HUMAN_ROUTES = {"security", "disputes", "kyc"}
KB_INTENTS = {"exchange_rate", "top_up_limits", "disposable_card_limits", "card_delivery_estimate",
              "transfer_timing"}


def _normalize(text: str) -> str:
    return " ".join(text.casefold().strip(" ?.!").split())


def _in_validation(origin_id: str, seed: int, val_fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{origin_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32 < val_fraction


def _record(origin_id: str, lang: str, text: str, intent: str, split: str, route_map: RouteMap) -> dict:
    route, urgency = route_map.route_of(intent), route_map.urgency_of(intent)
    return {
        "id": f"{origin_id}:{lang}", "origin_id": origin_id, "lang": lang, "text": text, "intent": intent,
        "route": route, "urgency": urgency,
        "needs_human": urgency >= 1 or route in HUMAN_ROUTES,
        "needs_kb": route == "knowledge" or intent in KB_INTENTS,
        "split": split,
    }


def build_dataset(en_train: list[dict], en_test: list[dict], translations: dict[str, str],
                  route_map: RouteMap, *, seed: int = 0, val_fraction: float = 0.1) -> list[dict]:
    """Return labelled records. A translation inherits its source's split, so paraphrases never straddle."""
    records = []
    test_texts = {_normalize(r["text"]) for r in en_test}
    for rows, fixed_split in ((en_train, None), (en_test, "test")):
        for r in rows:
            if fixed_split is None and _normalize(r["text"]) in test_texts:
                continue                                  # same text in test: drop it from train/val
            split = fixed_split or ("val" if _in_validation(r["origin_id"], seed, val_fraction) else "train")
            records.append(_record(r["origin_id"], "en", r["text"], r["intent"], split, route_map))
            if r["origin_id"] in translations:
                records.append(_record(r["origin_id"], "es", translations[r["origin_id"]], r["intent"], split,
                                       route_map))
    return records


def _round_robin(rows: list[dict], key, n: int, rng: random.Random) -> list[dict]:
    """Pick n rows taking turns across the key's groups, so every group is represented before any repeats."""
    groups: dict = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    queues = [rng.sample(group, len(group)) for _, group in sorted(groups.items())]
    picked = []
    while len(picked) < n and any(queues):
        for queue in queues:
            if queue and len(picked) < n:
                picked.append(queue.pop())
    return picked


def select_golden(regular: list[dict], hard: list[dict], *, size: int, hard_share: float,
                  seed: int) -> list[dict]:
    """The golden set: test rows balanced by (route, lang) plus hard cases balanced by category."""
    rng = random.Random(seed)
    hard_n = round(size * hard_share)
    picked_hard = _round_robin(hard, lambda r: r["category"], hard_n, rng)
    picked_regular = _round_robin(regular, lambda r: (r["route"], r["lang"]), size - len(picked_hard), rng)
    return picked_regular + picked_hard
