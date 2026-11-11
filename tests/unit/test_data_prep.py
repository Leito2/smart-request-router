from data_prep import build_dataset

from routercore.domain.routes import RouteMap

ROUTE_MAP = RouteMap(
    routes={"security": "fraud", "knowledge": "how-to", "general": "other"},
    intents={"lost_or_stolen_card": ("security", 3), "country_support": ("knowledge", 0)},
)


def row(origin_id, text, intent):
    return {"origin_id": origin_id, "text": text, "intent": intent}


def test_a_translation_lands_in_the_same_split_as_its_source_and_rows_carry_labels():
    train = [row(f"tr-{i}", f"my card was stolen {i}", "lost_or_stolen_card") for i in range(40)]
    test = [row("te-0", "which countries are supported", "country_support")]
    translations = {r["origin_id"]: f"me robaron la tarjeta {r['text'][-2:]}" for r in train}
    translations["te-0"] = "qué países soportan"

    records = build_dataset(train, test, translations, ROUTE_MAP, seed=7)

    split_of = {}
    for rec in records:
        split_of.setdefault(rec["origin_id"], set()).add(rec["split"])
    assert all(len(splits) == 1 for splits in split_of.values())
    assert split_of["te-0"] == {"test"}
    assert {rec["split"] for rec in records} == {"train", "val", "test"}
    stolen = next(r for r in records if r["origin_id"] == "tr-0" and r["lang"] == "en")
    labels = (stolen["route"], stolen["urgency"], stolen["needs_human"], stolen["needs_kb"])
    assert labels == ("security", 3, True, False)
    assert {r["lang"] for r in records if r["origin_id"] == "te-0"} == {"en", "es"}


def test_a_train_text_that_also_appears_in_test_is_dropped_so_nothing_leaks():
    train = [row("tr-0", "Which countries are supported?", "country_support"),
             row("tr-1", "my card was stolen", "lost_or_stolen_card")]
    test = [row("te-0", "which countries are supported", "country_support")]

    records = build_dataset(train, test, {}, ROUTE_MAP, seed=0)

    texts_outside_test = {r["text"].casefold().strip(" ?.!") for r in records if r["split"] != "test"}
    texts_in_test = {r["text"].casefold().strip(" ?.!") for r in records if r["split"] == "test"}
    assert texts_in_test == {"which countries are supported"}
    assert texts_in_test.isdisjoint(texts_outside_test)
    assert "my card was stolen" in texts_outside_test


def test_golden_set_is_balanced_across_routes_and_languages_and_includes_hard_cases():
    from collections import Counter

    from data_prep import select_golden

    regular = [{"id": f"{r}-{lang}-{i}", "route": r, "lang": lang, "split": "test"}
               for r in ("security", "cards") for lang in ("en", "es") for i in range(20)]
    hard = [{"id": f"hc-{c}-{i}", "route": "general", "lang": "en", "category": c, "split": "hard"}
            for c in ("ood", "short") for i in range(10)]

    golden = select_golden(regular, hard, size=20, hard_share=0.25, seed=1)

    assert len(golden) == 20 and len({g["id"] for g in golden}) == 20
    strata = Counter((g["route"], g["lang"]) for g in golden if g["split"] == "test")
    assert set(strata) == {("security", "en"), ("security", "es"), ("cards", "en"), ("cards", "es")}
    assert max(strata.values()) - min(strata.values()) <= 1
    hard_picked = Counter(g["category"] for g in golden if g["split"] == "hard")
    assert sum(hard_picked.values()) == 5 and set(hard_picked) == {"ood", "short"}
