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


def test_general_training_examples_never_overlap_the_hard_cases_used_for_evaluation():
    from data_prep import _normalize
    from general_train import general_examples
    from hard_cases import hand_written

    training = {_normalize(e["text"]) for e in general_examples()}
    evaluation = {_normalize(c["text"]) for c in hand_written()}

    assert len(training) >= 100
    assert training.isdisjoint(evaluation)
    assert {e["lang"] for e in general_examples()} == {"en", "es"}
    assert all(e["route"] == "general" and e["urgency"] == 0 for e in general_examples())


def test_general_records_go_only_to_train_and_val_with_the_standard_labels():
    from data_prep import general_records

    examples = [{"text": f"hello number {i}", "lang": "en", "route": "general", "urgency": 0}
                for i in range(60)]

    records = general_records(examples, seed=3, val_fraction=0.2)

    assert {r["split"] for r in records} == {"train", "val"}
    assert len({r["id"] for r in records}) == 60
    assert all(r["route"] == "general" and r["intent"] is None and not r["needs_human"] for r in records)
