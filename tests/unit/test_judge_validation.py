from judge_validation import SYSTEM, agreement, build_cases, parse_verdict, seeded, system_prompt


def test_parse_verdict_accepts_json_or_bare_digit_and_rejects_garbage():
    assert parse_verdict('{"acceptable": 1}') == 1 and parse_verdict("0") == 0
    assert parse_verdict('Sure! {"acceptable": 0, "reason": "x"}') == 0
    assert parse_verdict("maybe") is None


def test_agreement_uses_kappa_and_flags_unusable_judges():
    reference = {f"c{i}": i % 2 for i in range(10)}
    good = dict(reference)
    bad = {k: 1 - v for k, v in reference.items()}
    assert agreement(good, reference)["usable"] is True
    report = agreement(bad, reference)
    assert report["usable"] is False and report["kappa"] < 0


def test_unparsed_judge_answers_count_against_the_judge():
    reference = {"a": 1, "b": 0, "c": 1, "d": 0}
    judge = {"a": 1, "b": 0, "c": None, "d": None}
    report = agreement(judge, reference)
    assert report["n"] == 4 and report["unparsed"] == 2


def test_build_cases_mixes_correct_and_wrong_proposals_deterministically():
    rows = [{"id": f"r{i}", "text": f"t{i}", "route": "cards"} for i in range(20)]
    predictions = ["cards" if i % 2 else "fees" for i in range(20)]
    cases = build_cases(rows, predictions, size=10, seed=0)
    assert len(cases) == 10 and cases == build_cases(rows, predictions, size=10, seed=0)
    assert {c["proposed_route"] for c in cases} == {"cards", "fees"}


def test_rubrics_add_route_descriptions_then_strictness():
    routes = {"cards": "lost or blocked cards", "fees": "charges and fees"}
    assert system_prompt(1, routes) == SYSTEM
    covered = system_prompt(2, routes)
    assert "- cards: lost or blocked cards" in covered and "Be strict" not in covered
    strict = system_prompt(3, routes)
    assert "- fees: charges and fees" in strict and "When in doubt, answer 0" in strict
    assert strict.endswith(SYSTEM.split("Reply JSON only")[1])      # the JSON answer format survives


def test_seeded_keeps_seed_zero_names_and_suffixes_held_out_sets():
    assert seeded("cases.jsonl", 0) == "cases.jsonl"
    assert seeded("judge_kappa.json", 3) == "judge_kappa_seed3.json"
