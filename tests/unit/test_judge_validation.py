from judge_validation import agreement, build_cases, parse_verdict


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
