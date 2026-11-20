from scripts.langfuse_annotation import pick_cases, review_seconds, summarize


def test_pick_cases_is_seeded_and_sorted():
    cases = [{"id": f"c{i}"} for i in range(20)]
    first = pick_cases(cases, 5, seed=0)
    assert first == pick_cases(cases, 5, seed=0)
    assert [c["id"] for c in first] == sorted(c["id"] for c in first)
    assert len(pick_cases(cases, 50, seed=0)) == 20


def test_review_seconds_drops_breaks_instead_of_clipping_them():
    assert review_seconds([100.0, 110.0, 130.0, 1000.0, 1008.0]) == [10.0, 20.0, 8.0]
    assert review_seconds([5.0]) == []


def test_summarize_agreement_time_and_triage():
    human = {"a": 1, "b": 0, "c": 0, "d": 1}
    reference = {"a": 1, "b": 0, "c": 0, "d": 1, "e": 1}
    judge = {"a": 1, "b": 0, "c": 1, "d": 1, "e": 0}              # misses the error in c
    report = summarize(human, reference, judge, gaps=[20.0, 40.0, 30.0], judge_seconds=3.0)
    assert report["n"] == 4 and report["human_not_acceptable"] == 2
    assert report["kappa_human_vs_opus"] == 1.0
    assert report["human_seconds_per_case_median"] == 30.0
    assert report["qa_time_saved_pct"] == 90.0
    assert report["review_only_judge_rejected"] == {
        "share_of_cases_reviewed": 0.25, "real_errors_caught": "1/2", "time_saved_pct": 75.0}


def test_summarize_counts_a_judge_without_an_answer_as_not_flagging():
    report = summarize({"a": 0}, {"a": 0}, {"a": None}, gaps=[], judge_seconds=2.0)
    assert report["review_only_judge_rejected"]["real_errors_caught"] == "0/1"
    assert report["qa_time_saved_pct"] is None
