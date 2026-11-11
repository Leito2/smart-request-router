import asyncio
import math

from judgekit.drift import drift_report, psi
from judgekit.judges import ClaimsJudge, PairwiseJudge, Panel, RubricJudge


def scripted(*answers):
    queue = list(answers)

    async def complete(system: str, user: str) -> str:
        return queue.pop(0)
    return complete


def test_psi_is_zero_for_identical_and_grows_with_shift():
    same = {"a": 50, "b": 50}
    assert psi(same, same) == 0
    assert psi(same, {"a": 90, "b": 10}) > 0.25
    assert math.isclose(psi({"a": 1, "b": 1}, {"a": 100, "b": 100}), 0.0, abs_tol=1e-9)


def test_psi_handles_categories_missing_on_one_side():
    assert psi({"a": 50, "b": 50}, {"a": 50, "b": 40, "c": 10}) > 0


def test_drift_report_flags_features_above_threshold():
    report = drift_report({"route": {"a": 50, "b": 50}, "lang": {"en": 50, "es": 50}},
                          {"route": {"a": 95, "b": 5}, "lang": {"en": 52, "es": 48}}, threshold=0.2)
    assert report["route"]["drifted"] is True and report["lang"]["drifted"] is False


def test_rubric_judge_parses_score_and_reason():
    judge = RubricJudge(scripted('{"score": 4, "reason": "uses the evidence"}'), rubric="Grounded")
    result = asyncio.run(judge.judge({"answer": "x"}))
    assert (result.score, result.reason) == (4, "uses the evidence")


def test_rubric_judge_clamps_and_survives_garbage():
    assert asyncio.run(RubricJudge(scripted('{"score": 9}'), rubric="r").judge({})).score == 5
    assert asyncio.run(RubricJudge(scripted("no idea"), rubric="r").judge({})).score is None


def test_pairwise_judge_swaps_positions_and_ties_on_inconsistency():
    consistent = PairwiseJudge(scripted('{"winner": "A"}', '{"winner": "B"}'))
    assert asyncio.run(consistent.judge("q", "first", "second")) == "first"
    biased = PairwiseJudge(scripted('{"winner": "A"}', '{"winner": "A"}'))
    assert asyncio.run(biased.judge("q", "first", "second")) == "tie"


def test_claims_judge_scores_share_of_supported_claims():
    judge = ClaimsJudge(scripted('{"claims": [{"claim": "c1", "supported": true}, '
                                 '{"claim": "c2", "supported": false}]}'))
    result = asyncio.run(judge.judge("answer", "evidence"))
    assert result.score == 0.5 and result.unsupported == ["c2"]


def test_panel_averages_and_reports_disagreement():
    a = RubricJudge(scripted('{"score": 5, "reason": ""}'), rubric="r")
    b = RubricJudge(scripted('{"score": 2, "reason": ""}'), rubric="r")
    panel = asyncio.run(Panel([a, b], max_spread=1).judge({}))
    assert panel.score == 3.5 and panel.disagreement is True
