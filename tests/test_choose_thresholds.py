from choose_thresholds import choose, evaluate, pick_tau_urgent

from routercore.domain.policy import Policy


def record(route_p, truth, urgent_mass=0.0, urgency="0"):
    return {"lang": "en", "route_p": route_p, "route": truth, "urgency": urgency,
            "urgency_p": {"0": 1 - urgent_mass, "1": 0.0, "2": urgent_mass, "3": 0.0}}


def test_tau_urgent_catches_the_required_share_of_urgent_rows():
    masses = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05)
    rows = [record({"a": 1.0, "b": 0.0}, "a", mass, "2") for mass in masses]
    tau = pick_tau_urgent(rows, min_recall=0.9)
    flagged = sum(r["urgency_p"]["2"] > tau for r in rows)
    assert flagged >= 9


def test_evaluate_reports_coverage_and_accuracy_of_accepted_messages():
    rows = [record({"a": 0.95, "b": 0.05}, "a"), record({"a": 0.95, "b": 0.05}, "b"),
            record({"a": 0.55, "b": 0.45}, "a")]
    coverage, accuracy = evaluate(rows, Policy(tau=0.8, margin=0.1, tau_urgent=0.5, max_entropy=1.0))
    assert coverage == 2 / 3 and accuracy == 0.5


def test_choose_keeps_confident_correct_rows_and_escalates_the_unsure_wrong_ones():
    confident = [record({"a": 0.97, "b": 0.03}, "a") for _ in range(30)]
    unsure_wrong = [record({"a": 0.60, "b": 0.40}, "b") for _ in range(30)]
    result = choose(confident + unsure_wrong, target_accuracy=0.95)
    assert result["target_met"] and result["accuracy"] >= 0.95
    assert 0.45 <= result["coverage"] <= 0.5


def test_choose_flags_when_the_target_is_unreachable():
    rows = [record({"a": 0.9, "b": 0.1}, "b") for _ in range(10)]
    assert choose(rows, target_accuracy=0.9)["target_met"] is False


def test_tau_urgent_pools_extra_out_of_fold_masses_into_the_tail():
    few = [record({"a": 1.0, "b": 0.0}, "a", mass, "2") for mass in (0.9, 0.8, 0.7)]    # too few for a tail
    alone = pick_tau_urgent(few, min_recall=0.98)
    pooled = pick_tau_urgent(few, min_recall=0.98, extra_urgent_masses=[0.05] * 3 + [0.9] * 97)
    assert alone > 0.69                      # no allowed misses: below the lowest urgent mass of three rows
    assert pooled < 0.06                     # 103 rows allow 2 misses: the tail now sits at 0.05
