from judgekit.stats import paired_bootstrap


def test_clear_improvement_has_positive_interval():
    a = [1] * 90 + [0] * 10          # system A right 90%
    b = [1] * 60 + [0] * 40          # system B right 60%
    result = paired_bootstrap(a, b, samples=2000, seed=0)
    assert result.mean_diff == 0.3 and result.low > 0 and result.significant


def test_no_difference_interval_contains_zero():
    a = [1, 0] * 50
    result = paired_bootstrap(a, list(a), samples=1000, seed=0)
    assert result.mean_diff == 0 and result.low <= 0 <= result.high and not result.significant


def test_noisy_small_sample_is_not_significant():
    a, b = [1, 1, 0, 1, 0], [1, 0, 0, 1, 1]
    assert not paired_bootstrap(a, b, samples=2000, seed=1).significant


def test_is_deterministic_for_a_seed():
    a, b = [1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 0, 1]
    assert paired_bootstrap(a, b, seed=3) == paired_bootstrap(a, b, seed=3)
