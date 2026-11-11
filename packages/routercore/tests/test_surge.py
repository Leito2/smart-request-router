from routercore.domain.surge import SurgeConfig, SurgeDetector, SurgeMonitor

CFG = SurgeConfig(window_s=60, ewma_alpha=0.2, z_threshold=4, consecutive_windows=2, warmup_windows=5)


def feed(detector, counts, intent="card_declined"):
    alerts = []
    for window, count in enumerate(counts):
        alert = detector.observe(intent, window, count)
        if alert:
            alerts.append((window, alert))
    return alerts


def test_steady_traffic_never_alerts():
    assert feed(SurgeDetector(CFG), [20, 22, 19, 21, 20, 23, 18, 21, 20, 22, 19, 20]) == []


def test_sustained_spike_alerts_on_the_second_window():
    alerts = feed(SurgeDetector(CFG), [20, 22, 19, 21, 20, 22, 20, 100, 110, 120])
    assert [w for w, _ in alerts] == [8]
    alert = alerts[0][1]
    assert alert.intent == "card_declined" and alert.volume == 110 and alert.zscore > 4


def test_single_window_blip_is_ignored():
    assert feed(SurgeDetector(CFG), [20, 22, 19, 21, 20, 22, 20, 100, 21, 20]) == []


def test_no_alerts_during_warmup():
    assert feed(SurgeDetector(CFG), [5, 80, 90, 100, 110]) == []


def test_baseline_is_not_learned_from_the_surge():
    detector = SurgeDetector(CFG)
    feed(detector, [20, 22, 19, 21, 20, 22, 20, 100, 110, 120, 130])
    assert detector._series["card_declined"].mean < 30             # the surge did not move the baseline
    assert feed(detector, [20, 21], intent="other") == []          # other intents have their own state


def test_a_second_surge_alerts_again_after_traffic_normalizes():
    alerts = feed(SurgeDetector(CFG), [20, 22, 19, 21, 20, 22, 20, 100, 110, 21, 20, 100, 110])
    assert [w for w, _ in alerts] == [8, 12]


def test_monitor_aggregates_events_into_windows_and_fills_empty_ones():
    monitor = SurgeMonitor(CFG)
    ns = 1_000_000_000
    emitted = []
    for window in range(8):
        for _ in range(20):
            emitted += monitor.add("fees", window * 60 * ns + 1)
    for _ in range(100):
        emitted += monitor.add("fees", 8 * 60 * ns + 1)
    for _ in range(110):
        emitted += monitor.add("fees", 9 * 60 * ns + 1)
    emitted += monitor.add("fees", 10 * 60 * ns + 1)        # closes window 9
    assert len(emitted) == 1 and emitted[0].volume == 110


def test_surge_yaml_matches_the_domain_defaults():
    from pathlib import Path

    from routercore.adapters.surge_config import load_surge_config

    root = Path(__file__).resolve().parents[3]
    assert load_surge_config(root / "config" / "surge.yaml") == SurgeConfig()
