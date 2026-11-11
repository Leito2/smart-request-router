import asyncio

from judgekit.runner import run


def test_concurrency_never_exceeds_the_limit_and_all_items_finish():
    in_flight, peak = 0, 0

    async def work(item):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return item * 2

    result = asyncio.run(run([(str(i), i) for i in range(20)], work, concurrency=4))

    assert peak == 4
    assert result.outputs == {str(i): i * 2 for i in range(20)}
    assert result.failures == {}


def test_a_transient_failure_is_retried_and_a_permanent_one_is_recorded_without_stopping_the_rest():
    calls = {"flaky": 0, "broken": 0}

    async def work(item):
        if item in calls:
            calls[item] += 1
        if item == "flaky" and calls["flaky"] < 3:
            raise ConnectionError("timeout")
        if item == "broken":
            raise ValueError("bad input")
        return item.upper()

    items = [("a", "flaky"), ("b", "broken"), ("c", "fine")]
    result = asyncio.run(run(items, work, retries=2, backoff_s=0))

    assert result.outputs == {"a": "FLAKY", "c": "FINE"}
    assert calls == {"flaky": 3, "broken": 3}            # 1 try + 2 retries each
    assert list(result.failures) == ["b"] and "bad input" in result.failures["b"]


def test_a_rerun_resumes_from_the_checkpoint_and_only_calls_the_pending_items(tmp_path):
    checkpoint = tmp_path / "run.jsonl"
    seen = []

    async def work(item):
        seen.append(item)
        return {"label": item}

    asyncio.run(run([("a", "x"), ("b", "y")], work, checkpoint=checkpoint))
    seen.clear()
    result = asyncio.run(run([("a", "x"), ("b", "y"), ("c", "z")], work, checkpoint=checkpoint))

    assert seen == ["z"]
    assert result.outputs == {"a": {"label": "x"}, "b": {"label": "y"}, "c": {"label": "z"}}


def test_cost_is_estimated_before_running_from_token_counts_and_a_price():
    from judgekit.runner import estimate_cost

    # 4 items x 500 tokens = 2,000 tokens; at $2 per million tokens that is $0.004.
    assert estimate_cost(4, tokens_per_item=500, usd_per_1m_tokens=2.0) == 0.004
    assert estimate_cost(10, tokens_per_item=1000, usd_per_1m_tokens=0.0) == 0.0
