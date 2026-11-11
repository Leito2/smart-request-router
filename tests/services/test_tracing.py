import pytest

from services.system2.tracing import NoopTracer, trace_run


class RecordingTracer:
    def __init__(self):
        self.events = []

    def span(self, name, input):
        tracer = self

        class Span:
            def __enter__(self_inner):
                tracer.events.append(("start", name, input))
                return self_inner

            def update(self_inner, output=None, metadata=None):
                tracer.events.append(("update", output, metadata))

            def __exit__(self_inner, *exc):
                tracer.events.append(("end", exc[0] is not None))
        return Span()


def test_trace_run_records_input_output_and_metadata():
    tracer = RecordingTracer()
    result = trace_run(tracer, "system2", {"id": "m1"}, lambda: {"routed": {"route": "cards"}},
                       metadata=lambda out: {"route": out["routed"]["route"]})
    assert result["routed"]["route"] == "cards"
    assert tracer.events[0] == ("start", "system2", {"id": "m1"})
    assert tracer.events[1] == ("update", {"routed": {"route": "cards"}}, {"route": "cards"})
    assert tracer.events[-1] == ("end", False)


def test_trace_run_marks_failures_and_reraises():
    tracer = RecordingTracer()

    def boom():
        raise RuntimeError("llm down")
    with pytest.raises(RuntimeError):
        trace_run(tracer, "system2", {}, boom)
    assert tracer.events[-1] == ("end", True)


def test_noop_tracer_is_transparent():
    assert trace_run(NoopTracer(), "x", {}, lambda: 41 + 1) == 42
