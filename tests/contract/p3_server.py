"""The real P3 `rag_api.app.create_app` with a scripted answer generator. Run with P3's own interpreter:
`<P3>/.venv/.../python -m uvicorn --app-dir tests/contract --factory p3_server:build`."""
from rag_api.app import create_app
from ragcore.domain.contracts import SSEEvent, StreamEvent


async def answer(request):
    question = request.question
    yield StreamEvent(SSEEvent.STATUS, {"phase": "retrieving"})
    if question == "boom":
        raise RuntimeError("model down")
    yield StreamEvent(SSEEvent.TOKEN, {"text": "Draft "})
    if question == "retract":
        yield StreamEvent(SSEEvent.RETRACT, {"text": "Cannot answer.", "reason": "unsupported"})
        yield StreamEvent(SSEEvent.TOKEN, {"text": "Cannot answer."})
    else:
        yield StreamEvent(SSEEvent.TOKEN, {"text": "answer [1]"})
    yield StreamEvent(SSEEvent.CITATIONS, {"citations": []})
    yield StreamEvent(SSEEvent.DONE, {"abstained": question == "retract"})


def build():
    return create_app(answer)
