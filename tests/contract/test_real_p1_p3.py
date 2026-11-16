"""Our P1/P3 HTTP clients against the *real* P1 and P3 FastAPI apps (not mocks), each served by its own
repository's interpreter. Skipped when the sibling repositories are not next to this one (CI).

Override the locations with P1_DIR / P3_DIR."""
import os
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from routercore.adapters.http_clients import HttpKnowledgeBase, HttpPaymentDecisions

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
P1 = Path(os.environ.get("P1_DIR", ROOT.parent / "realtime-fraud-detection"))
P3 = Path(os.environ.get("P3_DIR", ROOT.parent / "live-rag-platform"))


def _python(repo: Path) -> Path | None:
    for candidate in (repo / ".venv" / "Scripts" / "python.exe", repo / ".venv" / "bin" / "python"):
        if candidate.exists():
            return candidate
    return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(repo: Path, module: str, health: str):
    python = _python(repo)
    if python is None:
        pytest.skip(f"{repo} has no .venv")
    port = _free_port()
    process = subprocess.Popen(
        [str(python), "-m", "uvicorn", "--app-dir", str(HERE), "--factory", f"{module}:build",
         "--port", str(port), "--log-level", "warning"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"{module} exited: {process.stderr.read().decode()[-800:]}")
        try:
            if httpx.get(url + health, timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        process.kill()
        pytest.fail(f"{module} did not start")
    return process, url


@pytest.fixture(scope="module")
def p1_url():
    process, url = _serve(P1, "p1_server", "/health")
    yield url
    process.terminate()
    process.wait(timeout=10)


@pytest.fixture(scope="module")
def p3_url():
    process, url = _serve(P3, "p3_server", "/healthz")
    yield url
    process.terminate()
    process.wait(timeout=10)


def test_real_p1_returns_decision_with_explanation(p1_url):
    answer = HttpPaymentDecisions(p1_url).decision_for("my card payment pay_explained was blocked")
    assert answer == "block: velocity spike on a new device"


def test_real_p1_null_explanation_does_not_leak_the_word_none(p1_url):
    assert HttpPaymentDecisions(p1_url).decision_for("what happened to pay_pending?") == "review"


def test_real_p1_unknown_payment_is_a_404_the_client_understands(p1_url):
    unknown = HttpPaymentDecisions(p1_url).decision_for("pay_missing please")
    assert unknown == "no decision found for that payment"


def test_real_p1_message_without_payment_id_never_calls_the_service():
    unreachable = HttpPaymentDecisions("http://127.0.0.1:9")
    assert unreachable.decision_for("my card was blocked") == "needs payment id"


def test_real_p3_streams_tokens_into_one_answer(p3_url):
    assert HttpKnowledgeBase(p3_url).answer("how do I reset my pin") == "Draft answer [1]"


def test_real_p3_retract_discards_the_streamed_draft(p3_url):
    assert HttpKnowledgeBase(p3_url).answer("retract") == "Cannot answer."


def test_real_p3_error_event_becomes_unavailable_marker(p3_url):
    assert HttpKnowledgeBase(p3_url).answer("boom") == "unavailable (internal_error)"


def test_real_p3_validation_error_degrades_instead_of_raising(p3_url):
    assert HttpKnowledgeBase(p3_url).answer("x").startswith("unavailable (")   # AskRequest needs >= 2 chars


def test_dead_services_degrade():
    assert HttpKnowledgeBase("http://127.0.0.1:9", timeout_s=1).answer("hello").startswith("unavailable (")
    dead = HttpPaymentDecisions("http://127.0.0.1:9", timeout_s=1)
    assert dead.decision_for("pay_x").startswith("unavailable (")
