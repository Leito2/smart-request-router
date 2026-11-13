import os

from services.envfile import load_env


def test_loads_values_but_never_overrides_real_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nGATEWAY_KEY=gw_abc   # inline note\nLLM_MODEL=smart_router\n\nbroken line\n"
                   "LANGFUSE_PUBLIC_KEY=\nBROKER=from-file\n", encoding="utf-8")
    monkeypatch.delenv("GATEWAY_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.setenv("BROKER", "from-shell")

    load_env(env)

    assert os.environ["GATEWAY_KEY"] == "gw_abc"
    assert os.environ["LLM_MODEL"] == "smart_router"
    assert "LANGFUSE_PUBLIC_KEY" not in os.environ      # empty value = unset (keeps the no-op tracer)
    assert os.environ["BROKER"] == "from-shell"


def test_missing_file_is_fine(tmp_path):
    load_env(tmp_path / "nope.env")
