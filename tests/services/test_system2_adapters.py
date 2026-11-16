import httpx

from services.system2.adapters import GatewayLlm, LiveTools, extra_body_from_env


def test_gateway_llm_sends_alias_and_returns_text_and_counts_tokens():
    seen = {}

    def handler(request):
        import json
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}],
                                         "usage": {"total_tokens": 42}})
    llm = GatewayLlm("http://gw", api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert llm("system", "user") == "{}"
    assert seen["body"]["model"] == "smart_router" and seen["auth"] == "Bearer k"
    assert seen["body"]["messages"][1] == {"role": "user", "content": "user"} and llm.tokens == 42


def test_live_tools_delegate_to_ports():
    class History:
        def get(self, customer_id):
            return [customer_id]

    class Payments:
        def decision_for(self, text):
            return "ALLOW"

    class Kb:
        def answer(self, question):
            return "kb"

    tools = LiveTools(History(), Payments(), Kb())
    assert (tools.get_customer_history("c1"), tools.lookup_payment_decision("x"), tools.search_kb("y")) == (
        ["c1"], "ALLOW", "kb")


def test_extra_body_defaults_to_no_thinking_for_ollama_and_is_overridable_for_the_gateway(monkeypatch):
    sent = []

    def handler(request):
        import json
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    GatewayLlm("http://ollama", client=client)("s", "u")
    GatewayLlm("http://gw", client=client, extra_body={})("s", "u")
    assert sent[0]["reasoning_effort"] == "none" and "reasoning_effort" not in sent[1]
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"reasoning_effort": "low"}')
    assert extra_body_from_env({}) == {"reasoning_effort": "low"}
    monkeypatch.delenv("LLM_EXTRA_BODY")
    assert extra_body_from_env({"x": 1}) == {"x": 1}
