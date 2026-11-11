import httpx

from services.system2.adapters import GatewayLlm, LiveTools


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
