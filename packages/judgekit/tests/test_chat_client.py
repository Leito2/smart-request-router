import asyncio

from aiohttp import web
from aiohttp.test_utils import TestServer

from judgekit.clients import ChatClient


def test_chat_client_returns_text_and_token_usage_from_an_openai_compatible_endpoint():
    seen = {}

    async def completions(request):
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = await request.json()
        return web.json_response({
            "choices": [{"message": {"content": '{"route": "cards"}'}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 9},
        })

    async def scenario():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", completions)
        async with TestServer(app) as server:
            async with ChatClient(str(server.make_url("")), api_key="k", model="smart_router") as client:
                return await client.complete([{"role": "user", "content": "hi"}], max_tokens=50)

    completion = asyncio.run(scenario())

    assert completion.text == '{"route": "cards"}'
    assert (completion.prompt_tokens, completion.completion_tokens) == (120, 9)
    assert completion.latency_ms >= 0
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "smart_router" and seen["body"]["max_tokens"] == 50
