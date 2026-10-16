# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import sys
from types import SimpleNamespace

import httpx
import pytest

import llamafactory.api.guard as guard_module
from llamafactory.api import app as app_module
from llamafactory.api.protocol import (
    ChatCompletionMessage,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseUsage,
)


GUARD_ENVS = [
    "API_GUARD",
    "API_GUARD_RATE_LIMIT",
    "API_GUARD_RATE_LIMIT_WINDOW",
    "API_GUARD_WHITELIST",
    "API_GUARD_BLACKLIST",
    "API_GUARD_TRUSTED_PROXIES",
    "API_GUARD_SCAN_BODY",
    "API_GUARD_REDIS_URL",
]
ATTACK = "<script>alert(1)</script>"
CHAT_BODY = {"model": "m", "messages": [{"role": "user", "content": "Hi"}]}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for name in GUARD_ENVS:
        monkeypatch.delenv(name, raising=False)


def _make_app(monkeypatch: pytest.MonkeyPatch, **env: str):
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    async def fake_completion(request, chat_model):
        return ChatCompletionResponse(
            id="chatcmpl-test",
            model="m",
            choices=[
                ChatCompletionResponseChoice(
                    index=0, message=ChatCompletionMessage(role="assistant", content="ok"), finish_reason="stop"
                )
            ],
            usage=ChatCompletionResponseUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr(app_module, "create_chat_completion_response", fake_completion)
    chat_model = SimpleNamespace(engine=SimpleNamespace(name="hf", can_generate=True))
    return app_module.create_app(chat_model)


def _request(app, method: str, url: str, ip: str = "203.0.113.10", **kwargs) -> httpx.Response:
    async def run():
        transport = httpx.ASGITransport(app=app, client=(ip, 12345))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(run())


def _has_guard(app) -> bool:
    return any(middleware.cls.__name__ == "PureASGISecurityMiddleware" for middleware in app.user_middleware)


def test_disabled_adds_nothing_and_never_imports(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delitem(sys.modules, "guard", raising=False)
    app = _make_app(monkeypatch)
    assert not _has_guard(app)
    assert [m.cls.__name__ for m in app.user_middleware] == ["CORSMiddleware"]
    assert "guard" not in sys.modules
    assert _request(app, "GET", "/v1/models").status_code == 200


def test_enabled_without_package_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(guard_module, "is_fastapi_guard_available", lambda: False)
    with pytest.raises(RuntimeError, match="fastapi-guard"):
        _make_app(monkeypatch, API_GUARD="1")


def test_cors_is_outside_the_guard(monkeypatch: pytest.MonkeyPatch):
    app = _make_app(monkeypatch, API_GUARD="1")
    assert [m.cls.__name__ for m in app.user_middleware] == ["CORSMiddleware", "PureASGISecurityMiddleware"]


def test_rate_limit_burst_returns_429_with_cors(monkeypatch: pytest.MonkeyPatch):
    app = _make_app(monkeypatch, API_GUARD="1", API_GUARD_RATE_LIMIT="3", API_GUARD_RATE_LIMIT_WINDOW="60")
    headers = {"Origin": "https://example.com"}
    codes = [_request(app, "GET", "/v1/models", ip="198.51.100.1", headers=headers).status_code for _ in range(3)]
    assert codes == [200, 200, 200]
    blocked = _request(app, "GET", "/v1/models", ip="198.51.100.1", headers=headers)
    assert blocked.status_code == 429
    assert blocked.headers["access-control-allow-origin"] == "*"


def test_whitelist_blocks_other_clients(monkeypatch: pytest.MonkeyPatch):
    app = _make_app(monkeypatch, API_GUARD="1", API_GUARD_WHITELIST="198.51.100.7")
    assert _request(app, "GET", "/v1/models", ip="198.51.100.7").status_code == 200
    assert _request(app, "GET", "/v1/models", ip="203.0.113.99").status_code == 403


def test_spoofed_forwarded_for_is_ignored(monkeypatch: pytest.MonkeyPatch):
    app = _make_app(monkeypatch, API_GUARD="1", API_GUARD_WHITELIST="198.51.100.7")
    spoofed = {"X-Forwarded-For": "198.51.100.7"}
    assert _request(app, "GET", "/v1/models", ip="203.0.113.50", headers=spoofed).status_code == 403


def test_forwarded_for_is_honored_from_trusted_proxy(monkeypatch: pytest.MonkeyPatch):
    app = _make_app(monkeypatch, API_GUARD="1", API_GUARD_WHITELIST="198.51.100.7")
    forwarded = {"X-Forwarded-For": "198.51.100.7"}
    assert _request(app, "GET", "/v1/models", ip="127.0.0.1", headers=forwarded).status_code == 200


def test_attack_in_query_is_blocked(monkeypatch: pytest.MonkeyPatch):
    app = _make_app(monkeypatch, API_GUARD="1")
    assert _request(app, "GET", "/v1/models", ip="198.51.100.20", params={"q": ATTACK}).status_code == 400


def test_body_scan_is_off_by_default(monkeypatch: pytest.MonkeyPatch):
    body = {**CHAT_BODY, "messages": [{"role": "user", "content": f"explain {ATTACK}"}]}
    app = _make_app(monkeypatch, API_GUARD="1")
    assert _request(app, "POST", "/v1/chat/completions", ip="198.51.100.30", json=body).status_code == 200


def test_body_scan_can_be_enabled(monkeypatch: pytest.MonkeyPatch):
    body = {**CHAT_BODY, "messages": [{"role": "user", "content": f"explain {ATTACK}"}]}
    app = _make_app(monkeypatch, API_GUARD="1", API_GUARD_SCAN_BODY="1")
    assert _request(app, "POST", "/v1/chat/completions", ip="198.51.100.31", json=body).status_code == 400


def test_streaming_chunks_are_delivered_incrementally(monkeypatch: pytest.MonkeyPatch):
    app = _make_app(monkeypatch, API_GUARD="1")
    state: dict = {}

    async def fake_stream(request, chat_model):
        yield json.dumps({"n": 1})
        # Only continues once the client side saw chunk 1, so a buffering middleware would deadlock here.
        await asyncio.wait_for(state["event"].wait(), timeout=5)
        yield json.dumps({"n": 2})

    monkeypatch.setattr(app_module, "create_stream_chat_completion_response", fake_stream)

    async def run():
        state["event"] = asyncio.Event()
        body = json.dumps({**CHAT_BODY, "stream": True}).encode()
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"test"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            "client": ("198.51.100.40", 1234),
            "server": ("test", 80),
        }
        sent_body = False

        async def receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.sleep(3600)

        seen: list[bytes] = []

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                seen.append(message["body"])
                if b'"n": 1' in message["body"]:
                    state["event"].set()

        await asyncio.wait_for(app(scope, receive, send), timeout=10)
        return seen

    seen = asyncio.run(run())
    assert any(b'"n": 1' in chunk for chunk in seen)
    assert any(b'"n": 2' in chunk for chunk in seen)
    assert len(seen) >= 2
