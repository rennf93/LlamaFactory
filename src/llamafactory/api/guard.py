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

import os
from typing import TYPE_CHECKING

from ..extras.misc import is_env_enabled
from ..extras.packages import is_fastapi_guard_available


if TYPE_CHECKING:
    from fastapi import FastAPI


def _env_list(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, default).split(",") if item.strip())


def add_guard_middleware(app: "FastAPI") -> None:
    r"""Optionally add the fastapi-guard security middleware (rate limit, IP policy, request screening).

    Disabled unless ``API_GUARD=1``. Call it before adding ``CORSMiddleware`` so that CORS stays the outermost layer
    and the guard's 403/429 responses still carry CORS headers.

    Environment variables:
        API_GUARD: enable the middleware.
        API_GUARD_RATE_LIMIT: requests allowed per window and per client IP (default 60).
        API_GUARD_RATE_LIMIT_WINDOW: window in seconds (default 60).
        API_GUARD_WHITELIST: comma separated IPs or CIDRs; when set, only these clients are served.
        API_GUARD_BLACKLIST: comma separated IPs or CIDRs to block.
        API_GUARD_TRUSTED_PROXIES: comma separated proxy IPs or CIDRs whose X-Forwarded-For is honored
            (default loopback only; the header is ignored when the peer is not listed).
        API_GUARD_SCAN_BODY: also scan request bodies for attack patterns (default off, prompts are free text).
        API_GUARD_REDIS_URL: share rate-limit and ban state across workers (default in-memory).
    """
    if not is_env_enabled("API_GUARD"):
        return

    if not is_fastapi_guard_available():
        raise RuntimeError("API_GUARD is enabled but fastapi-guard is not installed: `pip install fastapi-guard`.")

    from guard import PureASGISecurityMiddleware, SecurityConfig  # pure ASGI: SSE streams are not buffered
    from guard_core.handlers.suspatterns_handler import ALL_DETECTION_CATEGORIES

    redis_url = os.getenv("API_GUARD_REDIS_URL") or None
    config = SecurityConfig(
        rate_limit=int(os.getenv("API_GUARD_RATE_LIMIT", "60")),
        rate_limit_window=int(os.getenv("API_GUARD_RATE_LIMIT_WINDOW", "60")),
        whitelist=_env_list("API_GUARD_WHITELIST") or None,
        blacklist=_env_list("API_GUARD_BLACKLIST"),
        trusted_proxies=_env_list("API_GUARD_TRUSTED_PROXIES", "127.0.0.1,::1"),
        detection_scan_body=is_env_enabled("API_GUARD_SCAN_BODY"),
        # Browsers send the page address as Referer, and guard-core 4.4.0 flags localhost and LAN addresses in it as ssrf.
        enabled_detection_categories=frozenset(ALL_DETECTION_CATEGORIES) - {"ssrf"},
        enable_redis=redis_url is not None,
        redis_url=redis_url,
        enable_cors=False,  # CORS is handled by CORSMiddleware
    )
    app.add_middleware(PureASGISecurityMiddleware, config=config)
