# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Round-robin reverse proxy for data-parallel vLLM serving."""

import asyncio
import logging

import aiohttp
from aiohttp import web


logger = logging.getLogger("ReverseProxy")

# Headers that must not be forwarded between hops.
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)


class RoundRobinProxy:
    """Reverse proxy that distributes requests across upstream vLLM servers."""

    def __init__(
        self,
        upstream_ports: list[int],
        listen_port: int,
        listen_host: str = "0.0.0.0",
    ):
        self.upstream_urls = [f"http://127.0.0.1:{p}" for p in upstream_ports]
        self.listen_port = listen_port
        self.listen_host = listen_host
        self._rr_counter = 0
        self._session: aiohttp.ClientSession | None = None
        self._runner: web.AppRunner | None = None

    def _next_upstream(self) -> str:
        url = self.upstream_urls[self._rr_counter % len(self.upstream_urls)]
        self._rr_counter += 1
        return url

    async def _health_handler(self, request: web.Request) -> web.Response:
        """Aggregate health check: returns 200 only if ALL backends are healthy."""
        assert self._session is not None
        unhealthy = []
        for url in self.upstream_urls:
            try:
                async with self._session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        unhealthy.append(url)
            except Exception:
                unhealthy.append(url)
        if unhealthy:
            return web.Response(status=503, text=f"Unhealthy backends: {unhealthy}")
        return web.Response(status=200)

    async def _proxy_handler(self, request: web.Request) -> web.StreamResponse:
        """Forward a request to the next upstream and stream the response back."""
        assert self._session is not None
        upstream = self._next_upstream()
        target_url = f"{upstream}{request.path_qs}"

        # Forward headers, filtering hop-by-hop.
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP_HEADERS}

        body = await request.read()

        try:
            upstream_resp = await self._session.request(
                method=request.method,
                url=target_url,
                headers=headers,
                data=body if body else None,
                timeout=aiohttp.ClientTimeout(total=None),
            )
        except aiohttp.ClientError as e:
            return web.Response(status=502, text=f"Upstream error: {e}")

        # Build response headers.
        resp_headers = {k: v for k, v in upstream_resp.headers.items() if k.lower() not in HOP_BY_HOP_HEADERS}

        # Stream the response body.
        response = web.StreamResponse(status=upstream_resp.status, headers=resp_headers)
        await response.prepare(request)
        async for chunk in upstream_resp.content.iter_any():
            await response.write(chunk)
        await response.write_eof()
        upstream_resp.release()
        return response

    async def wait_for_backends(self, timeout: float = 600, poll_interval: float = 2.0):
        """Block until all upstream servers respond to /health."""
        session = aiohttp.ClientSession()
        try:
            remaining = set(self.upstream_urls)
            deadline = asyncio.get_event_loop().time() + timeout
            while remaining:
                if asyncio.get_event_loop().time() > deadline:
                    raise TimeoutError(f"Backends not ready after {timeout}s: {remaining}")
                for url in list(remaining):
                    try:
                        async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                            if resp.status == 200:
                                logger.info("Backend ready: %s", url)
                                remaining.discard(url)
                    except Exception:
                        pass
                if remaining:
                    await asyncio.sleep(poll_interval)
        finally:
            await session.close()

    async def run(self):
        """Start the reverse proxy server. Blocks until shutdown."""
        self._session = aiohttp.ClientSession()
        app = web.Application()
        app.router.add_route("*", "/health", self._health_handler)
        app.router.add_route("*", "/{path_info:.*}", self._proxy_handler)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.listen_host, self.listen_port)
        await site.start()
        logger.info("Reverse proxy listening on %s:%d", self.listen_host, self.listen_port)

        # Block until cancelled.
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    async def shutdown(self):
        if self._session:
            await self._session.close()
            self._session = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
