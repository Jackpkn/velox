"""Lightweight FastAPI + WebSocket server for the Vanilla HTML/CSS/JS Velox test console."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .agent import Agent
from .browser import ActionError, BrowserSession
from .fast import FastConfig, FastPolicy
from .llm import LLMClient, LLMConfig

log = logging.getLogger("velox")
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
DEFAULT_PROFILE_DIR = Path(os.environ.get("BROWSER_PROFILE_DIR", ".velox/browser-profile"))
_ACTIVE_SESSIONS: set[WebSession] = set()


class WebSession:
    def __init__(self, ws: WebSocket, llm: LLMClient, fast: FastPolicy | None) -> None:
        self.ws = ws
        self.llm = llm
        self.fast = fast
        self.browser: BrowserSession | None = None
        self.agent: Agent | None = None
        self.task: asyncio.Task | None = None
        self.persist_profile: bool = True
        self.closed = False
        _ACTIVE_SESSIONS.add(self)

    async def send(self, event: dict[str, Any]) -> None:
        if self.closed:
            return
        with contextlib.suppress(Exception):
            await self.ws.send_json(event)

    async def on_frame(self, frame: dict[str, Any]) -> None:
        await self.send({"type": "frame", **frame})

    async def ensure_browser(self, url: str = "https://www.bing.com/") -> BrowserSession:
        if self.browser is not None and self.browser.context is not None and self.browser._tabs_list():
            return self.browser
        if self.browser is not None:
            await self.browser.close()
        if self.persist_profile:
            for other in list(_ACTIVE_SESSIONS):
                if other is not self and other.browser is not None:
                    await other.browser.close()
                    other.browser = None
        self.browser = BrowserSession(on_frame=self.on_frame)
        profile = DEFAULT_PROFILE_DIR if self.persist_profile else None
        await self.browser.start(url, trusted=True, user_data_dir=profile)
        await self.browser.send_frame()
        return self.browser

    async def handle(self, msg: dict[str, Any]) -> None:
        cmd = str(msg.get("type") or "")
        try:
            if cmd == "open_url":
                url = str(msg.get("url") or "https://www.bing.com/").strip()
                self.persist_profile = bool(msg.get("persist", True))
                if self.browser is None or self.browser.context is None:
                    await self.ensure_browser(url)
                else:
                    await self.browser.navigate(url, trusted=False)
                    await self.browser.send_frame()
                await self.send({"type": "notice", "level": "info", "message": f"Opened {url}"})

            elif cmd == "observe":
                browser = await self.ensure_browser()
                obs = await browser.observe()
                await self.send({
                    "type": "observation",
                    "url": obs.url,
                    "title": obs.title,
                    "text": obs.text[:6000],
                    "element_count": len(obs.elements),
                    "elements": [
                        {"id": eid, "tag": el.get("tag"), "role": el.get("role"), "label": el.get("label")}
                        for eid, el in list(obs.elements.items())[:80]
                    ],
                })

            elif cmd == "run_goal":
                if self.task and not self.task.done():
                    await self.send({"type": "notice", "level": "error", "message": "A goal is already running."})
                    return
                goal = str(msg.get("goal") or "").strip()
                if not goal:
                    return
                start_url = str(msg.get("start_url") or "").strip()
                self.persist_profile = bool(msg.get("persist", True))
                browser = await self.ensure_browser(start_url or "https://www.bing.com/")
                if start_url and browser.page and browser.page.url != start_url:
                    with contextlib.suppress(ActionError):
                        await browser.navigate(start_url, trusted=False)
                self.agent = Agent(
                    browser=browser,
                    llm=self.llm,
                    emit=self.send,
                    goal=goal,
                    model=str(msg.get("model") or self.llm.config.default_model),
                    approval_mode=str(msg.get("mode") or "risky"),
                    max_steps=int(msg.get("max_steps") or 30),
                    fast=self.fast,
                )
                self.task = asyncio.create_task(self._run_agent(self.agent))

            elif cmd == "pause" and self.agent:
                await self.agent.pause()
            elif cmd == "resume" and self.agent:
                await self.agent.resume()
            elif cmd == "stop" and self.agent:
                await self.agent.stop()
            elif cmd in ("approve", "reject") and self.agent:
                self.agent.resolve_approval(str(msg.get("step_id") or ""), cmd == "approve", "")

            elif cmd == "click" and self.browser:
                res = await self.browser.human_click(float(msg["x"]), float(msg["y"]))
                await self.browser.send_frame()
                await self.send({"type": "notice", "level": "info", "message": res})

            elif cmd == "key" and self.browser:
                await self.browser.human_key(str(msg.get("key") or ""))
                await self.browser.send_frame()

            elif cmd == "type" and self.browser:
                res = await self.browser.human_type(str(msg.get("text") or ""), bool(msg.get("press_enter", False)))
                await self.browser.send_frame()
                await self.send({"type": "notice", "level": "info", "message": res})

            elif cmd == "scroll" and self.browser:
                await self.browser.human_scroll(float(msg.get("dy") or 300))
                await self.browser.send_frame()

            elif cmd == "switch_tab" and self.browser:
                await self.browser.switch_tab(int(msg.get("index") or 0))
                await self.browser.send_frame()

            elif cmd == "clear_profile":
                if self.browser:
                    await self.browser.close()
                    self.browser = None
                BrowserSession.clear_profile(DEFAULT_PROFILE_DIR)
                await self.send({
                    "type": "notice",
                    "level": "info",
                    "message": f"Cleared persistent profile ({DEFAULT_PROFILE_DIR}).",
                })
                await self.send({"type": "profile_status", "exists": DEFAULT_PROFILE_DIR.exists()})

        except ActionError as exc:
            await self.send({"type": "notice", "level": "error", "message": str(exc)})
        except Exception as exc:  # noqa: BLE001
            log.exception("command error")
            await self.send({"type": "notice", "level": "error", "message": f"Error: {exc}"})

    async def _run_agent(self, agent: Agent) -> None:
        try:
            result = await agent.run()
            state = "stopped" if agent.stopped else "done" if result.get("success") else "failed"
            await agent.set_state(state)
        except Exception as exc:  # noqa: BLE001
            log.exception("agent run failed")
            await self.send({"type": "notice", "level": "error", "message": f"Agent error: {exc}"})
            await self.send({"type": "status", "state": "failed", "step": agent.actions_taken})

    async def close(self) -> None:
        self.closed = True
        _ACTIVE_SESSIONS.discard(self)
        if self.agent:
            await self.agent.stop()
        if self.task and not self.task.done():
            self.task.cancel()
        if self.browser:
            await self.browser.close()
            self.browser = None


def _load_dotenv() -> None:
    for env_path in (Path(".env"), Path(__file__).resolve().parents[2] / ".env"):
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
            break


def create_app() -> FastAPI:
    _load_dotenv()
    app = FastAPI(title="Velox HTML/JS Test Console")
    config = LLMConfig.from_env()
    llm = LLMClient(config)
    fast_cfg = FastConfig.from_env()
    fast = FastPolicy(fast_cfg) if fast_cfg.enabled else None

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        session = WebSession(ws, llm, fast)
        await session.send({
            "type": "hello",
            "models": list(config.models),
            "default_model": config.default_model,
            "profile_dir": str(DEFAULT_PROFILE_DIR),
            "profile_exists": DEFAULT_PROFILE_DIR.exists(),
        })
        try:
            while True:
                data = await ws.receive_json()
                if isinstance(data, dict):
                    await session.handle(data)
        except WebSocketDisconnect:
            pass
        finally:
            await session.close()

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")

    return app


def main() -> None:
    os.environ.setdefault("HEADLESS", "1")
    port = int(os.environ.get("VELOX_PORT", "8788"))
    uvicorn.run(create_app(), host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
