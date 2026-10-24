"""Velox: high-speed browser automation engine with persistent sessions and CDP screencast."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from .agent import Agent
from .browser import BrowserSession
from .fast import FastConfig, FastPolicy
from .llm import LLMClient, LLMConfig

__all__ = ["Agent", "BrowserSession", "main"]


async def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Velox autonomous browser agent")
    parser.add_argument("goal", help="Natural-language goal to execute")
    parser.add_argument("--url", default="https://www.bing.com/", help="Starting URL")
    parser.add_argument("--model", default=None, help="Override planner model")
    parser.add_argument("--mode", choices=["risky", "always", "never"], default="risky")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument(
        "--profile-dir",
        default=".velox/browser-profile",
        help="Persistent Chromium user data directory (Cookies + localStorage + IndexedDB). Pass '' to disable.",
    )
    parser.add_argument("--clear-profile", action="store_true", help="Clear the persistent profile before starting")
    parser.add_argument("--headed", action="store_true", help="Open a visible Chromium window (HEADLESS=0)")
    parser.add_argument("--keep-open", action="store_true", help="Keep the browser window open after completion")
    args = parser.parse_args()

    if args.headed:
        os.environ["HEADLESS"] = "0"

    profile_dir = Path(args.profile_dir) if args.profile_dir else None
    if args.clear_profile and profile_dir:
        BrowserSession.clear_profile(profile_dir)

    config = LLMConfig.from_env()
    llm = LLMClient(config)
    fast_config = FastConfig.from_env()
    fast = FastPolicy(fast_config) if fast_config.enabled else None
    started = time.time()

    async def emit(event: dict) -> None:
        kind = event["type"]
        if kind == "step" and event["status"] in ("ok", "error", "rejected", "skipped"):
            outcome = event.get("error") or event.get("result")
            tag = "jev" if event.get("system") == "fast" else "s2 "
            print(
                f"[{time.time() - started:6.1f}s] {tag} #{event['n']} {event['status']:8} "
                f"{event.get('action', {}).get('label', '')} -> {outcome}",
                flush=True,
            )

    browser = BrowserSession()
    await browser.start(args.url, user_data_dir=profile_dir)
    agent = Agent(
        browser=browser,
        llm=llm,
        emit=emit,
        goal=args.goal,
        model=args.model or config.default_model,
        approval_mode=args.mode,
        max_steps=args.max_steps,
        fast=fast,
    )
    try:
        result = await agent.run()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.keep_open:
            print("Keeping browser window open (Ctrl+C to close)...", flush=True)
            await asyncio.sleep(3600)
    finally:
        await browser.close()
        await llm.close()
        if fast:
            await fast.close()


def main() -> None:
    asyncio.run(_run_cli())
