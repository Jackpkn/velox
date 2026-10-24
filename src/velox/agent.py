"""The observe → decide → (approve) → act loop, with live events and operator control."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from playwright.async_api import Error as PlaywrightError

from .anthropic_backend import _recover_xml_params
from .browser import ActionError, BrowserSession, Observation
from .extract import ExtractMixin, normalize_url, resolve_urls
from .fast import FastPolicy
from .llm import Decision, LLMClient, LLMError
from .prompts import CHARS_PER_TOKEN, SYSTEM, build_system_prompt, context_breakdown, step_context
from .system1 import FastLoopMixin
from .telemetry import RunLedger
from .tools import TOOL_NAMES, TOOLS, TOOLS_WITH_FAST, label, missing_args, needs_approval

Emit = Callable[[dict[str, Any]], Awaitable[None]]
log = logging.getLogger(__name__)
SEARCH_URL = "https://www.bing.com/search?q={}"
MAX_CONSECUTIVE_ERRORS = 3
MAX_CHECKED = 40


class Stopped(Exception):
    """The operator stopped the run."""


class Agent(FastLoopMixin, ExtractMixin):
    def __init__(
        self, *, browser: BrowserSession, llm: LLMClient, emit: Emit, goal: str, model: str,
        approval_mode: str = "risky", max_steps: int = 40, vision: bool = False, search_url: str = SEARCH_URL,
        max_cost_usd: float = 3.0, fast: FastPolicy | None = None, timezone: str | None = None,
        domain_memory: dict[str, list[str]] | None = None, conversation: list[dict[str, Any]] | None = None,
        user_memory: list[str] | None = None,
    ):
        self.browser, self.llm, self.emit = browser, llm, emit
        self.goal, self.model, self.approval_mode = goal, model, approval_mode
        self.max_steps, self.vision, self.search_url = max_steps, vision, search_url
        self.max_cost_usd = max_cost_usd
        self.timezone = timezone
        self.domain_memory = domain_memory or {}
        self.conversation = conversation or []  # earlier turns of this chat, summarised by the console
        self.user_memory = user_memory or []  # facts the operator asked to be remembered across chats
        self.fast = fast if fast and fast.config.enabled else None
        self.system_prompt = build_system_prompt(use_jev=bool(self.fast))
        self.detached = False  # the operator's socket closed; skip anything that only serves them
        self.plan: list[dict[str, str]] = []
        self.history: list[dict[str, Any]] = []
        self.findings: list[dict[str, Any]] = []
        self._findings_at_start = 0
        self.checked: dict[str, str] = {}  # page → what was saved from it, including "nothing"
        self.extracted: set[tuple[str, frozenset[str]]] = set()  # (page, fields + filter) already run through extract
        self.guidance: list[str] = []
        self.guidance_version = 0
        self.read_output: dict[str, Any] | None = None
        self.paused = False
        self.stopped = False
        self._wake = asyncio.Event()
        self._approval: tuple[str, asyncio.Future] | None = None
        self._question: asyncio.Future | None = None
        self.state = "idle"
        self.n = 0
        self.actions_taken = 0
        self.usage = {"llm_calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                      "fast_calls": 0, "fast_ms_total": 0, "text_calls": 0, "extract_calls": 0}
        self.ledger = RunLedger()
        self.started = time.time()
        self.result: dict[str, Any] | None = None
        self._last_page_sig: str | None = None
        self._consecutive_errors = 0
        self.memory = ""
        self._observed_corpus = ""
        self._learned_domain_rules: dict[str, list[str]] = {}

    # ---------- operator controls (called from the socket handler) ----------

    async def pause(self) -> None:
        if not self.paused and not self.stopped:
            self.paused = True
            self.guidance_version += 1  # a quick pause/resume must still invalidate pending decisions
            await self.notice("info", "Pausing. I'll finish the current action, then wait for you.")
            if self.state in ("thinking", "running", "acting", "awaiting_approval", "awaiting_user", "starting"):
                return
            await self.set_state("paused")

    async def resume(self) -> None:
        if self.paused:
            self.paused = False
            self._consecutive_errors = 0
            self._wake.set()
            await self.notice("info", "Resuming.")

    async def stop(self) -> None:
        self.stopped = True
        self._wake.set()
        for fut in (self._approval[1] if self._approval else None, self._question):
            if fut and not fut.done():
                fut.set_exception(Stopped())

    def _remember_domain_rule(self, rule_text: str, url: str = "") -> str | None:
        raw_url = url or str(getattr(getattr(self.browser, "page", None), "url", "") or "")
        domain = (urlsplit(raw_url).hostname or "").lower().removeprefix("www.")
        if not domain or domain in ("bing.com", "google.com", "duckduckgo.com", "localhost"):
            return None
        clean = rule_text.strip()[:160]
        if not clean:
            return None
        note = f"Operator rule on {domain}: {clean}"
        self._learned_domain_rules.setdefault(domain, []).append(note)
        self.domain_memory.setdefault(domain, []).append(note)
        return domain

    async def guide(self, text: str) -> None:
        if self._question and not self._question.done():
            self._question.set_result(text)
            return
        self.guidance.append(text)
        self.guidance_version += 1
        domain = self._remember_domain_rule(text)
        suffix = f" (Remembered rule for {domain})" if domain else ""
        await self.notice("info", f"Got it. I'll take that into account from the next step.{suffix}")

    def resolve_approval(self, step_id: str, approved: bool, reason: str = "") -> bool:
        if not self._approval or self._approval[0] != step_id or self._approval[1].done():
            return False
        self._approval[1].set_result((approved, reason))
        return True

    async def operator_action(self, description: str) -> None:
        """Record a manual takeover so the model knows the page was changed by the person."""
        self.history.append({"n": self.n, "label": "Operator took over", "result": description})
        await self.notice("info", description)

    # ---------- events ----------

    async def set_state(self, state: str, detail: str = "") -> None:
        self.state = state
        await self.emit({"type": "status", "state": state, "step": self.actions_taken, "detail": detail})

    async def notice(self, level: str, message: str) -> None:
        await self.emit({"type": "notice", "level": level, "message": message})

    async def send_step(self, step: dict[str, Any]) -> None:
        self.ledger.note_step(step)
        await self.emit({"type": "step", **step})

    async def send_usage(self) -> None:
        await self.emit({"type": "usage", **self.usage, "cost_usd": round(self.usage["cost_usd"], 5),
                         "elapsed_s": round(time.time() - self.started, 1),
                         "accounting": self.ledger.snapshot(details=False)})

    # ---------- main loop ----------

    async def run(self) -> dict[str, Any]:
        # A continued chat starts with the previous turn's findings; only rows saved after this point are this turn's.
        self._findings_at_start = len(self.findings)
        try:
            while not self.stopped and self.result is None:
                if self.actions_taken >= self.max_steps:
                    await self.notice("warn", f"Reached the {self.max_steps}-step limit. Wrapping up with what I have.")
                    break
                if self.usage["cost_usd"] >= self.max_cost_usd:
                    await self.notice("warn", f"Reached the ${self.max_cost_usd:.2f} cost limit for one task. "
                                              "Wrapping up with what I have.")
                    break
                await self._wait_if_paused()
                if self.stopped:
                    break
                await self._step()
        except Stopped:
            pass
        if self.result is None:
            reason = ("Stopped before the task finished." if self.stopped
                      else f"Hit the cost limit (${self.max_cost_usd:.2f}) before finishing."
                      if self.usage["cost_usd"] >= self.max_cost_usd
                      else f"Ran out of steps ({self.max_steps}) before finishing.")
            await self._finalize(reason)
        return self.result

    async def _wait_if_paused(self) -> None:
        while self.paused and not self.stopped:
            self._wake.clear()  # before any await, so a resume() during set_state is not lost
            await self.set_state("paused", "Paused. Take over in the browser, give guidance, or resume.")
            if self.paused and not self.stopped:
                await self._wake.wait()

    def _needs_vision(self) -> bool:
        """The indexed element list is what the planner acts on, so screenshots stay off by default. With vision
        enabled, one is attached only when that text failed it: repeated errors or a click that changed nothing."""
        if not self.vision or not self.history:
            return False
        last = self.history[-1]
        return self._consecutive_errors >= 2 or "did not visibly change" in str(last.get("result", ""))

    def _planner_tools(self) -> list[dict[str, Any]]:
        """Every direct tool, always; Jev only adds delegate. The prompt says when to delegate and when a single
        click is the right call, so the planner never has to improvise around a missing tool."""
        return TOOLS_WITH_FAST if self.fast else TOOLS

    def _hard_turn(self) -> bool:
        """Most steps are obvious, so native thinking is paid for only on the first plan, after a failure or an
        unfinished Jev hand-off, and when saved findings are about to be compared on the plan's last item."""
        if not self.history or self._consecutive_errors:
            return True
        last = str(self.history[-1].get("result", ""))
        if "did not visibly change" in last or _unfinished_handback(last):
            return True
        if len(self.findings) >= 4:
            return True
        return bool(self.findings and self.plan and self.plan[-1].get("status") == "active")

    def _remember_text(self, text: str) -> None:
        if not text:
            return
        combined = f"{self._observed_corpus}\n{text}"
        self._observed_corpus = combined[-200000:]

    async def _step(self) -> None:
        self.n += 1
        step: dict[str, Any] = {"id": f"s{self.n}", "n": self.n, "status": "thinking", "system": "planner",
                                "started_at": time.time()}
        await self.set_state("thinking", "The planner is deciding the next step…")
        try:
            page = await self.browser.observe(screenshot=self._needs_vision())
        except (ActionError, PlaywrightError) as exc:
            await self._fail_step(step, _browser_error(exc))
            return
        self._remember_text(f"{page.title}\n{page.text}")
        self._mark_no_change(page)
        step["url"] = page.url
        await self.send_step(step)
        version = self.guidance_version
        decision = await self._decide(page, step)
        if decision is None:
            step.update(status="skipped", result="Could not reach the model.", ended_at=time.time())
            await self.send_step(step)
            await self.send_usage()
            return
        if self.stopped:
            step.update(status="skipped", result="Stopped before acting.", ended_at=time.time())
            await self.send_step(step)
            await self.send_usage()
            raise Stopped()
        if self.guidance_version != version or self.paused:
            why = "New guidance arrived; re-planning with it." if self.guidance_version != version else \
                "Paused before acting; this decision was discarded so nothing stale runs."
            step.update(status="skipped", result=why, ended_at=time.time())
            await self.send_step(step)
            await self.send_usage()
            return
        await self._handle(step, decision, page, version=version)

    async def _decide(self, page: Observation, step: dict[str, Any] | None = None) -> Decision | None:
        events, self.browser.events = self.browser.events, []
        domain = (urlsplit(page.url).hostname or "").lower().removeprefix("www.")
        domain_notes = self.domain_memory.get(domain, []) if domain else []
        tools = self._planner_tools()
        context = step_context(
            goal=self.goal, guidance=self.guidance, plan=self.plan, history=self.history, findings=self.findings,
            page=page, read_output=self.read_output, warnings=self._loop_warnings() + _page_warnings(page),
            step=self.actions_taken + 1,
            max_steps=self.max_steps, events=events, memory=self.memory, now=local_now(self.timezone),
            domain_notes=domain_notes, conversation=self.conversation, user_memory=self.user_memory,
            checked=self.checked,
        )
        self.read_output = None
        content: list[dict[str, Any]] = [{"type": "text", "text": context}]
        if page.screenshot:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{page.screenshot}"}})
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": content}]
        last_delta_at = 0.0

        async def _on_delta(partial: dict[str, str]) -> None:
            nonlocal last_delta_at
            if step is None or self.stopped or step.get("status") != "thinking":
                return
            now = time.monotonic()
            if now - last_delta_at < 0.08:
                return
            last_delta_at = now
            step["thought"] = {**step.get("thought", {}), **partial}
            await self.send_step(step)

        with self.ledger.scope("planner", step.get("id") if step else None):
            call = asyncio.create_task(
                self.llm.decide(self.model, messages, tools, on_retry=self._retry_notice,
                                on_delta=_on_delta, hard=self._hard_turn())
            )
        stop_watch = asyncio.create_task(self._until_stopped())
        try:
            done, _ = await asyncio.wait({call, stop_watch}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop_watch.cancel()
            if not call.done():
                call.cancel()  # never leave a billed request running for a decision nobody will use
                await asyncio.gather(call, return_exceptions=True)  # settle accounting before publishing the result
        if call not in done:
            return None
        try:
            decision = call.result()
        except LLMError as exc:
            await self.notice("error", f"{exc} I've paused; press Resume to try again.")
            self.paused = True
            return None
        self._count(decision)
        self._log_call(decision, tools, context)
        return decision

    def _log_call(self, d: Decision, tools: list[dict[str, Any]], context: str) -> None:
        """One server log line per planner call: estimated tokens per prompt part next to what the API billed, so
        it is visible what each call is actually spent on."""
        est = lambda text: round(len(text) / CHARS_PER_TOKEN)  # noqa: E731
        parts = {"system": est(self.system_prompt), "tools": est(json.dumps(tools)), **context_breakdown(context)}
        shown = " ".join(f"{k}={v}" for k, v in parts.items())
        log.info("planner call step=%d model=%s tools=%d | est tok: %s | est total=%d | actual in=%d "
                 "(cached %d) out=%d thinking=%s", self.actions_taken + 1, self.model, len(tools), shown,
                 sum(parts.values()), d.input_tokens, d.cached_tokens, d.output_tokens,
                 "yes" if d.thinking else "no")

    async def _until_stopped(self) -> None:
        while not self.stopped:
            await asyncio.sleep(0.1)

    async def _retry_notice(self, message: str) -> None:
        await self.notice("warn", message)

    def _count(self, d: Decision) -> None:
        self.usage["llm_calls"] += 1
        self.usage["input_tokens"] += d.input_tokens
        self.usage["cached_input_tokens"] += d.cached_tokens
        self.usage["output_tokens"] += d.output_tokens
        self.usage["cost_usd"] += d.cost_usd

    async def _handle(self, step: dict[str, Any], d: Decision, page: Observation, *,
                      version: int | None = None) -> None:
        await self.send_usage()
        args = _recover_xml_params(d.args)
        if d.name not in TOOL_NAMES:
            await self._fail_step(step, f"The model chose an unknown action {d.name!r}; nothing was executed.")
            return
        missing = missing_args(d.name, args)
        if missing:
            step.update(thought=_thought(args, d.thinking), action=_action(d.name, args, ""))
            await self._fail_step(step, f"{d.name} was chosen without {', '.join(missing)}; nothing was executed.")
            return
        if args.get("memory"):
            self.memory = _text(args["memory"])[:1500]
        if isinstance(args.get("plan"), list) and args["plan"]:
            # Plain strings are the schema; objects with a status are still accepted from older models.
            self.plan = [{"text": str(p.get("text", ""))[:160], "status": p.get("status", "pending")}
                         if isinstance(p, dict) else {"text": str(p)[:160], "status": "pending"}
                         for p in args["plan"] if isinstance(p, dict | str) and p][:10]
            await self.emit({"type": "plan", "items": self.plan})
        await self._advance_plan(args.get("plan_step"))
        target_label = ""
        secret = False
        if "element_id" in args:
            try:
                element = page.element(int(args["element_id"]))
                target_label = element["label"] or element["tag"]
                secret = d.name == "type_text" and element.get("type") == "password"
                rect = await self.browser.element_rect(int(args["element_id"]))
                step["target"] = {"label": target_label, "rect": rect or element["rect"]}
            except (ActionError, PlaywrightError, ValueError, TypeError) as exc:
                step.update(thought=_thought(args, d.thinking), action=_action(d.name, args, ""))
                await self._fail_step(step, _browser_error(exc))
                return
        shown = {**args, "text": "••••••••"} if secret else args
        step.update(thought=_thought(args, d.thinking), action=_action(d.name, shown, target_label), status="planned")
        if await self._discard_interrupted(step, version):
            return
        risky, reason = needs_approval(d.name, args, target_label, self.approval_mode)
        if secret:
            # Credentials are always the operator's call, whatever the approval mode; the value is never displayed.
            step["secret"] = True
            risky, reason = True, "This is a password field. The agent only types into it with your approval."
        if risky:
            step["risk"] = {"risky": True, "reason": reason}
            if not await self._approve(step, reason):
                return
        if self.stopped:
            step.update(status="skipped", result="Stopped before acting.", ended_at=time.time())
            await self.send_step(step)
            raise Stopped()
        await self._execute(step, d.name, args, version=version)

    async def _discard_interrupted(self, step: dict[str, Any], version: int | None) -> bool:
        if self.stopped or self.paused or (version is not None and version != self.guidance_version):
            step.update(status="skipped", result="Discarded because the operator interrupted the decision.",
                        ended_at=time.time())
            await self.send_step(step)
            return True
        return False

    async def _approve(self, step: dict[str, Any], reason: str) -> bool:
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._approval = (step["id"], fut)  # registered before any await so stop() can always cancel it
        if self.stopped:
            fut.set_exception(Stopped())
        step["status"] = "awaiting_approval"
        await self.send_step(step)
        await self.emit({"type": "approval", "step_id": step["id"], "reason": reason,
                         "action_label": step["action"]["label"],
                         "target_label": step.get("target", {}).get("label"),
                         "args": {k: v for k, v in step["action"]["args"].items() if k != "element_id"}})
        await self.set_state("awaiting_approval", reason)
        try:
            approved, why = await fut
        except Stopped:
            step.update(status="skipped", result="Stopped while waiting for your approval.", ended_at=time.time())
            await self.send_step(step)
            raise
        finally:
            self._approval = None
        if not approved:
            text = "You rejected this step" + (f": {why}" if why else ".")
            step.update(status="rejected", result=text, ended_at=time.time())
            await self.send_step(step)
            if why:
                self._remember_domain_rule(f"Do not {step['action']['label']} ({why})", str(step.get("url") or ""))
            self.history.append({"n": step["n"], "label": step["action"]["label"],
                                 "result": f"REJECTED by the operator. {why}".strip()})
            return False
        return True

    async def _execute(self, step: dict[str, Any], name: str, args: dict[str, Any], *,
                       version: int | None = None) -> None:
        step["status"] = "running"
        await self.send_step(step)
        await self.set_state("acting", step["action"]["label"])
        if step.get("target") and getattr(self.browser, "on_frame", None) is not None:
            await asyncio.sleep(0.22)
        if await self._discard_interrupted(step, version):
            return
        try:
            result = await self._run_tool(name, args, step)
        except ActionError as exc:
            await self._fail_step(step, str(exc))
            return
        except Stopped:
            step.update(status="skipped", result="Stopped.", ended_at=time.time())
            await self.send_step(step)
            raise
        except PlaywrightError as exc:
            await self._fail_step(step, _browser_error(exc))
            return
        except Exception as exc:  # noqa: BLE001 - any browser failure becomes a visible, recoverable step error
            await self._fail_step(step, f"Unexpected browser error: {type(exc).__name__}: {str(exc)[:160]}")
            return
        if getattr(self.browser, "on_frame", None) is not None and hasattr(self.browser, "send_frame"):
            try:
                await self.browser.send_frame()
            except Exception:  # noqa: BLE001
                pass
        self._consecutive_errors = 0
        self.actions_taken += 1
        step.update(status="ok", result=result, ended_at=time.time())
        await self.send_step(step)
        if name == "done":
            await self._publish_result()
        self.history.append({"n": step["n"], "label": step["action"]["label"], "result": result,
                             "name": name, "url": step.get("url", ""),
                             "sig": _signature(name, args, step.get("url", ""))})

    async def _run_tool(self, name: str, args: dict[str, Any], step: dict[str, Any]) -> str:
        b = self.browser
        eid = lambda: int(args["element_id"])  # noqa: E731
        match name:
            case "search":
                return await b.navigate(self.search_url.format(quote_plus(str(args.get("query", "")))))
            case "navigate":
                return await b.navigate(str(args.get("url", "")))
            case "click":
                return await b.click(eid())
            case "type_text":
                return await b.type_text(eid(), str(args.get("text", "")), bool(args.get("press_enter")),
                                         secret_ok=bool(step.get("secret")))
            case "select_option":
                return await b.select_option(eid(), str(args.get("option", "")))
            case "press_key":
                return await b.press_key(str(args.get("key", "Enter")))
            case "scroll":
                return await b.scroll(str(args.get("direction", "down")), float(args.get("pages") or 1))
            case "go_back":
                return await b.go_back()
            case "switch_tab":
                return await b.switch_tab(int(args.get("tab_index", 0)))
            case "close_tab":
                return await b.close_tab()
            case "read_page":
                self.read_output = await b.read_page()
                self._remember_text(str(self.read_output.get("text", "")))
                return f"Read {self.read_output['length']:,} characters of page text."
            case "read_urls_parallel":
                raw_urls = args.get("urls") if isinstance(args.get("urls"), list) else []
                clean_urls = resolve_urls(b.page.url, raw_urls, limit=4)
                fresh, skipped = self._split_already_read(clean_urls)
                note = (f" Skipped {len(skipped)} already saved in findings (re-reading them only repeats what "
                        "findings_saved holds)." if skipped else "")
                if not fresh:
                    return "Nothing read: every page's facts are already saved in findings." + note
                self.read_output = await b.read_urls_parallel(fresh)
                self._remember_text(str(self.read_output.get("text", "")))
                return (f"Read {len(fresh)} page{'s' if len(fresh) != 1 else ''} in parallel "
                        f"({self.read_output['length']:,} characters).{note}")
            case "extract":
                return await self._extract(args, step)
            case "record_findings":
                return await self._record(args, step)
            case "wait":
                await asyncio.sleep(max(1.0, min(float(args.get("seconds") or 2), 10.0)))
                return "Waited."
            case "delegate":
                if not self.fast:
                    raise ActionError("The fast System 1 policy is not configured on this server.")
                return await self._delegate(str(args.get("subgoal", "")), step)
            case "ask_user":
                return await self._ask(str(args.get("question", "")), step)
            case "done":
                rows = args.get("rows")
                saved_this_turn = len(self.findings) > self._findings_at_start
                if not saved_this_turn and isinstance(rows, list) and rows:
                    # A table read straight off the page still needs saved findings to link each value to its
                    # source; _record checks every value against the observed page text and flags misses.
                    await self._record({"items": rows, "source_url": self.browser.page.url}, step)
                self.result = self._build_result(args)
                if self.result.get("success"):
                    await self._set_plan([p if p["status"] == "skipped" else {**p, "status": "done"}
                                          for p in self.plan])
                return "Delivered the result."
        raise ActionError(f"Unsupported action {name}")

    async def _advance_plan(self, plan_step: Any) -> None:
        """Earlier items become done, the named item active; the model sends one integer instead of the plan."""
        if isinstance(plan_step, bool) or not isinstance(plan_step, int | float) or not self.plan:
            return
        active = int(plan_step) - 1
        if not 0 <= active < len(self.plan):
            return

        def status(i: int, current: str) -> str:
            if current == "skipped":
                return current
            if i < active:
                return "done"
            if i == active:
                return "active"
            return "pending" if current == "active" else current

        await self._set_plan([{**p, "status": status(i, p["status"])} for i, p in enumerate(self.plan)])

    async def _set_plan(self, plan: list[dict[str, str]]) -> None:
        if plan != self.plan:
            self.plan = plan
            await self.emit({"type": "plan", "items": self.plan})

    async def _record(self, args: dict[str, Any], step: dict[str, Any] | None = None) -> str:
        items = args.get("items") if isinstance(args.get("items"), list) else []
        source = str(args.get("source_url") or self.browser.page.url)
        added = 0
        suspect_fields: list[str] = []
        for item in items:
            if not isinstance(item, dict) or not item:
                continue
            data = {str(k)[:60]: _text(v) for k, v in item.items()}
            unverified = [k for k, v in data.items() if not _verify_field(v, self._observed_corpus)]
            entry: dict[str, Any] = {"source_url": source, "data": data}
            if step and step.get("id"):
                entry["step_id"] = str(step["id"])
            if step and step.get("n") is not None:
                entry["step_n"] = int(step["n"])
            if unverified:
                entry["unverified"] = unverified
                for k in unverified:
                    if k not in suspect_fields:
                        suspect_fields.append(k)
            same_page = [i for i, f in enumerate(self.findings) if f["source_url"] == source]
            if any(_contains(self.findings[i]["data"], data) for i in same_page):
                continue  # already saved, possibly with more fields
            partial = next((i for i in same_page if _contains(data, self.findings[i]["data"])), None)
            if partial is None:
                self.findings.append(entry)
            else:
                self.findings[partial] = entry  # a richer copy of the same item replaces the partial one
            added += 1
        await self.emit({"type": "findings", "items": self.findings})
        self._mark_checked(source, f"recorded {added} new row{'s' if added != 1 else ''}" if items
                           else "checked, nothing on it to record")
        suffix = (f" (⚠ unverified in page text: {', '.join(suspect_fields[:3])})"
                  if suspect_fields else " (✓ verified against page text)")
        return f"Saved {added} new finding{'s' if added != 1 else ''} ({len(self.findings)} total){suffix}."

    def _mark_checked(self, url: str, note: str) -> None:
        """Remember that a page was covered even when nothing on it qualified, so it is not re-read."""
        key = normalize_url(url)
        self.checked.pop(key, None)
        self.checked[key] = note
        while len(self.checked) > MAX_CHECKED:
            self.checked.pop(next(iter(self.checked)))

    async def _ask(self, question: str, step: dict[str, Any]) -> str:
        self._question = asyncio.get_running_loop().create_future()
        if self.stopped:
            raise Stopped()
        await self.emit({"type": "question", "step_id": step["id"], "question": question})
        await self.set_state("awaiting_user", question)
        try:
            answer = await self._question
        finally:
            self._question = None
        self.guidance.append(f"Answer to your question {question!r}: {answer}")
        return f"You answered: {answer}"

    async def _fail_step(self, step: dict[str, Any], error: str) -> None:
        self._consecutive_errors += 1
        step.update(status="error", error=error, ended_at=time.time())
        await self.send_step(step)
        self.history.append({"n": step["n"], "label": step.get("action", {}).get("label", "Read the page"),
                             "error": error})
        if self._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            self.paused = True
            await self.notice("error", f"I'm stuck: {MAX_CONSECUTIVE_ERRORS} actions in a row failed. Last error: "
                                       f"{error} Give me a hint, take over in the browser, or stop.")

    # ---------- recovery signals ----------

    def _mark_no_change(self, page: Observation) -> None:
        sig = hashlib.sha1(f"{page.url}\n{page.text}".encode()).hexdigest()
        last = self.history[-1] if self.history else None
        if last and sig == self._last_page_sig and last.get("name") in ("click", "press_key", "select_option"):
            last["result"] += " (but the page did not visibly change)"
        self._last_page_sig = sig

    def _loop_warnings(self) -> list[str]:
        recent = [h.get("sig") for h in self.history[-8:] if h.get("sig")]
        if not recent:
            return []
        top = max(set(recent), key=recent.count)
        count = recent.count(top)
        if count >= 3:
            return [f"You have repeated the same action {count} times recently without finishing. "
                    "Change approach: different element, different page, or call done/ask_user."]
        return []

    # ---------- results ----------

    def _build_result(self, args: dict[str, Any], *, partial_reason: str = "") -> dict[str, Any]:
        rows = [r for r in (args.get("rows") or []) if isinstance(r, dict)]
        columns = [str(c) for c in (args.get("columns") or []) if str(c).strip()]
        if not rows and self.findings and partial_reason:
            rows = [f["data"] for f in self.findings]
        if rows and not columns:
            columns = list(dict.fromkeys(k for r in rows for k in r))
        rows = [{c: _text(r.get(c, "")) for c in columns} for r in rows]
        sources = [{"title": _text(s.get("title") or s.get("url")), "url": str(s.get("url"))}
                   for s in (args.get("sources") or []) if isinstance(s, dict)
                   and str(s.get("url", "")).startswith(("http://", "https://"))]
        known = {s["url"] for s in sources}
        for f in self.findings:
            if f["source_url"] not in known and f["source_url"].startswith("http"):
                sources.append({"title": f["source_url"], "url": f["source_url"]})
                known.add(f["source_url"])
        issues = [_text(i) for i in (args.get("issues") or []) if _text(i)]
        if partial_reason:
            issues.insert(0, partial_reason)
        checklist = self._checklist(args.get("checklist"))
        unmet = [c["requirement"] for c in checklist if not c["met"]]
        success = bool(args.get("success")) and not partial_reason
        if success and unmet:
            # The agent's own checklist outranks its success claim: never report a goal done with open items.
            success = False
            issues.insert(0, "Not every requirement was met: " + "; ".join(unmet[:4]))
        return {
            "success": success,
            "summary": str(args.get("summary") or "").strip() or "No summary was produced.",
            "columns": columns, "rows": rows, "sources": sources[:30], "issues": issues,
            "checklist": checklist, "assumptions": _strings(args.get("assumptions"), 6, 240),
            "follow_ups": _strings(args.get("follow_ups"), 3, 160),
            "remember": self._grounded_memories(args.get("remember")),
        }

    def _grounded_memories(self, raw: Any) -> list[str]:
        """Keep only memories the operator actually stated (goal, guidance, earlier goals), never page content."""
        said = "\n".join([self.goal, *self.guidance, *(str(t.get("goal", "")) for t in self.conversation)])
        kept = []
        for item in _strings(raw, 3, 160):
            # Meaningful words only (3+ characters): "as", "in" and the like would make any text look grounded.
            words = [w for w in re.findall(r"[a-z0-9£$€₹%]{3,}", item.lower()) if w not in _MEMORY_FRAMING]
            if words and sum(w in said.lower() for w in words) / len(words) >= 0.6:
                kept.append(item)
        return kept

    def _checklist(self, raw: Any) -> list[dict[str, Any]]:
        """The model's requirement checklist; evidence steps are kept only if they really ran in this run."""
        ran = {h["n"] for h in self.history if isinstance(h.get("n"), int)}
        out: list[dict[str, Any]] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict) or not _text(item.get("requirement")):
                continue
            entry: dict[str, Any] = {"requirement": _text(item["requirement"])[:160],
                                     "met": item.get("met") is True,
                                     "evidence": _text(item.get("evidence"))[:240]}
            n = item.get("step")
            if isinstance(n, int) and not isinstance(n, bool) and n in ran:
                entry.update(step_id=f"s{n}", step_n=n)
            out.append(entry)
        return out[:8]

    async def _publish_result(self) -> None:
        self.result["accounting"] = self.ledger.snapshot()
        await self.emit({"type": "result", **self.result})
        await self.send_usage()

    async def _finalize(self, reason: str) -> None:
        """Produce an honest partial result from the evidence gathered so far."""
        await self.set_state("finishing", "Writing up what I found so far.")
        args: dict[str, Any] = {}
        if self.llm.config.api_key and (self.findings or self.history) and not self.detached:
            context = (f"<goal>\n{self.goal}\n</goal>\n<history>\n"
                       + "\n".join(f"{h['n']}. {h['label']} → {h.get('error') or h.get('result', '')}"
                                   for h in self.history[-40:])
                       + f"\n</history>\n<findings_saved>\n{json.dumps(self.findings, ensure_ascii=False)[:20000]}"
                       f"\n</findings_saved>\nThe task ended early: {reason} Call done now with success=false, "
                       "summarising only what the findings and history support.")
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": context}]
            try:
                with self.ledger.scope("planner"):
                    decision = await self.llm.decide(self.model, messages, TOOLS, force="done")
                self._count(decision)
                args = decision.args
            except LLMError as exc:
                await self.notice("error", f"Could not write a summary: {exc}")
        if not args:
            args = {"summary": "The task ended before a result was produced. Saved findings are listed below."}
        self.result = self._build_result(args, partial_reason=reason)
        await self._publish_result()

    def distill_domain_notes(self) -> dict[str, list[str]]:
        """Extract concise per-domain workflow notes and operator rules from this run."""
        notes: dict[str, list[str]] = {d: list(dict.fromkeys(rules))[:5]
                                       for d, rules in self._learned_domain_rules.items() if rules}
        if not self.result or not (self.result.get("success") or self.findings):
            return notes
        by_domain: dict[str, list[str]] = {}
        for h in self.history:
            if h.get("error") or h.get("name") in ("done", "wait", "record_findings", "ask_user", None):
                continue
            domain = (urlsplit(str(h.get("url") or "")).hostname or "").lower().removeprefix("www.")
            if not domain or domain in ("bing.com", "google.com", "duckduckgo.com"):
                continue
            by_domain.setdefault(domain, []).append(str(h.get("label", ""))[:80])
        for domain, actions in by_domain.items():
            deduped = list(dict.fromkeys(a for a in actions if a))[:6]
            if deduped:
                notes.setdefault(domain, []).append(f"Working flow on {domain}: {' → '.join(deduped)}"[:240])
        return notes


BOT_WALL = re.compile(r"verify (that )?you are (a )?human|are you a robot|unusual traffic|\bcaptcha\b"
                      r"|press (and|&) hold|checking your browser|just a moment\.\.\.|access denied|request blocked",
                      re.I)


def _page_warnings(page: Observation) -> list[str]:
    title_hit = bool(BOT_WALL.search(page.title or ""))
    body_hit = len(page.elements) < 20 and bool(BOT_WALL.search((page.text or "")[:2000]))
    if title_hit or body_hit:
        return ["This page looks like a bot check or block page. Do not try to solve it: get the same information "
                "from a different site or search engine, or ask_user to solve it via takeover."]
    return []


def local_now(timezone: str | None) -> str:
    """The operator's current date and time, with the weekday spelled out so the model never has to infer it."""
    try:
        zone = ZoneInfo(timezone) if timezone else None
    except (ZoneInfoNotFoundError, ValueError):
        zone = None
    now = datetime.now(zone) if zone else datetime.now().astimezone()
    return now.strftime("%A %d %B %Y, %H:%M") + f" ({timezone if zone else now.tzname()})"


def _unfinished_handback(result: str) -> bool:
    """A delegate result where Jev gave the page back without claiming the subgoal is done."""
    return "Returned control because" in result and "subgoal is done" not in result


def _browser_error(exc: Exception) -> str:
    if isinstance(exc, ActionError):
        return str(exc)
    first = (str(exc).splitlines() or [""])[0][:160]
    if "context was destroyed" in first or "navigat" in first:
        return "The page navigated while I was acting on it. I'll look again."
    return f"Browser error: {first}"


def _thought(args: dict[str, Any], native_thinking: str = "") -> dict[str, str]:
    evaluation = _text(args.get("evaluation", ""))
    memory = _text(args.get("memory", ""))
    message = _text(args.get("message", ""))
    next_goal = _text(args.get("next_goal", "")) or message
    think = _text(args.get("think", ""))
    out = {
        "evaluation": evaluation,
        "memory": memory,
        "next_goal": next_goal,
        "message": message,
    }
    if think:
        out["think"] = think
    # The optional think field and native thinking are both reasoning the timeline shows under "thinking".
    thinking = "\n\n".join(t for t in (_text(native_thinking), think) if t)
    if thinking:
        out["thinking"] = thinking
    return out


# Words a model adds when restating a preference; they are not evidence the user said it.
_MEMORY_FRAMING = {"the", "user", "users", "prefers", "prefer", "wants", "want", "likes", "like", "always",
                   "their", "they", "should", "to", "be", "in", "a", "an", "of", "for", "and", "is"}


def _contains(big: dict[str, Any], small: dict[str, Any]) -> bool:
    """True when every field of `small` is present in `big` with the same value."""
    return all(big.get(k) == v for k, v in small.items())


def _strings(raw: Any, limit: int, max_len: int) -> list[str]:
    items = raw if isinstance(raw, list) else []
    return [_text(v)[:max_len] for v in items if isinstance(v, str) and _text(v)][:limit]


def _public_args(args: dict[str, Any]) -> dict[str, Any]:
    hidden = {"evaluation", "memory", "next_goal", "think", "message", "plan", "plan_step", "irreversible",
              "_thinking"}
    return {k: v for k, v in args.items() if k not in hidden}


def _action(name: str, args: dict[str, Any], target_label: str) -> dict[str, Any]:
    return {"name": name, "args": _public_args(args), "label": label(name, args, target_label)}


def _signature(name: str, args: dict[str, Any], url: str) -> str:
    return f"{url}|{name}|{json.dumps(_public_args(args), sort_keys=True)}"


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()[:2000]
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)[:2000] if isinstance(value, (dict, list)) else str(value)


def _verify_field(value: str, corpus: str) -> bool:
    """Layer 3 anti-hallucination check: verify an extracted field value appears in observed page text."""
    clean = value.strip().lower()
    if not clean or not corpus.strip():
        return True
    # Placeholders for a missing or blank value have nothing to match on the page.
    if clean in ("not_found", "not found", "n/a", "na", "none", "null", "unknown", "true", "false", "-", "—",
                 "empty", "(empty)", "blank", "(blank)", "(none)", "not collected"):
        return True
    if clean.startswith(("http://", "https://")):
        return True
    norm_corpus = re.sub(r"\s+", " ", corpus.lower())
    norm_val = re.sub(r"\s+", " ", clean)
    if norm_val in norm_corpus:
        return True
    tokens = [t for t in re.findall(r"[a-z0-9$€£¥.%+-]{2,}", norm_val)
              if t not in ("the", "and", "for", "with", "from", "per", "usd", "inc")]
    if not tokens:
        return True
    matched = sum(1 for t in tokens if t in norm_corpus)
    return (matched / len(tokens)) >= 0.6

