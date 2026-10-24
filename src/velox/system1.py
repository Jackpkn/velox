"""The System 1 loop: Jev executes a delegated UI subgoal with fast typed decisions, then hands back."""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from playwright.async_api import Error as PlaywrightError

from .browser import ActionError
from .fast import FastError
from .telemetry import DelegationRecord
from .tools import needs_approval

if TYPE_CHECKING:
    from .agent import Agent

FAST_MAX_ACTIONS = 15  # per subgoal; beyond this the planner looks at the page before Jev goes on
NO_CHANGE_LIMIT = 3  # identical observations in a row before Jev hands back
JEV_WAIT_SECONDS = 1
OPERATION_TOOLS = {"CLICK": "click", "TYPE_TEXT": "type_text", "SELECT": "select_option",
                   "SCROLL_DOWN": "scroll", "SCROLL_UP": "scroll", "WAIT": "wait"}


def pending_autocomplete(page: Any, autocomplete_id: str | None) -> dict[str, Any] | None:
    """The autocomplete field Jev typed into, while its suggestions are still open. Until one is chosen and the
    popup closes, the typed query is not a committed value, so nothing else may be touched."""
    if not autocomplete_id:
        return None
    field = next((e for e in page.elements.values() if e.get("autocomplete_id") == autocomplete_id), None)
    if field and (field.get("expanded") == "true" or field.get("suggestion_ids")):
        return field
    return None


def allowed_while_pending(d: Any, field: dict[str, Any]) -> bool:
    """With a suggestion pending, Jev may only work on that field, pick one of its suggestions, wait, scroll or
    give up: another control or a completion claim would leave the query uncommitted."""
    if d.operation in ("WAIT", "STUCK", "SCROLL_DOWN", "SCROLL_UP"):
        return True
    if d.operation == "TYPE_TEXT":
        return d.element_id == field["id"]
    if d.operation == "CLICK":
        return d.element_id == field["id"] or d.element_id in (field.get("suggestion_ids") or [])
    return False


class FastLoopMixin:
    """Mixed into Agent. Each Jev action is a visible, approvable, interruptible step of its own."""

    async def _delegate(self: Agent, subgoal: str, parent: dict[str, Any]) -> str:
        episode = DelegationRecord(uuid4().hex, self.ledger.run_id, parent["id"], time.time())
        self._delegation_record = episode
        try:
            return await self._delegate_loop(subgoal, parent)
        finally:
            if self.stopped:
                episode.reason = "stopped"
            elif self.paused:
                episode.reason = "paused"
            parent["delegation"] = self.ledger.record_delegation(episode)
            self._delegation_record = None

    def _interruption_reason(self: Agent) -> str:
        return "stopped" if self.stopped else "paused" if self.paused else "guidance_changed"

    async def _delegate_loop(self: Agent, subgoal: str, parent: dict[str, Any]) -> str:
        log: list[str] = []
        recent: list[str] = []
        unchanged = 0
        last_sig = None
        typed_autocomplete: str | None = None
        version = self.guidance_version
        for _ in range(FAST_MAX_ACTIONS):
            if self.stopped or self.paused or self.guidance_version != version:
                return self._handback(log, "the operator paused or stopped the run", self._interruption_reason())
            if self.actions_taken >= self.max_steps:
                return self._handback(log, "the action budget is used up", "action_budget")
            self.n += 1
            step: dict[str, Any] = {"id": f"s{self.n}", "n": self.n, "status": "thinking", "system": "fast",
                                    "parent": parent["id"], "started_at": time.time()}
            await self.set_state("thinking", "Jev is choosing the next action…")
            try:
                page = await self.browser.observe(screenshot=False)
            except (ActionError, PlaywrightError) as exc:
                if await self._discard_interrupted(step, version):
                    return self._handback(log, "the operator interrupted the decision", self._interruption_reason())
                await self._fail_step(step, str(exc).splitlines()[0][:200])
                return self._handback(log, "the page could not be read", "observation_error")
            if await self._discard_interrupted(step, version):
                return self._handback(log, "the operator interrupted the decision", self._interruption_reason())
            sig = hashlib.sha1(f"{page.url}\n{page.text}".encode()).hexdigest()
            unchanged = unchanged + 1 if sig == last_sig else 0
            last_sig = sig
            if unchanged >= NO_CHANGE_LIMIT:
                return self._handback(log, f"the last {NO_CHANGE_LIMIT} attempts did not change the page",
                                      "no_change")
            step["url"] = page.url
            await self.send_step(step)
            try:
                with self.ledger.scope("jev", step["id"], parent["id"]):
                    d = await self.fast.decide(page, goal=self.goal, subgoal=subgoal, recent=recent,
                                               guidance=self.guidance)
            except FastError as exc:
                if await self._discard_interrupted(step, version):
                    return self._handback(log, "the operator interrupted the decision", self._interruption_reason())
                await self._fail_step(step, str(exc))
                return self._handback(log, f"Jev failed: {exc}", "model_error")
            self.usage["fast_calls"] += 1
            self.usage["fast_ms_total"] += d.latency_ms
            step.update(confidence=round(d.confidence, 3), alternatives=d.alternatives, latency_ms=d.latency_ms,
                        thought={"evaluation": "", "memory": "", "next_goal": subgoal})
            if self.stopped or self.paused or self.guidance_version != version:
                why = "new guidance arrived" if self.guidance_version != version else "the run was paused"
                step.update(status="skipped", result=f"Discarded because {why}.", ended_at=time.time())
                await self.send_step(step)
                return self._handback(log, why, self._interruption_reason())
            pending = pending_autocomplete(page, typed_autocomplete)
            if pending is None:
                typed_autocomplete = None
            elif not allowed_while_pending(d, pending):
                return await self._fast_unconfirmed(step, pending, log)
            outcome = await self._fast_terminal(step, d, log)
            if outcome:
                return outcome
            if d.operation == "TYPE_TEXT":
                typed_autocomplete = page.elements.get(d.element_id, {}).get("autocomplete_id")
            result = await self._fast_act(step, d, page, subgoal, recent, version=version)
            if self.stopped or self.paused or self.guidance_version != version:
                return self._handback(log, "the operator interrupted the decision", self._interruption_reason())
            if step["status"] == "skipped":
                entry = f"Skipped: {step['action']['label']} → {result} Continue with the remaining subgoal."
                log.append(entry)
                recent.append(entry)
                continue
            if step["status"] == "error":
                return self._handback(log, f"an action failed: {step.get('error')}", "action_error")
            if step["status"] == "rejected":
                return self._handback(log, "you rejected an action", "rejected")
            entry = f"{step['action']['label']} → {result}"
            log.append(entry)
            recent.append(entry)
        return self._handback(log, f"Jev used its {FAST_MAX_ACTIONS}-attempt budget for this subgoal", "attempt_budget")

    async def _fast_unconfirmed(self: Agent, step: dict[str, Any], field: dict[str, Any], log: list[str]) -> str:
        label = str(field.get("label") or "this field")[:60]
        step.update(status="skipped", action={"name": "autocomplete_unconfirmed", "args": {},
                                              "label": "Autocomplete suggestion not chosen"},
                    result=f'"{label}" still shows open suggestions; the typed text is not a committed value.',
                    ended_at=time.time())
        await self.send_step(step)
        return self._handback(log, f'the "{label}" autocomplete suggestion was not chosen, so nothing else was '
                                   "touched", "autocomplete_unconfirmed")

    async def _fast_terminal(self: Agent, step: dict[str, Any], d: Any, log: list[str]) -> str | None:
        pct = f"{d.confidence:.0%}"
        if d.operation == "SUBGOAL_DONE":
            step.update(status="ok", action={"name": "subgoal_done", "args": {}, "label": "Subgoal looks complete"},
                        result=f"Jev reports the subgoal is done ({pct} confident).", ended_at=time.time())
            await self.send_step(step)
            return self._handback(log, f"Jev reports the subgoal is done ({pct} confident); verify on the page",
                                  "claimed_done")
        if d.operation == "STUCK":
            step.update(status="ok", action={"name": "stuck", "args": {}, "label": "Hand back to the planner"},
                        result="Jev can't progress here with the visible controls.", ended_at=time.time())
            await self.send_step(step)
            return self._handback(log, "Jev is stuck on this page", "stuck")
        if d.confidence < self.fast.config.min_confidence:
            step.update(status="skipped", action={"name": "unsure", "args": {}, "label": "Not confident enough"},
                        result=f"Jev was only {pct} sure; the planner will decide instead.", ended_at=time.time())
            await self.send_step(step)
            return self._handback(log, f"Jev was unsure ({pct}) about the next action", "low_confidence")
        return None

    async def _fast_act(self: Agent, step: dict[str, Any], d: Any, page: Any, subgoal: str,
                        recent: list[str], *, version: int) -> str:
        name = OPERATION_TOOLS[d.operation]
        args: dict[str, Any] = {"irreversible": d.irreversible >= 0.6}
        target_label = ""
        if d.element_id is not None:
            element = page.element(d.element_id)
            target_label = element["label"] or element["tag"]
            args["element_id"] = d.element_id
            try:
                rect = await self.browser.element_rect(d.element_id)
            except (ActionError, PlaywrightError):
                rect = None
            step["target"] = {"label": target_label, "rect": rect or element["rect"]}
        if name == "scroll":
            args["direction"] = "down" if d.operation == "SCROLL_DOWN" else "up"
        elif name == "wait":
            args["seconds"] = JEV_WAIT_SECONDS
        elif name == "select_option":
            args["option"] = d.option
        elif name == "type_text":
            await self.set_state("thinking", "Writing the text to type…")
            try:
                with self.ledger.scope("text", step["id"], step.get("parent")):
                    text, ms = await self.fast.write_text(goal=self.goal, subgoal=subgoal, page=page, recent=recent,
                                                          field={k: element.get(k) for k in ("label", "role", "value")})
            except FastError as exc:
                await self._fail_step(step, str(exc))
                return ""
            self.usage["text_calls"] += 1
            step["latency_ms"] = step.get("latency_ms", 0) + ms
            args.update(text=text, press_enter=False)
        from .agent import _action  # local import avoids a cycle
        if await self._discard_interrupted(step, version):
            return ""
        if name == "type_text" and text == str(element.get("value") or ""):
            step.update(action=_action(name, args, target_label), status="skipped", ended_at=time.time(),
                        result="That field already holds this value; nothing was typed.")
            await self.send_step(step)
            return step["result"]
        step.update(action=_action(name, args, target_label), status="planned")
        risky, reason = needs_approval(name, args, target_label, self.approval_mode)
        if risky:
            step["risk"] = {"risky": True, "reason": reason}
            if not await self._approve(step, reason):
                return ""
        await self._execute(step, name, args, version=version)
        return step.get("result", "")

    def _handback(self: Agent, log: list[str], reason: str, code: str) -> str:
        episode = getattr(self, "_delegation_record", None)
        if episode is not None:
            episode.reason = code
            episode.completion_claimed = code == "claimed_done"
        count = sum(not entry.startswith("Skipped:") for entry in log)
        done = f"Jev took {count} action{'s' if count != 1 else ''}"
        detail = (": " + "; ".join(log[-8:])) if log else ""
        return f"{done}{detail}. Returned control because {reason}."
