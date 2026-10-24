"""Run-local, metadata-only model accounting. Never participates in action decisions."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Coroutine
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from functools import wraps
from typing import Any, Literal, ParamSpec, TypeVar
from uuid import uuid4

Purpose = Literal["planner", "extract", "jev", "text"]
P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class CallContext:
    run_id: str
    purpose: Purpose
    step_id: str | None = None
    parent_step_id: str | None = None


@dataclass(frozen=True)
class CallRecord:
    run_id: str
    purpose: Purpose
    step_id: str | None
    parent_step_id: str | None
    logical_call_id: str
    attempt_id: str
    attempt: int
    provider: str
    requested_model: str
    resolved_model: str | None = None
    duration_ms: int = 0
    outcome: str = "success"
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost_usd: float | None = None
    cost_basis: str = "unknown"


@dataclass
class DelegationRecord:
    episode_id: str
    run_id: str
    parent_step_id: str
    started_at: float
    ended_at: float = 0
    reason: str = "model_error"
    completion_claimed: bool = False
    jev_attempts: int = 0
    text_attempts: int = 0
    executed_actions: int = 0
    skipped_actions: int = 0


@dataclass
class LogicalCall:
    ledger: RunLedger
    context: CallContext
    id: str
    attempts: int = 0
    last_id: str | None = None


_scope: ContextVar[tuple[Any, CallContext] | None] = ContextVar("accounting_scope", default=None)
_call: ContextVar[LogicalCall | None] = ContextVar("accounting_call", default=None)


def _number(value: Any, *, integer: bool = False) -> int | float | None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return None
    if integer and int(value) != value:
        return None
    return int(value) if integer else float(value)


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    unknown_cost = sum(r["cost_usd"] is None for r in records)
    unknown_usage = sum(r["input_tokens"] is None or r["output_tokens"] is None for r in records)
    known_cost = sum(r["cost_usd"] or 0 for r in records)
    return {
        "attempts": len(records), "successful_responses": sum(r["outcome"] == "success" for r in records),
        "input_tokens": sum(r["input_tokens"] or 0 for r in records),
        "output_tokens": sum(r["output_tokens"] or 0 for r in records),
        "cache_read_tokens": sum(r["cache_read_tokens"] or 0 for r in records),
        "cache_write_tokens": sum(r["cache_write_tokens"] or 0 for r in records),
        "duration_ms": sum(r["duration_ms"] for r in records),
        "known_cost_usd": known_cost,
        "total_cost_usd": None if unknown_cost else known_cost,
        "unknown_cost_count": unknown_cost, "unknown_usage_count": unknown_usage,
        "cost_basis": "unknown" if unknown_cost else
                      "estimate" if any(r["cost_basis"] == "estimate" for r in records) else "provider",
    }


class RunLedger:
    def __init__(self):
        self.run_id = uuid4().hex
        self._records: dict[str, CallRecord] = {}
        self._delegations: dict[str, dict[str, Any]] = {}
        self._fast_actions: set[str] = set()
        self._steps: dict[str, dict[str, Any]] = {}

    @contextmanager
    def scope(self, purpose: Purpose, step_id: str | None = None, parent_step_id: str | None = None):
        token = _scope.set((self, CallContext(self.run_id, purpose, step_id, parent_step_id)))
        call_token = _call.set(None)
        try:
            yield
        finally:
            _call.reset(call_token)
            _scope.reset(token)

    def record(self, record: CallRecord) -> None:
        if record.run_id != self.run_id:
            raise ValueError("A model attempt belongs to a different run")
        self._records.setdefault(record.attempt_id, record)

    def invalidate(self, attempt_id: str | None) -> None:
        record = self._records.get(attempt_id)
        if record and record.outcome == "success":
            self._records[attempt_id] = replace(record, outcome="invalid_response")

    def record_delegation(self, record: DelegationRecord) -> dict[str, Any]:
        record.ended_at = time.time()
        records = [r for r in self._records.values() if r.parent_step_id == record.parent_step_id]
        record.jev_attempts = sum(r.purpose == "jev" for r in records)
        record.text_attempts = sum(r.purpose == "text" for r in records)
        steps = [s for s in self._steps.values() if s["parent"] == record.parent_step_id]
        record.executed_actions = sum(s["status"] == "ok" for s in steps)
        record.skipped_actions = sum(s["status"] == "skipped" for s in steps)
        result = asdict(record)
        self._delegations[record.episode_id] = result
        return result

    def record_fast_action(self, step_id: str) -> None:
        self._fast_actions.add(step_id)

    def note_step(self, step: dict[str, Any]) -> None:
        if step.get("system") != "fast":
            return
        action = (step.get("action") or {}).get("name")
        if action not in ("click", "type_text", "select_option", "scroll"):
            return
        self._steps[step["id"]] = {"parent": step.get("parent"), "status": step["status"]}
        if step["status"] == "ok":
            self.record_fast_action(step["id"])

    def snapshot(self, *, details: bool = True) -> dict[str, Any]:
        records = [asdict(r) for r in self._records.values()]
        snapshot = {
            "schema_version": 1, "run_id": self.run_id, "summary": _summary(records),
            "by_purpose": {p: _summary([r for r in records if r["purpose"] == p])
                           for p in ("planner", "extract", "jev", "text")},
            "executed_fast_actions": len(self._fast_actions),
        }
        if details:
            snapshot.update(records=records, delegations=list(self._delegations.values()))
        return snapshot


def model_call(fn: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
    """Group transport retries, including nested native-client calls, as one logical call."""
    @wraps(fn)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        scope = _scope.get()
        if scope is None or _call.get() is not None:
            return await fn(*args, **kwargs)
        call = LogicalCall(scope[0], scope[1], uuid4().hex)
        token = _call.set(call)
        try:
            return await fn(*args, **kwargs)
        except Exception:
            call.ledger.invalidate(call.last_id)
            raise
        finally:
            _call.reset(token)
    return wrapped


def invalid_response() -> None:
    call = _call.get()
    if call:
        call.ledger.invalidate(call.last_id)


class ProviderAttempt:
    def __init__(self, provider: str, model: str):
        self.call = _call.get()
        self.provider, self.model = provider, model
        self.values: dict[str, Any] = {}
        self.outcome = "success"

    def capture(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        model = payload.get("model")
        if isinstance(model, str):
            self.values["resolved_model"] = model[:200]
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return
        native = self.provider == "anthropic"
        inp = _number(usage.get("input_tokens" if native or self.provider == "typesafe" else "prompt_tokens"),
                      integer=True)
        out = _number(usage.get("output_tokens" if native or self.provider == "typesafe" else "completion_tokens"),
                      integer=True)
        read = write = None
        if native:
            read = _number(usage.get("cache_read_input_tokens", 0), integer=True)
            write = _number(usage.get("cache_creation_input_tokens", 0), integer=True)
            from .anthropic_backend import PRICES, cost
            if inp is not None and out is not None and read is not None and write is not None:
                if self.model in PRICES:
                    self.values.update(cost_usd=cost(self.model, inp, out, write, read), cost_basis="estimate")
                inp += read + write
        else:
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                read = _number(details.get("cached_tokens"), integer=True)
            spent = _number(usage.get("cost")) if self.provider != "typesafe" else None
            if spent is not None:
                self.values.update(cost_usd=spent, cost_basis="provider")
        self.values.update(input_tokens=inp, output_tokens=out, cache_read_tokens=read, cache_write_tokens=write)

    def capture_native(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        self.capture({"model": getattr(response, "model", None), "usage": {
            k: getattr(usage, k, None) for k in
            ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        } | {k: getattr(usage, k, 0) or 0 for k in ("cache_read_input_tokens", "cache_creation_input_tokens")}})

    def capture_native_partial(self, response: Any) -> None:
        self.capture_native(response)
        # Stream-start output counts are provisional. Cancellation can still incur further billing.
        self.values.update(output_tokens=None, cost_usd=None, cost_basis="unknown")

    def __enter__(self):
        self.started = time.monotonic()
        if self.call:
            self.call.attempts += 1
        return self

    def __exit__(self, kind, exc, tb):
        if not self.call:
            return False
        if kind:
            self.outcome = "cancelled" if issubclass(kind, asyncio.CancelledError) else \
                "provider_error" if getattr(exc, "status_code", None) else "transport_error"
        record = CallRecord(**asdict(self.call.context), logical_call_id=self.call.id, attempt_id=uuid4().hex,
                            attempt=self.call.attempts, provider=self.provider, requested_model=self.model,
                            duration_ms=round((time.monotonic() - self.started) * 1000),
                            outcome=self.outcome, **self.values)
        self.call.ledger.record(record)
        self.call.last_id = record.attempt_id
        return False


async def tracked_post(http, url: str, *, provider: str, **kwargs):
    """Record physical HTTP attempts; validation remains in the existing callers."""
    with ProviderAttempt(provider, kwargs.get("json", {}).get("model", "")) as attempt:
        response = await http.post(url, **kwargs)
        try:
            attempt.capture(response.json())
        except ValueError:
            attempt.outcome = "invalid_response"
        if response.is_error:
            attempt.outcome = "provider_error"
        return response
