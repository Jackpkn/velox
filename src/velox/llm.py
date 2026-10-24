"""Planner (System 2) tool-calling client: Claude API directly, or any OpenAI-compatible endpoint (OpenRouter).

Credentials stay on the server.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

from .anthropic_backend import AnthropicBackend, AnthropicError, _recover_xml_params
from .anthropic_backend import cost as anthropic_cost
from .telemetry import invalid_response, model_call, tracked_post

ANTHROPIC_MODELS = "claude-haiku-5-5,claude-sonnet-5-5,claude-opus-5-5"  # first is the default planner
DEFAULT_MODELS = "anthropic/claude-sonnet-5.5,anthropic/claude-haiku-5.5,inception/mercury-2.5,google/gemini-3.7-flash"


class LLMError(RuntimeError):
    """The model could not produce a usable decision. No browser action was taken."""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    models: tuple[str, ...]
    provider: str = "openrouter"
    extract_model: str = ""  # small, fast model that turns page text into rows for the extract tool

    @property
    def default_model(self) -> str:
        return self.models[0]

    @classmethod
    def from_env(cls) -> LLMConfig:
        provider = os.environ.get("LLM_PROVIDER") or ("anthropic" if os.environ.get("ANTHROPIC_API_KEY")
                                                      else "openrouter")
        if provider == "anthropic":
            key, defaults = os.environ.get("ANTHROPIC_API_KEY", ""), ANTHROPIC_MODELS
        else:
            key = (os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
                   or os.environ.get("TEXT_MODEL_API_KEY") or "")
            defaults = DEFAULT_MODELS
        models = tuple(m.strip() for m in os.environ.get("LLM_MODELS", defaults).split(",") if m.strip())
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
            api_key=key,
            models=models or tuple(defaults.split(",")),
            provider=provider,
            extract_model=os.environ.get("EXTRACT_MODEL")
            or ("claude-haiku-5-5" if provider == "anthropic" else "anthropic/claude-haiku-5.5"),
        )


@dataclass(frozen=True)
class Decision:
    name: str
    args: dict[str, Any]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    thinking: str = ""
    cached_tokens: int = 0  # part of input_tokens served from the prompt cache (billed at a tenth)


class LLMClient:
    def __init__(self, config: LLMConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(90, connect=10), transport=transport)
        self.text_only: set[str] = set()  # models that rejected image input; screenshots are dropped for them
        self.anthropic = AnthropicBackend(config.api_key) if config.provider == "anthropic" and config.api_key \
            else None

    @model_call
    async def decide(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        *, force: str | None = None, on_retry: Any = None, on_delta: Any = None, hard: bool = True,
    ) -> Decision:
        """`hard` turns get native thinking on the Anthropic backend; the OpenAI-compatible path ignores it."""
        if not self.config.api_key:
            raise LLMError("No model API key is configured on the server (set ANTHROPIC_API_KEY or LLM_API_KEY).")
        if self.anthropic:
            try:
                name, args, tokens_in, tokens_out, spent, cached = await self.anthropic.decide(
                    model, messages, tools, force=force, on_retry=on_retry, on_delta=on_delta, hard=hard)
            except AnthropicError as exc:
                raise LLMError(str(exc)) from None
            native_thinking = str(args.pop("_thinking", "") or "").strip()
            return Decision(name=name, args=args, input_tokens=tokens_in, output_tokens=tokens_out,
                            cost_usd=spent, thinking=native_thinking, cached_tokens=cached)
        if model in self.text_only:
            messages = strip_images(messages)
        body = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": force}} if force else "required",
            "parallel_tool_calls": False,
            "temperature": 0.2,
            "max_tokens": 4096,
            "usage": {"include": True},
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}", "X-Title": "HyperAgent"}
        last = "unknown error"
        for attempt in range(4):
            try:
                response = await tracked_post(self.http, f"{self.config.base_url}/chat/completions",
                                              provider=self.config.provider, json=body, headers=headers)
            except httpx.HTTPError as exc:
                last = f"connection error ({type(exc).__name__})"
            else:
                if response.status_code < 400:
                    try:
                        return parse_decision(response.json())
                    except LLMError as exc:
                        invalid_response()
                        last = str(exc)
                        # A malformed tool call is retried with a corrective note; nothing was executed.
                        body["messages"] = [*messages, {"role": "user", "content": f"Your last reply was invalid: "
                                                        f"{exc}. Call exactly one tool with valid JSON arguments."}]
                elif response.status_code in (401, 403):
                    raise LLMError("The model provider rejected the API key.")
                elif response.status_code == 402:
                    raise LLMError("The model provider account is out of credit.")
                elif response.status_code in (400, 404) and "image" in response.text.lower() \
                        and model not in self.text_only:
                    # A text-only model: drop screenshots and retry at once, without backoff.
                    self.text_only.add(model)
                    messages = strip_images(messages)
                    body["messages"] = messages
                    continue
                else:
                    last = f"HTTP {response.status_code}: {response.text[:200]}"
                    if response.status_code == 400 and "tool" not in response.text.lower():
                        raise LLMError(f"The model provider refused the request ({last}).")
            if attempt < 3:
                if on_retry:
                    await on_retry(f"Model call failed ({last}); retrying ({attempt + 1}/3)…")
                await asyncio.sleep(1.5 * 2**attempt)
        raise LLMError(f"The model did not respond after 4 attempts: {last}")

    @model_call
    async def complete_json(self, system: str, user: str, *, max_tokens: int = 4000) -> tuple[dict[str, Any], int,
                                                                                         int, float]:
        """One JSON answer from the extraction model: (object, input tokens, output tokens, cost in USD)."""
        model = self.config.extract_model
        if not self.config.api_key or not model:
            raise LLMError("No extraction model is configured on the server.")
        if self.anthropic:
            try:
                output, tokens_in, tokens_out = await self.anthropic.complete_json(model, system, user,
                                                                                   max_tokens=max_tokens)
            except AnthropicError as exc:
                raise LLMError(str(exc)) from None
            return output, tokens_in, tokens_out, anthropic_cost(model, tokens_in, tokens_out)
        body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "usage": {"include": True},
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        headers = {"Authorization": f"Bearer {self.config.api_key}", "X-Title": "HyperAgent"}
        try:
            response = await tracked_post(self.http, f"{self.config.base_url}/chat/completions",
                                              provider=self.config.provider, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"Could not reach the extraction model ({type(exc).__name__}).") from None
        if response.status_code >= 400:
            raise LLMError(f"The extraction model failed (HTTP {response.status_code}).")
        try:
            payload = response.json()
            raw = payload["choices"][0]["message"]["content"]
            output = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        except (KeyError, IndexError, TypeError, ValueError):
            raise LLMError("The extraction model did not return JSON.") from None
        usage = payload.get("usage") or {}
        return (output, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0),
                float(usage.get("cost") or 0.0))

    async def close(self) -> None:
        await self.http.aclose()
        if self.anthropic:
            await self.anthropic.close()


def strip_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = [part for part in content if part.get("type") != "image_url"]
        stripped.append({**m, "content": content})
    return stripped


def _extract_message_thinking(message: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("thinking", "reasoning", "reasoning_content"):
        val = message.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    details = message.get("reasoning_details")
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict):
                txt = item.get("text") or item.get("summary")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt.strip())
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        parts.append(content.strip())
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                txt = block.get("thinking") or block.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt.strip())
    return "\n\n".join(dict.fromkeys(parts))


def parse_decision(payload: dict[str, Any]) -> Decision:
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        raise LLMError(f"Empty model response {str(error)[:200]}") from None
    calls = message.get("tool_calls") or []
    if not calls:
        raise LLMError("The model replied without choosing an action")
    fn = calls[0].get("function", {})
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        raise LLMError("The action arguments were not valid JSON") from None
    if not isinstance(args, dict):
        raise LLMError("The action arguments were not an object")
    args = _recover_xml_params(args)
    usage = payload.get("usage") or {}
    return Decision(
        name=fn.get("name", ""),
        args=args,
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        cost_usd=float(usage.get("cost") or 0.0),
        thinking=_extract_message_thinking(message),
    )
