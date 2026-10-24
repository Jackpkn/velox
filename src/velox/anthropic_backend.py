"""Direct Anthropic (Claude API) backend for the planner and the text helper, via the official SDK."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import anthropic

from .telemetry import ProviderAttempt, invalid_response, model_call

# USD per million tokens (input, output), Claude API first-party rates.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

_XML_PARAM_RE = re.compile(
    r'<parameter\s+name=["\']([^"\']+)["\']\s*>([\s\S]*?)(?:</parameter>|(?=<parameter\b)|$)',
    re.I,
)
_XML_TAG_SPLIT_RE = re.compile(r"</(?:evaluation|memory|think|next_goal|message|plan)>|<parameter\b", re.I)
_INT_FIELDS = {"element_id", "tab_index", "plan_step"}
_NUM_FIELDS = {"pages", "seconds"}
_BOOL_FIELDS = {"press_enter", "irreversible", "success"}
_JSON_FIELDS = {"plan", "items", "rows", "columns", "sources", "issues", "urls", "checklist", "assumptions",
                "follow_ups", "fields", "remember"}


def _recover_xml_params(inputs: dict[str, Any]) -> dict[str, Any]:
    """Recover parameters if the model leaked <parameter name='...'> XML tags into a string argument."""
    out = dict(inputs)
    for key, val in list(out.items()):
        if not isinstance(val, str):
            continue
        tracked_tags = ("evaluation", "memory", "think", "next_goal", "message")
        if "<parameter" in val or any(f"</{t}>" in val for t in tracked_tags):
            for match in _XML_PARAM_RE.finditer(val):
                pname = match.group(1).strip()
                raw_pval = re.sub(r"</[a-zA-Z0-9_]+>\s*$", "", match.group(2).strip()).strip()
                if not raw_pval or out.get(pname) not in (None, ""):
                    continue
                if pname in _INT_FIELDS:
                    m = re.search(r"-?\d+", raw_pval)
                    if m:
                        out[pname] = int(m.group(0))
                elif pname in _NUM_FIELDS:
                    m = re.search(r"-?\d+(?:\.\d+)?", raw_pval)
                    if m:
                        out[pname] = float(m.group(0))
                elif pname in _BOOL_FIELDS:
                    out[pname] = raw_pval.lower() in ("true", "1", "yes")
                elif pname in _JSON_FIELDS:
                    try:
                        out[pname] = json.loads(raw_pval)
                    except ValueError:
                        pass
                else:
                    out[pname] = raw_pval
            out[key] = _XML_TAG_SPLIT_RE.split(val)[0].strip()
    for int_key in _INT_FIELDS:
        raw_int = out.get(int_key)
        if isinstance(raw_int, str):
            m = re.search(r"-?\d+", raw_int)
            if m:
                out[int_key] = int(m.group(0))
    return out


def extract_partial_thought(partial_json: str) -> dict[str, str]:
    """Extract in-flight evaluation / next_goal / think / message strings from a streaming partial JSON buffer."""
    out: dict[str, str] = {}
    for field in ("evaluation", "next_goal", "memory", "think", "message"):
        m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)', partial_json)
        if m:
            raw = m.group(1).replace('\\"', '"').replace("\\n", " ").replace("\\\\", "\\")
            cleaned = _XML_TAG_SPLIT_RE.split(raw)[0].strip()
            if cleaned:
                out[field] = cleaned
    return out


class AnthropicError(RuntimeError):
    """Raised with an operator-safe message; nothing was executed."""


def convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """OpenAI-style chat messages (as the agent builds them) → Anthropic system + messages."""
    system_parts, out = [], []
    for m in messages:
        content = m.get("content")
        if m["role"] == "system":
            system_parts.append(content if isinstance(content, str) else json.dumps(content))
            continue
        if isinstance(content, str):
            blocks: list[dict[str, Any]] = [{"type": "text", "text": content}]
        else:
            blocks = []
            for part in content or []:
                if part.get("type") == "text":
                    blocks.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    header, _, data = url.partition(",")
                    media = header.removeprefix("data:").split(";")[0] or "image/jpeg"
                    blocks.insert(0, {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": m["role"], "content": blocks})
    return "\n\n".join(system_parts), out


def convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"name": t["function"]["name"], "description": t["function"]["description"],
             "input_schema": t["function"]["parameters"]} for t in tools]


def speed_settings(model: str, *, for_planner: bool = True, hard: bool = True) -> dict[str, Any]:
    """Native thinking only where it pays. PLANNER_THINKING=auto (default) thinks on hard planner turns only
    (first plan, recovery, final comparison); 1 thinks on every turn; 0 never does."""
    mode = os.environ.get("PLANNER_THINKING", "auto")
    if not for_planner or mode == "0" or (mode != "1" and not hard):
        if model.startswith("claude-haiku"):
            return {"thinking": {"type": "disabled"}}
        return {"output_config": {"effort": "low"}}
    effort = os.environ.get("PLANNER_EFFORT", "medium")
    return {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": effort},
    }


def cost(model: str, input_tokens: int, output_tokens: int, cache_write: int = 0, cache_read: int = 0) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    # Prompt-cache writes cost 1.25x the input rate and reads 0.1x.
    billed_in = input_tokens + cache_write * 1.25 + cache_read * 0.1
    return (billed_in * price_in + output_tokens * price_out) / 1_000_000


def _cached_system(system: str) -> list[dict[str, Any]] | str:
    """The system prompt and tool list are identical every step, so they are cached; the page state is not."""
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}] if system else system


# Thinking counts toward the output limit, and a final reply can carry a whole results table, so 4k was cut off in
# practice. Only generated tokens are billed. Kept under the SDK's ~21k ceiling for non-streamed requests.
PLANNER_MAX_TOKENS = 16000


class AnthropicBackend:
    def __init__(self, api_key: str, client: Any = None):
        # The agent reports each retry to the operator, so the SDK's silent retries are turned off.
        self.client = client or anthropic.AsyncAnthropic(api_key=api_key, max_retries=0, timeout=90)

    @model_call
    async def decide(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *,
        force: str | None = None, on_retry: Any = None, on_delta: Any = None, hard: bool = True,
    ) -> tuple[str, dict[str, Any], int, int, float, int]:
        """(tool name, arguments, input tokens incl. cache, output tokens, cost, of which read from cache)."""
        system, converted = convert_messages(messages)
        if force:
            # Forced tool_choice is rejected on current models; instruct instead and verify the result.
            converted[-1]["content"].append({"type": "text", "text": f"Call the `{force}` tool now."})
        last = "unknown error"
        wait = 0.0
        for attempt in range(5):
            try:
                with ProviderAttempt("anthropic", model) as measured:
                    req_kwargs = dict(
                        model=model, max_tokens=PLANNER_MAX_TOKENS, system=_cached_system(system), messages=converted,
                        tools=convert_tools(tools),
                        tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                        **speed_settings(model, for_planner=True, hard=hard),
                    )
                    if on_delta is not None and hasattr(self.client.messages, "stream"):
                        thinking_buf = ""
                        text_buf = ""
                        json_buf = ""
                        async with self.client.messages.stream(**req_kwargs) as stream:
                            async for event in stream:
                                if getattr(event, "type", None) == "message_start":
                                    measured.capture_native_partial(getattr(event, "message", None))
                                if getattr(event, "type", None) == "content_block_delta":
                                    delta = getattr(event, "delta", None)
                                    dtype = getattr(delta, "type", None)
                                    if dtype == "thinking_delta":
                                        thinking_buf += getattr(delta, "thinking", "") or ""
                                    elif dtype == "text_delta":
                                        text_buf += getattr(delta, "text", "") or ""
                                    elif dtype == "input_json_delta":
                                        json_buf += getattr(delta, "partial_json", "") or ""
                                    else:
                                        continue
                                    partial_thought = extract_partial_thought(json_buf)
                                    combined_thinking = "\n\n".join(
                                        p.strip() for p in (thinking_buf, text_buf) if p and p.strip()
                                    )
                                    if combined_thinking:
                                        partial_thought["thinking"] = combined_thinking
                                    if partial_thought:
                                        await on_delta(partial_thought)
                            response = await stream.get_final_message()
                    else:
                        response = await self.client.messages.create(**req_kwargs)
                    measured.capture_native(response)
            except anthropic.AuthenticationError:
                raise AnthropicError("Anthropic rejected the API key.") from None
            except anthropic.PermissionDeniedError:
                raise AnthropicError("This Anthropic API key may not use that model.") from None
            except anthropic.NotFoundError:
                raise AnthropicError(f"Anthropic does not know the model {model!r}.") from None
            except anthropic.BadRequestError as exc:
                message = str(getattr(exc, "message", exc))
                if "credit" in message.lower():
                    raise AnthropicError("The Anthropic account is out of credit.") from None
                raise AnthropicError(f"Anthropic refused the request: {message[:200]}") from None
            except anthropic.RateLimitError as exc:
                last = "rate limited"
                retry_after = exc.response.headers.get("retry-after") if exc.response is not None else None
                wait = min(float(retry_after), 30.0) if retry_after and retry_after.replace(".", "").isdigit() else 0
            except anthropic.APIStatusError as exc:
                last = f"HTTP {exc.status_code}"
            except anthropic.APIConnectionError:
                last = "connection error"
            else:
                if response.stop_reason == "refusal":
                    raise AnthropicError("Claude declined this step for safety reasons.")
                if response.stop_reason == "max_tokens":
                    # Cut off mid-reply: the tool call may be partial (half a results table, a missing element id),
                    # so it is never acted on. Ask again, nudging toward a shorter reply.
                    invalid_response()
                    last = "the reply was cut off at the output limit"
                    note = {"type": "text", "text": "Your previous reply hit the output limit before its action was "
                                                    "complete. Keep any thinking short and call exactly one tool now."}
                    converted = [*converted[:-1], {**converted[-1], "content": [*converted[-1]["content"], note]}]
                    continue
                usage = response.usage
                cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
                cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
                spent = cost(model, usage.input_tokens, usage.output_tokens, cache_write, cache_read)
                tokens_in = usage.input_tokens + cache_write + cache_read
                call = next((b for b in response.content if b.type == "tool_use"), None)
                if call is not None and (not force or call.name == force):
                    thinking_parts = [
                        getattr(b, "thinking", "") or ""
                        for b in response.content
                        if getattr(b, "type", None) == "thinking"
                    ]
                    text_parts = [
                        getattr(b, "text", "") or ""
                        for b in response.content
                        if getattr(b, "type", None) == "text"
                    ]
                    native_thinking = "\n\n".join(
                        p.strip() for p in (*thinking_parts, *text_parts) if p and p.strip()
                    )
                    inputs = _recover_xml_params(dict(call.input))
                    if native_thinking:
                        inputs["_thinking"] = native_thinking
                    return call.name, inputs, tokens_in, usage.output_tokens, spent, cache_read
                invalid_response()
                last = "the reply did not choose an action"
                converted = [*converted,
                             {"role": "assistant", "content": [b.model_dump(exclude_none=True)
                                                               for b in response.content if b.type == "text"]
                              or [{"type": "text", "text": "(no action)"}]},
                             {"role": "user", "content": [{"type": "text", "text": "Call exactly one tool"
                                                           + (f" (`{force}`)" if force else "") + " now."}]}]
                continue
            if attempt < 4:
                delay = max(wait, 2.0 * 2**attempt)
                wait = 0.0
                if on_retry:
                    await on_retry(f"Model call failed ({last}); retrying in {delay:.0f} s ({attempt + 1}/4)…")
                await asyncio.sleep(delay)
        raise AnthropicError(f"Claude did not respond after 5 attempts: {last}")

    @model_call
    async def complete_json(self, model: str, system: str, user: str, *,
                            max_tokens: int = 400) -> tuple[dict[str, Any], int, int]:
        """One JSON answer (the text helper and the extract tool)."""
        try:
            with ProviderAttempt("anthropic", model) as measured:
                response = await self.client.messages.create(
                    model=model, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}],
                    **speed_settings(model, for_planner=False),
                )
                measured.capture_native(response)
        except anthropic.APIError as exc:
            raise AnthropicError(f"The text helper failed: {type(exc).__name__}") from None
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            return json.loads(text[text.find("{"): text.rfind("}") + 1]), response.usage.input_tokens, \
                response.usage.output_tokens
        except ValueError:
            raise AnthropicError("The text helper did not return JSON.") from None

    async def close(self) -> None:
        await self.client.close()
