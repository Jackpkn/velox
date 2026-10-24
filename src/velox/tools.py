"""Tool schemas the model may call, human-readable labels, and the risk policy for approvals."""

from __future__ import annotations

import re
from typing import Any

REASONING = {
    "evaluation": {
        "type": "string",
        "description": "One sentence: did the previous action or Jev subgoal actually work? Cite what you see in "
                       "<browser_state> now.",
    },
    "memory": {
        "type": "string",
        "description": "Brief running notes: what is done, what was found, what remains. Keep under 60 words.",
    },
    "think": {
        "type": "string",
        "description": "Optional deep reasoning scratchpad. OMIT on routine steps. Use ONLY when recovering from "
                       "a stuck action, comparing multiple candidates/prices, or verifying constraints before done.",
    },
    "message": {
        "type": "string",
        "description": "One or two short first-person sentences for the person watching live: what you just learned "
                       "and what you will do next. Plain words, no tool or field names.",
    },
    "plan_step": {
        "type": "integer",
        "description": "1-based number of the plan item THIS action works on. Earlier items count as done.",
    },
    "plan": {
        "type": "array",
        "description": "The full updated plan (3-7 short items). Required on step 1; afterwards ONLY when the plan "
                       "changes. Progress comes from plan_step.",
        "items": {"type": "string"},
    },
}

# Only 3 short strings + plan_step required on every turn; 'think' and 'plan' are conditional.
REASONING_REQUIRED = ["evaluation", "memory", "message", "plan_step"]

RISK_FIELDS = {
    "irreversible": {
        "type": "boolean",
        "description": "True if this action (or delegated subgoal) submits, purchases, sends, books, deletes, "
                       "publishes, or otherwise commits something on the user's behalf that is hard to undo.",
    },
}


def _bare(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip descriptions from shared fields since they are explained once in the system prompt."""
    out = {k: v for k, v in schema.items() if k != "description"}
    if isinstance(out.get("items"), dict):
        out["items"] = _bare(out["items"])
    if isinstance(out.get("properties"), dict):
        out["properties"] = {k: _bare(v) for k, v in out["properties"].items()}
    return out


_SHARED = {k: _bare(v) for k, v in REASONING.items()}


def _tool(name: str, description: str, params: dict[str, Any], required: list[str], risk: bool = False) -> dict:
    # Schema order matters for autoregressive decoding:
    # 1. Short reasoning & optional 'think' -> 2. Action arguments -> 3. Risk flag -> 4. Optional verbose 'plan' last
    reasoning_head = {k: v for k, v in _SHARED.items() if k != "plan"}
    properties = {
        **reasoning_head,
        **params,
        **(RISK_FIELDS if risk else {}),
        "plan": _SHARED["plan"],
    }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": REASONING_REQUIRED + required + (["irreversible"] if risk else []),
            },
        },
    }


ELEMENT = {"element_id": {"type": "integer", "description": "The [id] of an element in the CURRENT <browser_state>."}}
ROW = {"type": "object", "additionalProperties": {"type": "string"}}

TOOLS = [
    _tool("search", "Web search for discovery; leaves the current page. Not for a site's own search box.",
          {"query": {"type": "string"}}, ["query"]),
    _tool("navigate", "Open a named site or an exact URL read from the page.",
          {"url": {"type": "string"}}, ["url"]),
    _tool("click", "Click an element [id]. With delegate available, prefer it for a group of related actions; "
          "click directly for a single button, a confirmation, or to finish a widget after Jev hands back.",
          ELEMENT, ["element_id"], risk=True),
    _tool("type_text", "Replace the text of an input field [id], optionally pressing Enter afterwards. "
          "Use directly for a single field, a password, or to finish after Jev hands back.",
          {**ELEMENT, "text": {"type": "string"}, "press_enter": {"type": "boolean"}},
          ["element_id", "text", "press_enter"], risk=True),
    _tool("select_option", "Choose an option of a <select> dropdown by its visible label.",
          {**ELEMENT, "option": {"type": "string"}}, ["element_id", "option"], risk=True),
    _tool("press_key", "Press a keyboard key such as Enter, Escape, Tab, ArrowDown.",
          {"key": {"type": "string"}}, ["key"], risk=True),
    _tool("scroll", "Scroll the page to reveal more content.",
          {"direction": {"type": "string", "enum": ["down", "up"]},
           "pages": {"type": "number", "description": "Screen heights to scroll, default 1."}}, ["direction"]),
    _tool("go_back", "Go back to the previous page in this tab.", {}, []),
    _tool("switch_tab", "Switch to another open tab.", {"tab_index": {"type": "integer"}}, ["tab_index"]),
    _tool("close_tab", "Close the current tab.", {}, []),
    _tool("read_page", "Full page text with links, shown once in the next step only.", {}, []),
    _tool("read_urls_parallel", "Read 2-4 URLs in background tabs once; for data use extract(urls).",
          {"urls": {"type": "array", "maxItems": 4, "items": {"type": "string"}}},
          ["urls"]),
    _tool("extract", "Save structured rows from this page, or 1-8 URLs/hrefs in parallel, as findings in one step.",
          {"fields": {"type": "array", "maxItems": 12, "items": {"type": "string"}, "description": "Column names."},
           "instruction": {"type": "string", "description": "Which items to keep; empty keeps all."},
           "urls": {"type": "array", "maxItems": 8, "items": {"type": "string"},
                    "description": "Empty means the current page."}},
          ["fields"]),
    _tool("record_findings", "Save facts verified on this page as flat rows; items=[] marks it checked.",
          {"items": {"type": "array", "items": ROW}, "source_url": {"type": "string"}},
          ["items", "source_url"]),
    _tool("wait", "Wait 1-10 s for the page to load.", {"seconds": {"type": "number"}}, []),
    _tool("ask_user", "Ask the operator for details, a decision, or a takeover; the task pauses.",
          {"question": {"type": "string"}}, ["question"]),
    _tool("done", "Finish with the verified result (see Finishing).",
          {"success": {"type": "boolean"},
           "summary": {"type": "string"},
           "columns": {"type": "array", "items": {"type": "string"}},
           "rows": {"type": "array", "items": ROW},
           "sources": {"type": "array", "items": {"type": "object", "properties": {
               "title": {"type": "string"}, "url": {"type": "string"}}, "required": ["title", "url"]}},
           "issues": {"type": "array", "items": {"type": "string"}},
           "checklist": {"type": "array", "items": {"type": "object", "properties": {
               "requirement": {"type": "string"}, "met": {"type": "boolean"}, "evidence": {"type": "string"},
               "step": {"type": "integer"}}, "required": ["requirement", "met", "evidence"]}},
           "assumptions": {"type": "array", "items": {"type": "string"}},
           "follow_ups": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
           "remember": {"type": "array", "maxItems": 3, "items": {"type": "string"},
                        "description": "Lasting facts the user stated about themselves; usually []."}},
          ["success", "summary", "columns", "rows", "sources", "issues", "checklist"]),
]

DELEGATE = _tool(
    "delegate",
    "Hand one group of related on-page actions to Jev, the fast UI executor: fields to fill, suggestions to "
    "choose, dates to pick, filters to set, popups to dismiss. Jev cannot invent text, open URLs, read data or "
    "type passwords.",
    {"subgoal": {"type": "string", "description": "The outcome, every exact string to type or select in quotes, "
                 "what is already correct and must stay, and the visible stopping condition."}},
    ["subgoal"],
    risk=True,  # Jev can click buttons that submit/commit; the planner flags irreversible=True
)

TOOLS_WITH_FAST = [*TOOLS[:2], DELEGATE, *TOOLS[2:]]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS_WITH_FAST}
_REQUIRED = {t["function"]["name"]: t["function"]["parameters"]["required"] for t in TOOLS_WITH_FAST}

_REASONING_KEYS = set(REASONING) | set(RISK_FIELDS) | {"next_goal"}
_LENIENT_KEYS = {"checklist", "columns", "rows"}
# Fields where an empty string "" is a valid deliberate argument (e.g. clearing an input or unfiltered extract)
_ALLOW_EMPTY_STR = {"text", "instruction"}


def missing_args(name: str, args: dict[str, Any]) -> list[str]:
    """Required action parameters the model left out or left empty (reasoning fields are optional here)."""
    required = [k for k in _REQUIRED.get(name, []) if k not in _REASONING_KEYS | _LENIENT_KEYS]
    return [
        k for k in required
        if args.get(k) is None or (args.get(k) == "" and k not in _ALLOW_EMPTY_STR)
    ]


# Actions that mutate browser state; includes delegate since Jev executes clicks/typing on the page.
MUTATING = {
    "search", "navigate", "click", "type_text", "select_option",
    "press_key", "go_back", "close_tab", "delegate",
}

# Word-bounded so "Order Status", "Sort by Order", "Market Place" or "Apply Filters" do not ask for approval.
RISKY_WORDS = re.compile(
    r"\b(submit|buy|purchase|pay\b|place (?:an? |the |my |your )?order|checkout|check out|confirm|"
    r"delete|remove|send|book\b|reserve|subscribe|sign ?up|register|publish|post\b|transfer|"
    r"apply now|submit application|donate|unsubscribe|cancel (?:my )?(?:account|order|booking))\b",
    re.I,
)


def needs_approval(name: str, args: dict[str, Any], target_label: str, mode: str) -> tuple[bool, str]:
    """Return (needs approval, plain-language reason)."""
    if mode == "never" or name not in MUTATING:
        return False, ""
    if mode == "always":
        return True, "You asked to approve every action."
    if args.get("irreversible"):
        return True, "The agent flagged this as hard to undo."
    if name == "delegate" and RISKY_WORDS.search(args.get("subgoal", "")):
        return True, "This delegated UI flow includes a step that may commit or submit something."
    if name == "click" and RISKY_WORDS.search(target_label or ""):
        return True, f'The button "{target_label[:60]}" looks like it commits something.'
    if name == "type_text" and args.get("press_enter") and RISKY_WORDS.search(target_label or ""):
        return True, "Pressing Enter here may submit a form."
    if name == "press_key" and str(args.get("key", "")).lower() == "enter" and RISKY_WORDS.search(target_label or ""):
        return True, "Pressing Enter here may submit a form."
    return False, ""


def label(name: str, args: dict[str, Any], target_label: str = "") -> str:
    """A short human sentence for the timeline."""
    el_id = args.get("element_id", "?")
    target = f'"{target_label[:60]}"' if target_label else f"element [{el_id}]"
    match name:
        case "search":
            return f'Search the web for "{args.get("query", "")}"'
        case "navigate":
            return f"Open {args.get('url', '')}"
        case "click":
            return f"Click {target}"
        case "type_text":
            text = str(args.get("text", ""))
            shown = text if len(text) <= 60 else text[:57] + "…"
            return f'Type "{shown}" into {target}' + (" and press Enter" if args.get("press_enter") else "")
        case "select_option":
            return f'Choose "{args.get("option", "")}" in {target}'
        case "press_key":
            return f"Press {args.get('key', '')}"
        case "scroll":
            return f"Scroll {args.get('direction', 'down')}"
        case "go_back":
            return "Go back"
        case "switch_tab":
            return f"Switch to tab {args.get('tab_index')}"
        case "close_tab":
            return "Close this tab"
        case "read_page":
            return "Read the whole page"
        case "read_urls_parallel":
            urls = args.get("urls") if isinstance(args.get("urls"), list) else []
            return f"Read {len(urls)} page{'s' if len(urls) != 1 else ''} in parallel"
        case "extract":
            fields = [str(f) for f in (args.get("fields") or [])][:4]
            shown = ", ".join(fields) + ("…" if len(args.get("fields") or []) > 4 else "")
            pages = len(args.get("urls") or [])
            where = f"{pages} page{'s' if pages != 1 else ''}" if pages else "this page"
            return f"Extract {shown} from {where}"
        case "record_findings":
            n = len(args.get("items") or [])
            return f"Save {n} finding{'s' if n != 1 else ''}"
        case "wait":
            return "Wait for the page"
        case "ask_user":
            return "Ask you a question"
        case "done":
            return "Deliver the result"
        case "delegate":
            return f"Hand off to Jev: {str(args.get('subgoal', ''))[:90]}"
    return name