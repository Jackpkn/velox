# Velox (`velox`)

Standalone, drop-in browser engine and dual-system autonomous browser agent with:
- **Persistent Chrome Profile Sessions** (`launch_persistent_context` saving Cookies, `localStorage`, `IndexedDB`, Service Workers, and device trust tokens in `.velox/browser-profile`)
- **25 FPS Hardware CDP Screencasting** (`Page.startScreencast` + cached tab/title metadata)
- **CDP Headless Stealth (`_apply_stealth`)** (runs invisibly in `HEADLESS=1` using your real Chrome binary while stripping `HeadlessChrome` from `User-Agent` and `Sec-CH-UA` so Google, Faire.com, and Shopify logins work seamlessly)
- **Vanilla HTML + CSS + JS Testbench** (`web/` + `velox.web_server` with interactive viewport click, keyboard typing, paste, and scroll forwarding)

---

## Quick Start (Web Testbench)

```bash
cp .env.example .env
# Add your LLM_API_KEY in .env
uv sync
uv run --env-file .env python -m velox.web_server
```

Open **http://127.0.0.1:8788** in your browser.

---

## Integration Modes

### Case 1: Browser Controller Only (`BrowserSession` — 3 Files)
If your project already has its own LLM/agent loop and only needs reading-order DOM indexing (`[id]<tag>label</tag>`), ARIA combobox/autocomplete synchronization, `MutationObserver` DOM settle, and parallel tab reading, copy only:
1. `src/velox/browser.py`
2. `src/velox/dom.js`
3. `src/velox/reader.js`

```python
from pathlib import Path
from velox.browser import BrowserSession

browser = BrowserSession()
# Persistent profile saves Cookies, localStorage, IndexedDB, and Cloudflare device tokens
await browser.start("https://www.faire.com", user_data_dir=Path(".velox/browser-profile"))
obs = await browser.observe()
print(obs.text)
```

### Case 2: Full Goal-Driven Autonomous Agent (`Agent` + `BrowserSession` — 13 Files)
Use the complete `src/velox/` package:
```bash
uv run --env-file .env velox --url "https://www.faire.com" "Check my Faire account or pause so I can sign in once"
```
Every subsequent run automatically reuses `.velox/browser-profile` (`launch_persistent_context`), keeping you logged in across runs without re-prompting.
