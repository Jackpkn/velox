# HyperAgent Extraction, Persistent Session & Smooth Viewport Blueprint

> **Purpose of this document**: Reference blueprint and living context for extracting HyperAgent's browser engine into a standalone/integrated project, upgrading session persistence (`launch_persistent_context` for "log in once, fetch anytime"), and upgrading the live browser stream (`CDP Page.startScreencast` + ref-based rendering for zero-glitch 15–25 FPS streaming).

---

## 0. Mandatory Working Rules

1. **Never manually type dependency versions into `pyproject.toml` or `package.json`**:
   - Always install dependencies via CLI commands (`uv add ...`, `npm install ...`) so the package manager resolves the latest compatible versions instead of guessing outdated versions.
2. **Keep the browser engine modular**:
   - `browser.py`, `dom.js`, and `reader.js` must remain decoupled from any specific web framework so they can be dropped into any project as a folder.
3. **Preserve safety & verification invariants**:
   - Element interactions use integer `[id]`s from the latest observation (`window.__ha.nodes`), never raw model-invented CSS/XPath selectors.
   - Browser mutations (`click`, `type_text`, `select_option`) are never auto-retried blindly.

---

## 1. Modular Extraction Modes (What to Copy Into a New Project)

### Case 1: Browser Controller Only (`BrowserSession` — 3 Files)
Use this when the target project already has its own LLM/agent orchestration and only needs high-speed DOM indexing, ARIA combobox/autocomplete synchronization, smart scrolling, and parallel tab reading.

**Required Files (0 internal imports outside these 3 files):**
1. [`hyperagent/browser.py`](../hyperagent/browser.py) — Playwright session manager, ARIA combobox/autocomplete handler (`AUTOCOMPLETE_STATE_JS` / `SUGGESTIONS_JS`), `MutationObserver` DOM settle (`SETTLE_JS`), native date/time normalization, parallel tab reader (`read_urls_parallel`), and SSRF network guard.
2. [`hyperagent/dom.js`](../hyperagent/dom.js) — Reading-order DOM walker that serializes visible text and interactive controls inline as `[id]<tag>label</tag>`, walks open `#shadow-root` trees, links `aria-controls`/`aria-owns` suggestions (`field.suggestion_ids`), and stores live DOM references in `window.__ha.nodes`.
3. [`hyperagent/reader.js`](../hyperagent/reader.js) — Full-page content & link extractor for `read_page()` and `read_urls_parallel()` (strips navbars, footers, cookie banners, and tracking noise).

**Install Command (via CLI only):**
```bash
uv init <project-name>
cd <project-name>
uv add playwright
uv run playwright install chromium
```

---

### Case 2: Full Autonomous Browser Agent (`Agent` + `BrowserSession` — 13 Files)
Use this when you want to pass a natural-language `goal` and run `await agent.run()` end-to-end (System 2 Planner + optional System 1 TypeSafe Jev fast loop + structured extraction).

**Required Files inside `hyperagent/`:**
- [`hyperagent/__init__.py`](../hyperagent/__init__.py) — Exports `Agent` and `BrowserSession`
- [`hyperagent/browser.py`](../hyperagent/browser.py), [`hyperagent/dom.js`](../hyperagent/dom.js), [`hyperagent/reader.js`](../hyperagent/reader.js) — Core browser engine
- [`hyperagent/agent.py`](../hyperagent/agent.py) — Main `observe → decide → (approve) → execute` loop, bot-wall detector (`_page_warnings`), loop detector, and domain memory distiller
- [`hyperagent/tools.py`](../hyperagent/tools.py) & [`hyperagent/prompts.py`](../hyperagent/prompts.py) — Tool schemas, risk gates (`needs_approval`), and system prompts
- [`hyperagent/llm.py`](../hyperagent/llm.py) & [`hyperagent/anthropic_backend.py`](../hyperagent/anthropic_backend.py) — Multi-provider LLM client (Anthropic prompt caching / OpenRouter / OpenAI-compatible)
- [`hyperagent/extract.py`](../hyperagent/extract.py) — Structured multi-URL data extraction mixin (`extract`, `record_findings`)
- [`hyperagent/telemetry.py`](../hyperagent/telemetry.py) — Token, latency, and cost ledger (`RunLedger`)
- [`hyperagent/system1.py`](../hyperagent/system1.py) & [`hyperagent/fast.py`](../hyperagent/fast.py) — Speculative multi-head System 1 executor (TypeSafe Jev + Haiku text helper)

**Install Command (via CLI only):**
```bash
uv add playwright "httpx[http2]" anthropic pydantic
uv run playwright install chromium
```

**Files/Folders to Exclude When Extracting:**
- `hyperagent/server.py` *(only needed if running the standalone FastAPI WebSocket server for `apps/web`)*
- `apps/web/`, `legacy/`, `core_engine/`, `dist/`, `artifacts/`, `examples/`, `scripts/`, `test.py`, `Dockerfile`, `railway.json`, `render.yaml`

---

## 2. Upgrade #1: Bulletproof Persistent Login Sessions ("Log In Once, Fetch Anytime")

### Problem with `storage_state.json` Alone on Sites Like Faire.com
Currently, [`BrowserSession.start`](../hyperagent/browser.py) calls `browser.launch()` + `browser.new_context(storage_state=".hyperagent/storage_state.json")`. This has three gaps on modern web apps (Faire.com, Shopify, Firebase Auth, Auth0, Clerk):
1. **Misses `IndexedDB` & Service Workers**: Playwright's `storage_state.json` only exports cookies and `localStorage`. Apps that store auth refresh tokens or device keys in `IndexedDB` appear logged out on the next run.
2. **Wipes Ephemeral Device State**: Spawning a fresh `new_context()` on every run clears HTTP cache, Service Workers, and TLS session resumption tickets, causing Cloudflare / PerimeterX / DataDome to treat the browser as an unrecognized device and re-trigger login or bot challenges.
3. **Not Default in CLI / UI**: `tools/run_task.py` does not load `storage_state` by default, and the Web UI defaults `persistSession` to `false`.

### Solution Architecture (`launch_persistent_context`)
1. **Persistent Chrome User Data Directory (`.hyperagent/browser-profile`)**:
   - Update [`BrowserSession.start`](../hyperagent/browser.py) to support `user_data_dir: Path | None = Path(".hyperagent/browser-profile")` using Playwright's `playwright.chromium.launch_persistent_context(user_data_dir=..., ...)`.
   - When `user_data_dir` is enabled (the default for persistent sessions), Chromium writes **Cookies, `localStorage`, `IndexedDB`, Service Workers, and Cloudflare/device trust tokens** directly to disk in real time.
   - Keep `save_storage_state(".hyperagent/storage_state.json")` as a portable JSON backup alongside the profile directory.
2. **Default Persistence Across Both CLI & Web UI**:
   - Log into Faire.com (or any site) once in headed (`HEADLESS=0`) or Takeover mode; every subsequent run reuses `.hyperagent/browser-profile` automatically without asking to log in again, until the operator explicitly runs **Clear session**.

---

## 3. Upgrade #2: Smooth, Zero-Glitch Browser Viewport in the Web Console

### Problem with the Current `_frame_loop` in `browser.py` + `BrowserView.tsx`
1. **2.5 FPS Screenshot Polling Blocks Playwright**:
   - [`BrowserSession._frame_loop`](../hyperagent/browser.py) loops every `0.4s` calling `await page.screenshot(type="jpeg", quality=62)`.
   - `page.screenshot()` is a heavy synchronous capture command (100–250ms) that competes on the same Playwright pipe with `click()`, `type_text()`, and `observe()`.
2. **Redundant Per-Frame Tab/Title Queries**:
   - `_send_frame` calls `await page.title()` and `await self.tabs()` (evaluating JS across every open tab) 2.5 times per second even when no navigation occurred.
3. **Full React State Re-renders on Every Frame**:
   - Every WebSocket `frame` message updates React state (`setFrame`), re-rendering the component tree on every JPEG frame.

### Solution Architecture (CDP `Page.startScreencast` + Direct Ref Rendering)
1. **Hardware Compositor Push via CDP `Page.startScreencast` (`hyperagent/browser.py`)**:
   - Attach a CDPSession (`cdp = await self.context.new_cdp_session(page)`) on the active page and call:
     ```python
     await cdp.send("Page.startScreencast", {
         "format": "jpeg",
         "quality": 68,
         "maxWidth": 1280,
         "maxHeight": 960,
         "everyNthFrame": 2,
     })
     ```
   - Listen to `cdp.on("Page.screencastFrame", ...)` and acknowledge immediately with `Page.screencastFrameAck`.
   - Chromium's GPU compositor pushes frames **only when screen pixels actually change** at **15–25 FPS** with near-zero CPU overhead and **zero blocking** of Playwright actions.
2. **Event-Driven Metadata Caching (`title` & `tabs`)**:
   - Cache `url`, `title`, and `tabs` in `BrowserSession` and refresh them only on `framenavigated`, `domcontentloaded`, or tab open/close/switch events—never inside the high-frequency frame callback.
3. **Direct DOM `imgRef` / `<canvas>` Frame Updates (`apps/web/src/components/BrowserView.tsx` & `use-agent.ts`)**:
   - Write incoming base64 JPEG frames directly to `imgRef.current.src` (or draw onto a `<canvas>`) without triggering a full React state re-render unless `url`, `title`, or `tabs` changed.
