"""One owned Playwright browser context per operator session, local or over remote CDP."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import functools
import ipaddress
import logging
import os
import re
import shutil
import socket
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from playwright.async_api import Browser, BrowserContext, Page, Playwright, Route, async_playwright
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeout

log = logging.getLogger("hyperagent")

OBSERVE_JS = Path(__file__).with_name("dom.js").read_text()
VIEWPORT = {"width": 1280, "height": 960}
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)
TEXT_BUDGET = 14000
TIME_SCALE = 1.0  # multiplies every @bounded limit; tests shrink it
READ_BUDGET = 40000
SUGGESTION_WAIT_MS = int(os.environ.get("SUGGESTION_WAIT_MS", "2000"))
AUTOCOMPLETE_STATE_JS = """(field) => {
  const autocomplete = field.tagName !== 'SELECT' && !field.list && (
    field.getAttribute('role') === 'combobox' ||
    ['list', 'both'].includes(field.getAttribute('aria-autocomplete')));
  const ids = ((field.getAttribute('aria-controls') || '') + ' ' +
    (field.getAttribute('aria-owns') || '')).split(/\\s+/).filter(Boolean);
  const owner = field.closest('[role="combobox"]');
  const roots = ids.length ? ids.map(id => document.getElementById(id)).filter(Boolean) :
    owner && owner !== field ? [owner] : [];
  const options = roots.flatMap(r => [...r.querySelectorAll('[role="option"]')]).filter(o => {
    const b = o.getBoundingClientRect();
    return b.width && b.height && b.bottom > 0 && b.top < innerHeight &&
      o.checkVisibility({checkOpacity: true, checkVisibilityCSS: true}) && !o.closest('[aria-hidden="true"]');
  });
  return {autocomplete, value: field.value || '',
    needs_open: autocomplete && field.getAttribute('aria-expanded') === 'false' &&
      field !== document.activeElement,
    signature: JSON.stringify(options.map(o => [o.getAttribute('aria-label'), o.textContent])),
    ready: options.length > 0 && field.getAttribute('aria-busy') !== 'true' &&
      !roots.some(r => r.getAttribute?.('aria-busy') === 'true')};
}"""

# Read-only: wait for this field's new options, not an old list or another control's suggestions.
SUGGESTIONS_JS = """(field, {timeout, before, changed}) => new Promise((resolve) => {
  const state = """ + AUTOCOMPLETE_STATE_JS + """;
  let frames = 0, stopped = false, previous = '';
  const finish = (ready) => { stopped = true; clearTimeout(timer); resolve(ready); };
  const timer = setTimeout(() => finish(false), timeout);
  const ready = () => {
    if (stopped) return;
    const current = state(field);
    const fresh = current.ready && (!changed || current.signature !== before);
    frames = fresh ? (current.signature === previous ? frames + 1 : 1) : 0;
    previous = current.signature;
    if (frames >= 2) finish(true); else requestAnimationFrame(ready);
  };
  requestAnimationFrame(ready);
})"""

# Full-page text with link targets and the noise removed, for extraction beyond the viewport.
READ_PAGE_JS = Path(__file__).with_name("reader.js").read_text()

# After an action, give the page a moment to react and then wait until its DOM stops changing: menus, dialogs
# and date pickers usually appear a little after the click and finish rendering a few hundred milliseconds
# later, and an observation taken before that shows the page the action started from, not the one it produced.
SETTLE_MIN_MS = 300
SETTLE_QUIET_MS = 150
SETTLE_TIMEOUT_MS = 1500
SETTLE_JS = """({min, quiet, timeout}) => new Promise((resolve) => {
  const started = performance.now();
  let timer = null, deadline = null, done = false;
  const finish = () => {
    if (done) return;
    done = true; observer.disconnect(); clearTimeout(timer); clearTimeout(deadline); resolve();
  };
  const settle = () => {
    clearTimeout(timer);
    timer = setTimeout(finish, Math.max(quiet, min - (performance.now() - started)));
  };
  const observer = new MutationObserver(settle);
  observer.observe(document.documentElement, {subtree: true, childList: true, attributes: true, characterData: true});
  settle();
  deadline = setTimeout(finish, timeout);
})"""


def normalize_url(url: str) -> str:
    """Same page, same key: no fragment, no trailing slash, lower-case scheme and host."""
    parts = urlsplit(str(url).strip())
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def reloads(target: str, current: str) -> bool:
    """True when opening target would reload the page the tab already shows. A URL with a #fragment only moves
    within the page (to a section or a tab), which keeps everything entered."""
    return not urlsplit(target).fragment and normalize_url(target) == normalize_url(current)


# Native date/time inputs only accept these value formats; typed text is converted to them.
NATIVE_FORMATS = {
    "date": ("%Y-%m-%d", ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y",
                          "%d %b %Y", "%m-%d-%Y", "%d.%m.%Y"]),
    "time": ("%H:%M", ["%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p", "%I %p", "%I%p"]),
    "month": ("%Y-%m", ["%Y-%m", "%m/%Y", "%B %Y", "%b %Y"]),
    "datetime-local": ("%Y-%m-%dT%H:%M", ["%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M", "%m/%d/%Y %I:%M %p"]),
}


def normalize_native(kind: str, text: str) -> str:
    """Convert e.g. '10/17/2026' → '2026-10-17' for <input type=date>; unknown shapes pass through unchanged."""
    target, accepted = NATIVE_FORMATS[kind]
    cleaned = text.strip().upper() if kind == "time" else text.strip()
    for fmt in accepted:
        try:
            return datetime.strptime(cleaned, fmt.upper() if kind == "time" and "%p" in fmt else fmt).strftime(target)
        except ValueError:
            continue
    return text


class ActionError(RuntimeError):
    """A browser action could not be carried out; the message is safe to show the operator and model."""


def bounded(seconds: float, *, mutation: bool = False) -> Callable:
    """Hard time limit for a browser operation. A hung page (e.g. a busy bot-check script) never stalls the agent:
    the tab is replaced and the failure becomes a visible, recoverable error."""
    def wrap(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def inner(self: BrowserSession, *args: Any, **kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(fn(self, *args, **kwargs), seconds * TIME_SCALE)
            except TimeoutError:
                await self._replace_stuck_tab()
                outcome = " Whether the action took effect is unknown." if mutation else ""
                raise ActionError(f"The page stopped responding for {round(seconds)} s, so I replaced that tab with "
                                  f"a blank one.{outcome} Open the page again or try another source.") from None
        return inner
    return wrap


@lru_cache(maxsize=512)
def _host_is_public(host: str) -> bool:
    """False for loopback, private, link-local and other internal targets (blocks SSRF from a hosted agent)."""
    host = host.strip("[]").lower()
    if not host or host == "localhost" or host.endswith((".internal", ".local", ".localhost")):
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        return True  # unresolvable names fail to load anyway
    return all(ipaddress.ip_address(info[4][0]).is_global for info in infos)


def _redact_endpoints(message: str, endpoint: str | None) -> str:
    """Browser-service URLs often carry an API key; keep them out of logs."""
    if endpoint:
        message = message.replace(endpoint, "<BROWSER_WS_URL>")
    return re.sub(r"(wss?://[^\s?]+)\?\S*", r"\1?<redacted>", message)


def private_network_allowed() -> bool:
    return os.environ.get("ALLOW_PRIVATE_NETWORK", "0") == "1"


@dataclass
class Observation:
    url: str
    title: str
    text: str
    truncated: bool
    elements: dict[int, dict[str, Any]]
    scroll: dict[str, int]
    tabs: list[dict[str, Any]]
    screenshot: str | None = None  # base64 jpeg, for the model when vision is enabled

    def element(self, element_id: int) -> dict[str, Any]:
        if element_id not in self.elements:
            raise ActionError(f"Element [{element_id}] is not on the current page. Use an id from the latest state.")
        return self.elements[element_id]


@dataclass
class BrowserSession:
    on_frame: Callable[[dict[str, Any]], Awaitable[None]] | None = None
    playwright: Playwright | None = None
    browser: Browser | None = None
    context: BrowserContext | None = None
    page: Page | None = None
    remote: bool = False
    events: list[str] = field(default_factory=list)  # tab/dialog events for the next observation
    _frame_task: asyncio.Task | None = None
    _last_frame: dict[str, Any] | None = None
    _last_frame_ts: float = 0.0
    _cdp: Any = None
    _cdp_page: Page | None = None
    _cached_title: str = ""
    _cached_url: str = ""
    _cached_tabs: list[dict[str, Any]] = field(default_factory=list)
    _bg_reading: int = 0

    async def start(
        self,
        url: str,
        *,
        trusted: bool = True,
        storage_state: Path | None = None,
        user_data_dir: Path | None = None,
    ) -> None:
        self.playwright = await async_playwright().start()
        endpoint = os.environ.get("BROWSER_WS_URL") or os.environ.get("CDP_WS_URL")
        env_profile = os.environ.get("BROWSER_PROFILE_DIR")
        profile_dir = user_data_dir or (Path(env_profile) if env_profile else None)
        state_arg = str(storage_state) if storage_state and storage_state.is_file() else None
        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--no-sandbox",
        ]
        ignore_default_args = ["--enable-automation"]
        headless = os.environ.get("HEADLESS", "1") != "0"
        use_chrome = Path("/Applications/Google Chrome.app").exists() or os.environ.get("BROWSER_CHANNEL") == "chrome"
        channel = "chrome" if use_chrome else None
        ctx_kwargs: dict[str, Any] = {
            "viewport": VIEWPORT, "locale": "en-US", "accept_downloads": False,
        }
        if not use_chrome:
            ctx_kwargs["user_agent"] = USER_AGENT
        try:
            if endpoint:
                self.remote = True
                self.browser = await self.playwright.chromium.connect_over_cdp(endpoint, timeout=30000)
                if state_arg:
                    ctx_kwargs["storage_state"] = state_arg
                self.context = await self.browser.new_context(**ctx_kwargs)
            elif profile_dir is not None:
                profile_dir.mkdir(parents=True, exist_ok=True)
                self.context = await self.playwright.chromium.launch_persistent_context(
                    str(profile_dir),
                    channel=channel,
                    headless=headless,
                    args=launch_args,
                    ignore_default_args=ignore_default_args,
                    **ctx_kwargs,
                )
            else:
                self.browser = await self.playwright.chromium.launch(
                    channel=channel,
                    headless=headless,
                    args=launch_args,
                    ignore_default_args=ignore_default_args,
                )
                if state_arg:
                    ctx_kwargs["storage_state"] = state_arg
                self.context = await self.browser.new_context(**ctx_kwargs)
        except PlaywrightError as exc:
            await self.close()
            where = "remote browser (BROWSER_WS_URL)" if endpoint else "local Chromium"
            log.error("could not start the %s: %s", where, _redact_endpoints(str(exc), endpoint)[:800])
            if endpoint:
                raise ActionError("Could not connect to the remote browser set in BROWSER_WS_URL. Check that "
                                  "URL, or clear it to use the built-in Chromium.") from exc
            raise ActionError("Could not launch the built-in Chromium; the server log has the reason.") from exc
        if not private_network_allowed():
            await self.context.route("**/*", self._guard)
        with contextlib.suppress(Exception):
            await self.context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
            )
        self.context.on("page", self._on_new_page)
        existing_pages = [p for p in self.context.pages if not p.is_closed()]
        self.page = existing_pages[0] if existing_pages else await self.context.new_page()
        self._watch(self.page)
        await self._apply_stealth(self.page)
        await self._start_screencast(self.page)
        self._frame_task = asyncio.create_task(self._frame_loop())
        await self.navigate(url, trusted=trusted)

    @staticmethod
    def clear_profile(profile_dir: Path) -> bool:
        try:
            if profile_dir.exists():
                shutil.rmtree(profile_dir, ignore_errors=True)
            return True
        except OSError:
            return False

    async def save_storage_state(self, path: Path) -> bool:
        if not self.context:
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            await self.context.storage_state(path=str(path))
            return True
        except PlaywrightError:
            return False

    async def _guard(self, route: Route) -> None:
        """Every request, including redirects and subresources, must target the public internet."""
        url = route.request.url
        if url.startswith(("data:", "blob:", "about:")):
            await route.continue_()
            return
        parts = urlsplit(url)
        public = parts.scheme in ("http", "https") and await asyncio.to_thread(_host_is_public, parts.hostname or "")
        if public:
            await route.continue_()
        else:
            if route.request.is_navigation_request():
                self.events.append(f"Blocked a request to a private or non-web address: {url[:120]}")
            await route.abort("blockedbyclient")

    # ---------- tabs and events ----------

    def _watch(self, page: Page) -> None:
        page.on("dialog", lambda d: asyncio.create_task(self._on_dialog(d)))
        page.on(
            "framenavigated",
            lambda f: asyncio.create_task(self._refresh_metadata()) if f == page.main_frame else None,
        )
        page.on("domcontentloaded", lambda _: asyncio.create_task(self._refresh_metadata()))

    async def _on_dialog(self, dialog: Any) -> None:
        self.events.append(f"A {dialog.type} dialog appeared and was dismissed: {dialog.message[:200]!r}")
        try:
            await dialog.dismiss()
        except PlaywrightError:
            pass

    def _on_new_page(self, page: Page) -> None:
        if self._bg_reading > 0:
            return
        self._watch(page)
        self.page = page
        self.events.append("A new tab opened and is now active.")
        asyncio.create_task(self._apply_stealth(page))
        asyncio.create_task(self._start_screencast(page))
        asyncio.create_task(self._refresh_metadata())

    def _tabs_list(self) -> list[Page]:
        return [p for p in self.context.pages if not p.is_closed()] if self.context else []

    def _active(self) -> Page:
        pages = self._tabs_list()
        if self.page is None or self.page.is_closed():
            if not pages:
                raise ActionError("All tabs are closed.")
            self.page = pages[-1]
        return self.page

    async def tabs(self) -> list[dict[str, Any]]:
        result = []
        for i, p in enumerate(self._tabs_list()):
            try:
                title = await p.title()
            except PlaywrightError:
                title = ""
            result.append({"index": i, "title": title[:80], "url": p.url, "active": p is self.page})
        self._cached_tabs = result
        return result

    async def _refresh_metadata(self) -> None:
        if not self.context:
            return
        with contextlib.suppress(Exception):
            page = self._active()
            self._cached_url = page.url
            self._cached_title = await page.title()
            await self.tabs()

    async def _apply_stealth(self, page: Page) -> None:
        """Strip HeadlessChrome from User-Agent and Sec-CH-UA Client Hints at the CDP protocol level."""
        if not self.context or not hasattr(self.context, "new_cdp_session"):
            return
        cdp = None
        try:
            cdp = await self.context.new_cdp_session(page)
            info = await cdp.send("Browser.getVersion")
            raw_ua = str(info.get("userAgent") or USER_AGENT)
            clean_ua = raw_ua.replace("HeadlessChrome/", "Chrome/")
            m = re.search(r"Chrome/(\d+\.\d+\.\d+\.\d+)", clean_ua)
            full_ver = m.group(1) if m else "145.0.0.0"
            major = full_ver.split(".")[0]
            await cdp.send(
                "Emulation.setUserAgentOverride",
                {
                    "userAgent": clean_ua,
                    "acceptLanguage": "en-US,en;q=0.9",
                    "platform": "macOS",
                    "userAgentMetadata": {
                        "brands": [
                            {"brand": "Not:A-Brand", "version": "99"},
                            {"brand": "Google Chrome", "version": major},
                            {"brand": "Chromium", "version": major},
                        ],
                        "fullVersionList": [
                            {"brand": "Not:A-Brand", "version": "99.0.0.0"},
                            {"brand": "Google Chrome", "version": full_ver},
                            {"brand": "Chromium", "version": full_ver},
                        ],
                        "fullVersion": full_ver,
                        "platform": "macOS",
                        "platformVersion": "15.0.0",
                        "architecture": "arm",
                        "model": "",
                        "mobile": False,
                    },
                },
            )
        except Exception:  # noqa: BLE001 - best-effort stealth override
            if cdp is not None:
                with contextlib.suppress(Exception):
                    await cdp.detach()
        else:
            # Keep the CDP session attached to the page; detaching it resets Emulation overrides in Chromium.
            setattr(page, "_stealth_cdp", cdp)

    # ---------- live frames (CDP Screencast + fallback loop) ----------

    async def _stop_screencast(self) -> None:
        cdp, self._cdp, self._cdp_page = self._cdp, None, None
        if cdp is not None:
            with contextlib.suppress(Exception):
                await cdp.send("Page.stopScreencast")
            with contextlib.suppress(Exception):
                await cdp.detach()

    async def _start_screencast(self, page: Page) -> None:
        if not self.on_frame or not self.context or not hasattr(self.context, "new_cdp_session"):
            return
        if self._cdp is not None and self._cdp_page is page:
            return
        await self._stop_screencast()
        try:
            cdp = await self.context.new_cdp_session(page)
            self._cdp = cdp
            self._cdp_page = page
            cdp.on("Page.screencastFrame", lambda params: asyncio.create_task(self._on_screencast_frame(cdp, params)))
            await cdp.send(
                "Page.startScreencast",
                {
                    "format": "jpeg",
                    "quality": 70,
                    "maxWidth": VIEWPORT["width"],
                    "maxHeight": VIEWPORT["height"],
                    "everyNthFrame": 1,
                },
            )
        except Exception:  # noqa: BLE001 - fallback to _frame_loop if CDP screencast is unsupported
            self._cdp = None
            self._cdp_page = None

    async def _on_screencast_frame(self, cdp: Any, params: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            await cdp.send("Page.screencastFrameAck", {"sessionId": params.get("sessionId")})
        if not self.on_frame or not self.context or cdp is not self._cdp:
            return
        data = params.get("data")
        if not data:
            return
        page = self._active()
        if page.url != self._cached_url or not self._cached_tabs:
            await self._refresh_metadata()
        frame = {
            "image": data,
            "url": page.url,
            "title": self._cached_title,
            "w": VIEWPORT["width"],
            "h": VIEWPORT["height"],
            "tabs": self._cached_tabs,
        }
        self._last_frame = frame
        self._last_frame_ts = time.monotonic()
        with contextlib.suppress(Exception):
            await self.on_frame(frame)

    async def _frame_loop(self) -> None:
        while True:
            try:
                # Only poll screenshot if CDP screencast is not active
                if self._cdp is None and time.monotonic() - self._last_frame_ts > 0.8:
                    await self.send_frame()
            except Exception:  # noqa: BLE001 - frames are best effort and must never stop the agent
                pass
            await asyncio.sleep(0.5)

    async def send_frame(self) -> None:
        if not self.on_frame or not self.context:
            return
        if self._cdp is not None and self._last_frame is not None:
            # CDP screencast pushes frames automatically from the GPU compositor; avoid blocking on page.screenshot
            return
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._send_frame(), 6)

    async def _send_frame(self) -> None:
        page = self._active()
        image = await page.screenshot(type="jpeg", quality=62, timeout=3000, animations="allow")
        title = await page.title()
        tabs = await self.tabs()
        self._cached_url = page.url
        self._cached_title = title
        frame = {
            "image": base64.b64encode(image).decode(), "url": page.url, "title": title,
            "w": VIEWPORT["width"], "h": VIEWPORT["height"], "tabs": tabs,
        }
        self._last_frame = frame
        self._last_frame_ts = time.monotonic()
        await self.on_frame(frame)

    # ---------- observation ----------

    @bounded(25)
    async def observe(self, *, screenshot: bool = False) -> Observation:
        page = self._active()
        data = None
        for _ in range(4):
            try:
                data = await page.evaluate(OBSERVE_JS, TEXT_BUDGET)
                break
            except PlaywrightError:
                # Navigation destroyed the context mid-read; wait for the new document.
                await self._settle(page, 3000)
        if data is None:
            raise ActionError("The page kept navigating and could not be read.")
        shot = None
        if screenshot:
            try:
                shot = base64.b64encode(await page.screenshot(type="jpeg", quality=55, timeout=4000)).decode()
            except PlaywrightError:
                shot = None
        return Observation(
            url=data["url"], title=data["title"], text=data["text"], truncated=data["truncated"],
            elements={e["id"]: e for e in data["elements"]}, scroll=data["scroll"],
            tabs=await self.tabs(), screenshot=shot,
        )

    @bounded(20)
    async def read_page(self) -> dict[str, Any]:
        return await self._active().evaluate(READ_PAGE_JS, READ_BUDGET)

    @bounded(35)
    async def read_urls_parallel(self, urls: list[str], *, trusted: bool = False) -> dict[str, Any]:
        """Open up to 4 URLs concurrently in temporary background tabs, extract their text, and close them."""
        if not self.context:
            raise ActionError("Browser is not running.")
        cleaned: list[str] = []
        for raw in urls[:4]:
            u = str(raw).strip()
            if not u:
                continue
            if trusted and u.startswith(("data:", "about:")):
                pass
            elif "://" not in u:
                u = "https://" + u
            elif not u.startswith(("http://", "https://")):
                raise ActionError(f"Only http and https pages can be opened: {u[:80]}")
            cleaned.append(u)
        if not cleaned:
            raise ActionError("Provide at least one valid http(s) URL.")

        per_page_budget = max(4000, READ_BUDGET // len(cleaned))

        async def _fetch_one(target_url: str) -> dict[str, Any]:
            self._bg_reading += 1
            tab: Page | None = None
            try:
                tab = await self.context.new_page()
                await tab.goto(target_url, wait_until="domcontentloaded", timeout=20000)
                await self._settle(tab, 2500)
                info = await tab.evaluate(READ_PAGE_JS, per_page_budget)
                return {
                    "url": tab.url,
                    "title": await tab.title(),
                    "text": info.get("text", ""),
                    "length": info.get("length", 0),
                    "truncated": bool(info.get("truncated")),
                }
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).splitlines()[0][:140] if str(exc) else type(exc).__name__
                return {"url": target_url, "title": "Error", "text": f"[Could not read {target_url}: {msg}]",
                        "length": 0, "truncated": False}
            finally:
                if tab is not None:
                    with contextlib.suppress(Exception):
                        await tab.close(run_before_unload=False)
                self._bg_reading -= 1

        results = await asyncio.gather(*(_fetch_one(u) for u in cleaned))
        sections = []
        total_len = 0
        any_truncated = False
        for idx, res in enumerate(results, 1):
            total_len += int(res["length"])
            any_truncated = any_truncated or bool(res["truncated"])
            sections.append(f"=== [{idx}] {res['title']} ({res['url']}) ===\n{res['text']}")
        return {
            "text": "\n\n".join(sections),
            "length": total_len,
            "truncated": any_truncated,
            "pages": results,
        }

    @bounded(6)
    async def describe_point(self, x: float, y: float) -> str:
        """Return a human+model description of the DOM element at viewport coordinates (x, y)."""
        page = self._active()
        try:
            info = await page.evaluate(
                """([x, y]) => {
                    const el = document.elementFromPoint(x, y);
                    if (!el) return null;
                    let foundId = null;
                    const map = window.__ha?.nodes;
                    if (map) {
                        for (const [id, node] of map.entries()) {
                            if (node === el || node.contains(el) || el.contains(node)) {
                                foundId = id;
                                break;
                            }
                        }
                    }
                    const tag = el.tagName.toLowerCase();
                    const label = (
                        el.getAttribute('aria-label') ||
                        el.getAttribute('placeholder') ||
                        el.value ||
                        el.innerText ||
                        ''
                    ).replace(/\\s+/g, ' ').trim().slice(0, 80);
                    return { id: foundId, tag, label };
                }""",
                [x, y],
            )
        except PlaywrightError:
            info = None
        if not info:
            return f"pinned point ({round(x)}, {round(y)})"
        id_part = f"[{info['id']}] " if info.get("id") is not None else ""
        label_part = f' "{info["label"]}"' if info.get("label") else ""
        return f"pinned point ({round(x)}, {round(y)}) on {id_part}<{info['tag']}>{label_part}"

    @bounded(8)
    async def element_rect(self, element_id: int) -> dict[str, float] | None:
        """Scroll the element into view (centered when off-screen) and return its current viewport rect."""
        handle = await self._handle(element_id)
        scrolled = False
        try:
            scrolled = bool(
                await handle.evaluate(
                    """(e) => {
                        const r = e.getBoundingClientRect();
                        const h = window.innerHeight || 820;
                        const w = window.innerWidth || 1280;
                        if (r.top < 48 || r.bottom > h - 48 || r.left < 0 || r.right > w) {
                            e.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'instant' });
                            return true;
                        }
                        return false;
                    }"""
                )
            )
            await handle.scroll_into_view_if_needed(timeout=2000)
        except PlaywrightError:
            pass
        if scrolled:
            await asyncio.sleep(0.08)
        box = await handle.bounding_box()
        if self.on_frame:
            with contextlib.suppress(Exception):
                await self.send_frame()
        return {"x": box["x"], "y": box["y"], "w": box["width"], "h": box["height"]} if box else None

    # ---------- actions ----------

    async def _handle(self, element_id: int) -> Any:
        page = self._active()
        handle = await page.evaluate_handle("(id) => window.__ha?.nodes.get(id) ?? null", element_id)
        element = handle.as_element()
        if element is None:
            raise ActionError(f"Element [{element_id}] no longer exists; the page changed. Re-read the page.")
        return element

    async def _settle(self, page: Page, timeout: int = 6000, *, check_network: bool = False) -> None:
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=timeout)
        except PlaywrightTimeout:
            pass
        if check_network:
            try:
                await page.wait_for_load_state("networkidle", timeout=min(timeout, 2000))
            except PlaywrightTimeout:
                pass

    async def _quiesce(self, page: Page) -> None:
        """Wait for the DOM to stop changing after an action (bounded by SETTLE_TIMEOUT_MS)."""
        try:
            await page.evaluate(SETTLE_JS, {"min": SETTLE_MIN_MS, "quiet": SETTLE_QUIET_MS,
                                            "timeout": SETTLE_TIMEOUT_MS})
        except PlaywrightError:
            pass  # the action navigated away; _settle waits for the new document

    @bounded(45)
    async def navigate(self, url: str, *, trusted: bool = False) -> str:
        """Open a URL. Model-chosen URLs are restricted to http(s) so file:// or chrome:// stay unreachable."""
        url = url.strip()
        if trusted and url.startswith(("data:", "about:")):
            pass
        elif "://" not in url:
            url = "https://" + url
        elif not url.startswith(("http://", "https://")):
            raise ActionError("Only http and https pages can be opened.")
        page = self._active()
        if not trusted and reloads(url, page.url):
            # A reload is never a step forward: it discards every value entered and every picker opened so far.
            raise ActionError("You are already on this page; opening it again would reload it and discard what "
                              "has been entered. Use the page's own controls, or go_back.")
        try:
            response = await page.goto(url, wait_until="domcontentloaded", timeout=25000)
        except PlaywrightTimeout:
            raise ActionError(f"Timed out loading {url}.") from None
        except PlaywrightError as exc:
            raise ActionError(f"Could not open {url}: {str(exc).splitlines()[0][:160]}") from None
        await self._settle(page, 4000, check_network=True)
        status = response.status if response else None
        if status and status >= 400:
            return f"Opened {page.url} but the server answered HTTP {status}."
        return f"Opened {page.url}"

    @bounded(40, mutation=True)
    async def click(self, element_id: int) -> str:
        page = self._active()
        handle = await self._handle(element_id)
        before_url, before_tabs = page.url, len(self._tabs_list())
        try:
            await handle.click(timeout=4000)
        except PlaywrightTimeout:
            # Covered or animating; a direct DOM click still reaches the intended element only.
            try:
                await handle.evaluate("(e) => e.click()")
            except PlaywrightError:
                raise ActionError(f"Element [{element_id}] could not be clicked (covered or detached).") from None
        except PlaywrightError as exc:
            raise ActionError(f"Click failed: {str(exc).splitlines()[0][:160]}") from None
        await self._quiesce(page)
        await self._settle(self._active(), 3000)
        if len(self._tabs_list()) > before_tabs:
            return "Clicked; a new tab opened and is now active."
        if self._active().url != before_url:
            return f"Clicked; navigated to {self._active().url}"
        return "Clicked."

    @bounded(40, mutation=True)
    async def type_text(self, element_id: int, text: str, press_enter: bool, *, secret_ok: bool = False) -> str:
        handle = await self._handle(element_id)
        kind = await handle.evaluate("(e) => e.type || ''")
        if kind == "password" and not secret_ok:
            raise ActionError("Password fields need the operator's approval first.")
        autocomplete = await handle.evaluate(AUTOCOMPLETE_STATE_JS)
        if autocomplete["needs_open"]:
            raise ActionError("Open this autocomplete with a click first, then re-observe its search field. "
                              "Nothing was typed.")
        if kind in NATIVE_FORMATS:
            text = normalize_native(kind, text)
        try:
            await handle.fill(text, timeout=4000)
        except PlaywrightError:
            try:
                await handle.click(timeout=3000)
                await self._active().keyboard.press("ControlOrMeta+A")
                await self._active().keyboard.type(text, delay=8)
            except PlaywrightError as exc:
                raise ActionError(f"Could not type into [{element_id}]: {str(exc).splitlines()[0][:160]}") from None
        if press_enter:
            page = self._active()
            before = page.url
            await handle.press("Enter")
            await self._quiesce(page)
            await self._settle(self._active(), 4000)
            if self._active().url != before:
                return f"Typed and pressed Enter; navigated to {self._active().url}"
            return "Typed and pressed Enter."
        if autocomplete["autocomplete"]:
            await self._await_suggestions(handle, autocomplete, text)
            return "Typed a query; select the matching autocomplete suggestion to confirm the value."
        await self._quiesce(self._active())
        return "Typed."

    async def _await_suggestions(self, handle: Any, before: dict[str, Any], text: str) -> None:
        """Typing has already happened. A wait failure is visible and must never replay the mutation."""
        try:
            ready = await handle.evaluate(SUGGESTIONS_JS, {
                "timeout": SUGGESTION_WAIT_MS, "before": before["signature"], "changed": before["value"] != text,
            })
        except PlaywrightError:
            raise ActionError("Text was entered, but its autocomplete suggestions could not be checked. "
                              "Re-observe the page; do not repeat typing automatically.") from None
        if not ready:
            raise ActionError("Text was entered, but fresh autocomplete suggestions did not appear in time. "
                              "The value is not confirmed. Re-observe or open the control; do not repeat typing "
                              "automatically.")

    @bounded(20, mutation=True)
    async def select_option(self, element_id: int, option: str) -> str:
        handle = await self._handle(element_id)
        tag = await handle.evaluate("(e) => e.tagName")
        if tag != "SELECT":
            raise ActionError(f"[{element_id}] is not a dropdown; click it and choose the option instead.")
        try:
            chosen = await handle.select_option(label=option, timeout=3000)
        except PlaywrightError:
            try:
                chosen = await handle.select_option(value=option, timeout=3000)
            except PlaywrightError:
                raise ActionError(f"Option {option!r} is not available in [{element_id}].") from None
        await self._quiesce(self._active())
        await self._settle(self._active(), 2500)
        return f"Selected {option!r}." if chosen else f"Option {option!r} was not selected."

    @bounded(15)
    async def scroll(self, direction: str, pages: float = 1.0) -> str:
        page = self._active()
        dy = int(VIEWPORT["height"] * 0.85 * max(0.25, min(pages, 10)))
        dy = -dy if direction == "up" else dy
        before_y = await page.evaluate("() => Math.round(window.scrollY)")
        await page.mouse.move(VIEWPORT["width"] / 2, VIEWPORT["height"] / 2)
        await page.mouse.wheel(0, dy)
        await asyncio.sleep(0.2)
        y = await page.evaluate(
            """([dy, beforeY]) => {
                if (Math.abs(Math.round(window.scrollY) - beforeY) > 4) {
                    return [Math.round(window.scrollY), document.documentElement.scrollHeight];
                }
                const root = document.scrollingElement || document.documentElement;
                const candidates = [root, ...document.querySelectorAll('*')].filter((el) => {
                    if (!el || el.scrollHeight <= el.clientHeight + 20) return false;
                    if (el === root) return true;
                    const s = getComputedStyle(el);
                    if (!['auto', 'scroll', 'overlay'].includes(s.overflowY)) return false;
                    const r = el.getBoundingClientRect();
                    return r.width > 160 && r.height > 120;
                });
                candidates.sort((a, b) => (b.clientWidth * b.clientHeight) - (a.clientWidth * a.clientHeight));
                for (const el of candidates) {
                    const prev = el.scrollTop;
                    el.scrollBy({ top: dy, behavior: 'instant' });
                    if (Math.abs(el.scrollTop - prev) > 1) {
                        return [Math.round(el.scrollTop), el.scrollHeight];
                    }
                }
                window.scrollBy(0, dy);
                return [Math.round(window.scrollY), root.scrollHeight];
            }""",
            [dy, before_y],
        )
        await asyncio.sleep(0.2)
        if self.on_frame:
            with contextlib.suppress(Exception):
                await self.send_frame()
        return f"Scrolled {direction}; now at {y[0]}px of {y[1]}px."

    @bounded(30)
    async def go_back(self) -> str:
        page = self._active()
        before = page.url
        try:
            response = await page.go_back(wait_until="domcontentloaded", timeout=15000)
        except PlaywrightError:
            raise ActionError("Could not go back.") from None
        if response is None and page.url == before:
            raise ActionError("There is no previous page in this tab.")
        await self._settle(page, 3000)
        return f"Went back to {page.url}"

    @bounded(10)
    async def switch_tab(self, index: int) -> str:
        pages = self._tabs_list()
        if not 0 <= index < len(pages):
            raise ActionError(f"There is no tab {index}.")
        self.page = pages[index]
        await self.page.bring_to_front()
        await self._start_screencast(self.page)
        await self._refresh_metadata()
        return f"Switched to tab {index}: {self.page.url}"

    @bounded(10)
    async def close_tab(self) -> str:
        pages = self._tabs_list()
        if len(pages) <= 1:
            raise ActionError("Cannot close the only tab.")
        await self._active().close()
        self.page = self._tabs_list()[-1]
        await self._start_screencast(self.page)
        await self._refresh_metadata()
        return f"Closed the tab; now on {self.page.url}"

    @bounded(20, mutation=True)
    async def press_key(self, key: str) -> str:
        page = self._active()
        try:
            await page.keyboard.press(key)
        except PlaywrightError:
            raise ActionError(f"Unknown key {key!r}.") from None
        await self._quiesce(page)
        await self._settle(self._active(), 3000)
        return f"Pressed {key}."

    # ---------- operator takeover ----------

    @bounded(20, mutation=True)
    async def human_click(self, x: float, y: float) -> str:
        page = self._active()
        if not (0 <= x <= VIEWPORT["width"] and 0 <= y <= VIEWPORT["height"]):
            raise ActionError("Click is outside the page.")
        await page.mouse.click(x, y)
        if self._cdp is None:
            await self._quiesce(page)
            await self._settle(self._active(), 3000)
        return f"Operator clicked at ({round(x)}, {round(y)})."

    @bounded(20, mutation=True)
    async def human_type(self, text: str, press_enter: bool) -> str:
        page = self._active()
        if len(text) > 500:
            raise ActionError("Type at most 500 characters at a time.")
        await page.keyboard.insert_text(text)
        if press_enter:
            await page.keyboard.press("Enter")
            if self._cdp is None:
                await self._settle(self._active(), 4000)
        return "Operator typed text" + (" and pressed Enter." if press_enter else ".")

    @bounded(10, mutation=True)
    async def human_key(self, key: str) -> str:
        page = self._active()
        if len(key) == 1:
            await page.keyboard.insert_text(key)
        else:
            await page.keyboard.press(key)
        return f"Key {key}"

    @bounded(10)
    async def human_scroll(self, dy: float) -> str:
        page = self._active()
        clamped = max(-4000, min(4000, dy))
        before_y = await page.evaluate("() => Math.round(window.scrollY)")
        await page.mouse.move(VIEWPORT["width"] / 2, VIEWPORT["height"] / 2)
        await page.mouse.wheel(0, clamped)
        await page.evaluate(
            """([dy, beforeY]) => {
                if (Math.abs(Math.round(window.scrollY) - beforeY) > 4) return;
                const root = document.scrollingElement || document.documentElement;
                const candidates = [root, ...document.querySelectorAll('*')].filter((el) => {
                    if (!el || el.scrollHeight <= el.clientHeight + 20) return false;
                    if (el === root) return true;
                    const s = getComputedStyle(el);
                    return ['auto', 'scroll', 'overlay'].includes(s.overflowY);
                });
                candidates.sort((a, b) => (b.clientWidth * b.clientHeight) - (a.clientWidth * a.clientHeight));
                for (const el of candidates) {
                    const prev = el.scrollTop;
                    el.scrollBy({ top: dy, behavior: 'instant' });
                    if (Math.abs(el.scrollTop - prev) > 1) return;
                }
            }""",
            [clamped, before_y],
        )
        return "Operator scrolled."

    async def _replace_stuck_tab(self) -> None:
        stuck = self.page
        try:
            fresh = await asyncio.wait_for(self.context.new_page(), 10)
        except (TimeoutError, PlaywrightError, AttributeError):
            return
        self.page = fresh
        self.events.append("A tab stopped responding and was replaced by a blank tab.")
        if stuck is not None:
            asyncio.ensure_future(self._close_quietly(stuck))

    @staticmethod
    async def _close_quietly(page: Page) -> None:
        try:
            await asyncio.wait_for(page.close(run_before_unload=False), 10)
        except (TimeoutError, PlaywrightError):
            pass

    async def close(self) -> None:
        if self._frame_task:
            self._frame_task.cancel()
            self._frame_task = None
        await self._stop_screencast()
        for closer in (self.context, None if self.remote else self.browser):
            if closer is not None:
                try:
                    await asyncio.wait_for(closer.close(), 15)
                except Exception:  # noqa: BLE001 - shutdown must continue, even past a hung page
                    pass
        self.context = self.browser = None
        if self.playwright:
            try:
                await asyncio.wait_for(self.playwright.stop(), 15)
            except Exception:  # noqa: BLE001
                pass
            self.playwright = None


def now() -> float:
    return time.time()
