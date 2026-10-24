from pathlib import Path

from velox.browser import BrowserSession


async def test_persistent_profile_and_cdp_screencast(tmp_path: Path) -> None:
    profile_dir = tmp_path / "browser-profile"
    frames: list[dict] = []

    async def on_frame(frame: dict) -> None:
        frames.append(frame)

    html = (
        "data:text/html,"
        "<title>Velox Test</title>"
        "<script>"
        "localStorage.setItem('token', 'persisted-ok');"
        "</script>"
        "<button id='btn'>Click Me</button>"
    )
    browser = BrowserSession(on_frame=on_frame)
    try:
        await browser.start("about:blank", trusted=True, user_data_dir=profile_dir)
        await browser.navigate(html, trusted=True)
        obs = await browser.observe()
        assert any("Click Me" in (el.get("label") or "") for el in obs.elements.values())
        ua_info = await browser.page.evaluate(
            "() => ({ ua: navigator.userAgent, wd: navigator.webdriver, "
            "brands: (navigator.userAgentData?.brands || []).map(b => b.brand) })"
        )
        assert "HeadlessChrome" not in ua_info["ua"]
        assert not ua_info["wd"]
        assert "HeadlessChrome" not in ua_info["brands"]
        await browser.send_frame()
        assert len(frames) >= 1
        assert frames[-1]["title"] == "Velox Test"
        assert profile_dir.is_dir()
    finally:
        await browser.close()

    assert BrowserSession.clear_profile(profile_dir) is True
    assert not profile_dir.exists()
