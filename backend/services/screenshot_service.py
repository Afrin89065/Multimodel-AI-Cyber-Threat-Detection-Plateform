"""
Auto-screenshot URLs using Playwright.
PATH: backend/services/screenshot_service.py
INSTALL: pip install playwright && playwright install chromium
"""
from loguru import logger
from typing import Optional


class ScreenshotService:
    def __init__(self):
        self.available = False
        try:
            from playwright.async_api import async_playwright
            self._playwright_mod = async_playwright
            self.available = True
            logger.info("Screenshot service ready (Playwright)")
        except ImportError:
            logger.warning("Playwright not installed. Run: pip install playwright && playwright install chromium")

    async def take_screenshot(self, url: str, timeout_ms: int = 10000) -> Optional[bytes]:
        if not self.available:
            return None
        if not url.startswith("http"):
            url = "https://" + url
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
                ctx = await browser.new_context(viewport={"width": 1280, "height": 720})
                page = await ctx.new_page()
                await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                screenshot = await page.screenshot(type="png", full_page=False)
                await browser.close()
                return screenshot
        except Exception as e:
            logger.warning(f"Screenshot failed for {url}: {e}")
            return None

    async def close(self):
        pass