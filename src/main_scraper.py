import argparse
import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
    Page,
    Response,
)

# ==========================
# CONFIG
# ==========================
ROOT = Path(__file__).resolve().parent.parent

LIMIT = 1000
NAV_TIMEOUT_MS = 60_000
WAIT_UNTIL = "domcontentloaded"      # good default for screenshots
FULL_PAGE = True

# Headed + “human pace”
HEADLESS = False
SLOW_MO_MS = 60  # adds delay to Playwright actions (helps observe + can reduce racey loads)

log = logging.getLogger("screenshots")


@dataclass(frozen=True)
class ScraperPaths:
    run_dir: Path
    output_dir: Path
    progress_file: Path
    log_file: Path
    state_file: Path


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )


# ==========================
# PROGRESS TRACKING
# ==========================
class ProgressTracker:
    """
    Persists per-URL scrape results to PROGRESS_FILE after each iteration.
    On restart, already-completed URLs (status 'ok' or 'skipped') are skipped;
    failed/timeout entries are retried automatically.
    Writes are atomic (tmp → rename) to prevent corruption on crash.
    """

    def __init__(self, limit: int, progress_file: Path):
        self.progress_file = progress_file
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)

        if self.progress_file.exists():
            try:
                data = json.loads(self.progress_file.read_text(encoding="utf-8"))
                done = sum(1 for e in data["urls"].values() if e["status"] in ("ok", "skipped"))
                log.info(
                    f"Loaded progress file: {self.progress_file} | "
                    f"{done} already done, {len(data['urls'])} total entries"
                )
            except Exception as e:
                log.warning(f"Could not load progress file ({e}) — starting fresh.")
                data = self._empty(limit)
        else:
            data = self._empty(limit)
            log.info("No existing progress file — starting fresh.")
        self.data = data

    @staticmethod
    def _empty(limit: int) -> dict:
        return {
            "run_started": datetime.now(timezone.utc).isoformat(),
            "last_updated": None,
            "limit": limit,
            "summary": {"ok": 0, "failed": 0, "blocked": 0, "skipped": 0, "timeout": 0},
            "urls": {},
        }

    def already_done(self, i: int) -> bool:
        entry = self.data["urls"].get(str(i))
        return entry is not None and entry["status"] in ("ok", "skipped")

    def record(self, i: int, url: str, status: str, **kwargs):
        """
        status: ok | failed | timeout | blocked | skipped
        kwargs: duration_s, http_status, screenshot_path, screenshot_kb,
                cloudflare_blocked, error
        """
        entry = {
            "url": url,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self.data["urls"][str(i)] = entry

        s = self.data["summary"]
        key = status if status in s else "failed"
        s[key] = s.get(key, 0) + 1

        self.data["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._flush()
        log.info(f"[{i}] Progress recorded → status={status} | summary={s}")

    def _flush(self):
        tmp = self.progress_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.progress_file)


# ==========================
# HELPERS
# ==========================
def short(s: str, n: int = 180) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def jitter(a: float, b: float) -> float:
    return random.uniform(a, b)


async def human_pause(a: float = 0.25, b: float = 1.2):
    await asyncio.sleep(jitter(a, b))


async def small_mouse_wiggle(page: Page):
    """Small mouse movements within the viewport (helps with some hover/lazy UI, purely for realism/settling)."""
    try:
        vp = page.viewport_size or {"width": 1200, "height": 800}
        x = random.randint(80, max(81, vp["width"] - 80))
        y = random.randint(80, max(81, vp["height"] - 80))
        await page.mouse.move(x, y, steps=random.randint(8, 16))
        await human_pause(0.1, 0.4)
        await page.mouse.move(
            x + random.randint(-40, 40),
            y + random.randint(-30, 30),
            steps=random.randint(6, 14),
        )
    except Exception as e:
        log.debug(f"mouse wiggle skipped: {e}")


async def organic_scroll(page: Page, seconds: float = 3.5):
    """Scroll in uneven steps for a few seconds (helps load images/infinite sections)."""
    end = time.time() + seconds
    while time.time() < end:
        step = random.choice([160, 240, 320, 420, 520]) + random.randint(-40, 60)
        await page.mouse.wheel(0, step)
        await human_pause(0.15, 0.6)


async def maybe_accept_cookies(page: Page):
    """
    Best-effort click on common cookie consent buttons.
    Safe to fail silently; helps reduce banners covering screenshots.
    """
    candidates = [
        "button:has-text('Accept')",
        "button:has-text('Accept all')",
        "button:has-text('Allow all')",
        "button:has-text('I agree')",
        "button:has-text('Agree')",
        "button:has-text('Tout accepter')",
        "button:has-text('Accepter')",
        "button:has-text(\"J'accepte\")",
        "[aria-label*='accept' i]",
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if await loc.is_visible(timeout=700):
                await human_pause(0.2, 0.8)
                await loc.click(timeout=1200)
                log.info("Cookie/consent banner clicked (best-effort).")
                await human_pause(0.2, 0.8)
                return
        except Exception:
            pass


async def wait_for_settle(page: Page):
    """
    Wait for DOM + try networkidle; don’t hard-fail if the site keeps connections open.
    Also waits for fonts where possible (reduces layout shift in screenshots).
    """
    await page.wait_for_load_state("domcontentloaded", timeout=NAV_TIMEOUT_MS)
    try:
        await page.wait_for_load_state("networkidle", timeout=8_000)
    except PlaywrightTimeoutError:
        log.info("networkidle not reached (site may keep connections open) — continuing.")
    try:
        await page.evaluate(
            """async () => { if (document.fonts && document.fonts.status !== 'loaded') { await document.fonts.ready; } }"""
        )
    except Exception:
        pass


async def is_cloudflare_interstitial(page: Page) -> bool:
    """
    Only for reporting status (not bypassing).
    """
    try:
        html = (await page.content()).lower()
        signals = [
            "cloudflare",
            "verify you are human",
            "checking your browser",
            "security verification",
            "ray id",
            "cf-challenge",
            "challenge-platform",
        ]
        return ("cloudflare" in html) and any(sig in html for sig in signals[1:])
    except Exception as e:
        log.warning(f"CF check failed (couldn't read page content): {e}")
        return False


def safe_filename(i: int) -> str:
    return f"screenshot_{i:02d}.png"


def attach_page_logging(page: Page, url_label: str):
    """
    Adds verbose runtime logs: console messages, page errors, failed requests, responses.
    """
    page.on("console", lambda msg: log.info(f"[{url_label}] console.{msg.type}: {short(msg.text)}"))
    page.on("pageerror", lambda exc: log.warning(f"[{url_label}] pageerror: {exc}"))

    def on_request_failed(req):
        failure = req.failure
        err = failure if failure else "unknown"
        log.warning(f"[{url_label}] request FAILED: {req.method} {req.url} | {err}")

    async def on_response(resp: Response):
        try:
            status = resp.status
            if status >= 400:
                log.warning(f"[{url_label}] response {status}: {resp.url}")
        except Exception:
            pass

    page.on("requestfailed", on_request_failed)
    page.on("response", on_response)


async def expand_accordions(page: Page):
    """
    Clicks all accordion heading elements to expand them before screenshotting.
    Uses a single JS evaluate call to click all headings before any DOM reflow
    can invalidate element handles.
    """
    selector = '[class*="AccordionHeading"]'
    try:
        await page.wait_for_selector(selector, timeout=3_000)
    except PlaywrightTimeoutError:
        log.info("No accordion headings found on this page — skipping.")
        return

    # Collapsed accordions have no next sibling — React fully removes the content
    # node from the DOM when closed, so nextElementSibling === null means collapsed.
    clicked = await page.evaluate(
        """(sel) => {
            const headings = Array.from(document.querySelectorAll(sel));
            const collapsed = headings.filter(el => el.nextElementSibling === null);
            collapsed.forEach(el => el.click());
            return collapsed.length;
        }""",
        selector,
    )
    log.info(f"Clicked {clicked} collapsed accordion heading(s) via JS.")

    # Give the page time to animate/expand all sections before screenshotting
    await human_pause(0.5, 1.0)
    await wait_for_settle(page)


async def goto_with_status(page: Page, url: str) -> Optional[Response]:
    """
    Navigate and return the main document response (if available).
    """
    t0 = time.time()
    resp = await page.goto(url, wait_until=WAIT_UNTIL, timeout=NAV_TIMEOUT_MS)
    dt = time.time() - t0

    if resp is None:
        log.warning(f"goto() returned no response object. (Loaded in {dt:.2f}s) | final_url={page.url}")
        return None

    log.info(f"Navigation done in {dt:.2f}s | status={resp.status} | final_url={page.url}")
    return resp


# ==========================
# MAIN
# ==========================
async def take_screenshots(urls: List[str], paths: ScraperPaths, limit: int = LIMIT):
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    log.info("===== START SCREENSHOT RUN =====")
    log.info(f"URLs received: {len(urls)} | limit: {limit}")
    log.info(f"Run dir: {paths.run_dir.resolve()}")
    log.info(f"Output dir: {paths.output_dir.resolve()}")

    if paths.state_file.exists():
        log.info(f"Using storage_state: {paths.state_file.resolve()}")
    else:
        log.error(f"storage_state file NOT FOUND: {paths.state_file.resolve()}")
        log.error("This run will proceed WITHOUT it (very likely to hit bot checks).")

    async with async_playwright() as p:
        log.info(f"Launching Chromium (headless={HEADLESS}, slow_mo={SLOW_MO_MS}ms)...")
        try:
            browser = await p.chromium.launch(
                headless=HEADLESS,
                slow_mo=SLOW_MO_MS,
                args=[
                    "--disable-dev-shm-usage",
                    "--window-position=3700,900",   # adjust to match your monitor layout
                    "--window-size=1920,1080",
                    # If you run inside containers, you might also need:
                    # "--no-sandbox",
                ],
            )
        except Exception as e:
            log.error(f"FAILED to launch Chromium: {e}")
            log.error("Fixes: run `python -m playwright install` and ensure Chromium is installed.")
            raise

        context_kwargs = {
            "viewport": {"width": 1920, "height": 1080},
            # A couple of “normal browser” defaults (for consistency)
            "locale": "en-US",
            "timezone_id": "Europe/Paris",
        }
        if paths.state_file.exists():
            context_kwargs["storage_state"] = str(paths.state_file)

        log.info(f"Creating browser context with: {list(context_kwargs.keys())}")
        context = await browser.new_context(**context_kwargs)

        tracker = ProgressTracker(limit=limit, progress_file=paths.progress_file)

        for i, url in enumerate(urls[:limit], start=1):
            label = f"{i}/{min(limit, len(urls))}"
            screenshot_path = paths.output_dir / safe_filename(i)

            log.info("------------------------------------------------------------")

            if tracker.already_done(i):
                log.info(f"[{label}] Already completed in previous run — skipping. ({url})")
                continue

            log.info(f"[{label}] Starting URL: {url}")
            log.info(f"[{label}] Screenshot path: {screenshot_path}")

            page = await context.new_page()
            attach_page_logging(page, label)

            t0 = time.time()
            http_status = None
            cf_blocked = False

            try:
                await human_pause(0.4, 1.4)

                # Navigate
                resp = await goto_with_status(page, url)
                http_status = resp.status if resp is not None else None

                if resp is not None:
                    if resp.status >= 500:
                        log.error(f"[{label}] SERVER ERROR status={resp.status} (still attempting screenshot)")
                    elif resp.status >= 400:
                        log.error(f"[{label}] CLIENT ERROR status={resp.status} (still attempting screenshot)")

                # Let things settle and render
                await wait_for_settle(page)

                # “Organic” activity to help lazy-load / stabilize page before screenshot
                await maybe_accept_cookies(page)
                await small_mouse_wiggle(page)
                await organic_scroll(page, seconds=jitter(2.0, 5.0))

                # Sometimes scroll up a bit before final shot
                if random.random() < 0.35:
                    await page.mouse.wheel(0, -random.randint(200, 700))
                    await human_pause(0.2, 0.9)

                await wait_for_settle(page)

                # Expand any accordion sections to reveal hidden details
                await expand_accordions(page)

                # Bot-check visibility (for your status output)
                cf_blocked = await is_cloudflare_interstitial(page)
                if cf_blocked:
                    log.error(f"[{label}] Cloudflare/interstitial detected. Saving screenshot for inspection.")

                # Screenshot
                t1 = time.time()
                await page.screenshot(path=str(screenshot_path), full_page=FULL_PAGE)
                dt_shot = time.time() - t1

                if screenshot_path.exists():
                    size_kb = screenshot_path.stat().st_size / 1024
                    log.info(f"[{label}] Screenshot saved ({size_kb:.1f} KB) in {dt_shot:.2f}s")
                    tracker.record(i, url,
                                   status="blocked" if cf_blocked else "ok",
                                   duration_s=round(time.time() - t0, 2),
                                   http_status=http_status,
                                   screenshot_path=str(screenshot_path),
                                   screenshot_kb=round(size_kb, 1),
                                   cloudflare_blocked=cf_blocked,
                                   error=None)
                else:
                    log.error(f"[{label}] Screenshot call finished, but file not found on disk.")
                    tracker.record(i, url, status="failed",
                                   duration_s=round(time.time() - t0, 2),
                                   http_status=http_status,
                                   screenshot_path=None,
                                   screenshot_kb=None,
                                   cloudflare_blocked=cf_blocked,
                                   error="screenshot file not found after save")

            except PlaywrightTimeoutError:
                log.error(f"[{label}] TIMEOUT: page.goto exceeded {NAV_TIMEOUT_MS/1000:.0f}s")
                log.error(f"[{label}] Last known URL: {getattr(page, 'url', '<unknown>')}")
                tracker.record(i, url, status="timeout",
                               duration_s=round(time.time() - t0, 2),
                               http_status=http_status,
                               screenshot_path=None,
                               screenshot_kb=None,
                               cloudflare_blocked=False,
                               error=f"page.goto exceeded {NAV_TIMEOUT_MS/1000:.0f}s")

            except Exception as e:
                log.exception(f"[{label}] UNEXPECTED ERROR: {e}")
                tracker.record(i, url, status="failed",
                               duration_s=round(time.time() - t0, 2),
                               http_status=http_status,
                               screenshot_path=None,
                               screenshot_kb=None,
                               cloudflare_blocked=cf_blocked,
                               error=str(e))

            finally:
                try:
                    await page.close()
                    log.info(f"[{label}] Page closed.")
                except Exception as e:
                    log.warning(f"[{label}] Failed to close page cleanly: {e}")

        log.info("------------------------------------------------------------")
        log.info("Closing context and browser...")
        try:
            await context.close()
        except Exception as e:
            log.warning(f"Context close issue: {e}")

        try:
            await browser.close()
        except Exception as e:
            log.warning(f"Browser close issue: {e}")

    s = tracker.data["summary"]
    log.info("===== RUN COMPLETE =====")
    log.info(f"Saved screenshots: {s['ok']}")
    log.info(f"Failures: {s['failed']}")
    log.info(f"Timeouts: {s['timeout']}")
    log.info(f"Cloudflare blocked: {s['blocked']}")
    log.info(f"Skipped (already done): {s['skipped']}")
    log.info(f"Progress file: {paths.progress_file.resolve()}")
    log.info(f"Output dir: {paths.output_dir.resolve()}")


# ==========================
# ENTRYPOINT
# ==========================
def parse_args() -> argparse.Namespace:
    default_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    p = argparse.ArgumentParser(description="Take listing screenshots with resilient progress tracking.")
    p.add_argument(
        "--urls-csv",
        type=Path,
        default=ROOT / "data" / "raw" / "ufurnish_sofa_urls.csv",
        help="Path to input CSV containing a 'url' column.",
    )
    p.add_argument("--limit", type=int, default=LIMIT, help="Maximum number of URLs to process.")
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ROOT / "artifacts" / "scraping",
        help="Base directory for scraping artifacts.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=default_run_id,
        help="Run identifier used to create artifacts subdirectory.",
    )
    p.add_argument(
        "--state-file",
        type=Path,
        default=ROOT / "artifacts" / "scraping" / "shared" / "cf_state.json",
        help="Path to Playwright storage_state JSON used for Cloudflare priming.",
    )
    return p.parse_args()


if __name__ == "__main__":
    import pandas as pd

    args = parse_args()
    run_dir = args.artifacts_dir / args.run_id
    paths = ScraperPaths(
        run_dir=run_dir,
        output_dir=run_dir / "screenshots",
        progress_file=run_dir / "progress" / "progress.json",
        log_file=run_dir / "logs" / "scrape.log",
        state_file=args.state_file,
    )
    configure_logging(paths.log_file)
    log.info(f"Artifacts base dir: {args.artifacts_dir.resolve()}")
    log.info(f"Run artifacts dir: {run_dir.resolve()}")
    log.info(f"Log file: {paths.log_file.resolve()}")
    log.info(f"Progress file: {paths.progress_file.resolve()}")

    urls = pd.read_csv(args.urls_csv)["url"].tolist()

    asyncio.run(take_screenshots(urls, paths=paths, limit=args.limit))