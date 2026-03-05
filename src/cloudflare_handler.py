import asyncio
import argparse
from pathlib import Path
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE_FILE = ROOT / "artifacts" / "scraping" / "shared" / "cf_state.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manually prime and persist Cloudflare browser state.")
    p.add_argument("--url", type=str, default="https://www.ufurnish.com")
    p.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    return p.parse_args()


async def prime_cloudflare(url: str, state_file: Path):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)

        input("Complete Cloudflare in the opened browser, then press Enter here...")

        state_file.parent.mkdir(parents=True, exist_ok=True)
        await context.storage_state(path=str(state_file))
        await browser.close()
        print("Saved:", state_file)

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(prime_cloudflare(args.url, args.state_file))