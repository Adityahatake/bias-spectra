"""
NewsScraper – Crawl Indian news sites for headline data.
========================================================
Config-driven scraper with rate limiting, deduplication,
and proper logging. Sources are defined in config.py.
"""

import logging
import random
import time
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    from newspaper import Article
except ImportError:
    Article = None

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    RAW_CSV,
    RAW_DATA_DIR,
    SCRAPE_DELAY_RANGE,
    SCRAPE_MAX_PAGES,
    SCRAPE_SOURCES,
    SCRAPE_TARGET_TOTAL,
    SCRAPE_USER_AGENT,
)

logger = logging.getLogger(__name__)


class NewsScraper:
    """
    Crawl configured news sources and collect headlines.

    Usage:
        scraper = NewsScraper()
        df = scraper.run()   # returns DataFrame and saves CSV
    """

    def __init__(
        self,
        sources: dict | None = None,
        target_total: int = SCRAPE_TARGET_TOTAL,
        max_pages: int = SCRAPE_MAX_PAGES,
        delay: tuple = SCRAPE_DELAY_RANGE,
    ) -> None:
        self.sources = sources or SCRAPE_SOURCES
        self.target_total = target_total
        self.max_pages = max_pages
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": SCRAPE_USER_AGENT})

    # ── Public ───────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Crawl all sources, deduplicate, and save to CSV."""
        all_rows: list[dict] = []
        per_site = max(60, int(self.target_total / max(1, len(self.sources)) + 0.5))
        logger.info("Target ≈ %d headlines per site", per_site)

        for domain, info in tqdm(self.sources.items(), desc="Sources"):
            logger.info("Crawling %s (%s)", domain, info["category"])
            try:
                rows = self._crawl_site(
                    info["base"], info.get("sections", ["/"]),
                    info["category"], per_site,
                )
                logger.info("  → %d headlines from %s", len(rows), domain)
                for title, url in rows:
                    all_rows.append({
                        "headline": title, "url": url,
                        "source": domain, "category": info["category"],
                    })
            except Exception as exc:
                logger.error("Error crawling %s: %s", domain, exc)

        df = self._deduplicate(pd.DataFrame(all_rows))
        logger.info("Total unique headlines: %d", len(df))

        # Second pass if we're short
        if len(df) < self.target_total:
            df = self._second_pass(df)

        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_CSV, index=False)
        logger.info("Saved → %s (%d rows)", RAW_CSV, len(df))
        return df

    # ── Crawl logic ──────────────────────────────────────────

    def _crawl_site(self, base: str, sections: list, label: str, limit: int) -> list:
        seen_urls: set[str] = set()
        results: list[tuple] = []
        queue = [urljoin(base, s) for s in sections]
        pages = 0

        while queue and len(results) < limit and pages < self.max_pages:
            page_url = queue.pop(0)
            if page_url in seen_urls:
                continue
            seen_urls.add(page_url)
            pages += 1

            html = self._get(page_url)
            if not html:
                continue

            # Extract headline from page itself
            headline = self._headline_from_html(html)
            if headline:
                results.append((headline, page_url))

            # Gather links and try extracting headlines
            links = self._gather_links(base, html)
            for link in list(links - seen_urls)[:150]:
                if len(results) >= limit:
                    break
                if any(x in link for x in ["/video", "/gallery", "/photos", "/tag/", "/author/"]):
                    continue
                title = self._headline_from_article(link)
                if title:
                    results.append((title, link))

            # Expand queue
            for link in links:
                if link not in seen_urls and len(queue) < 500:
                    queue.append(link)

        # Deduplicate titles within site
        seen_titles: set[str] = set()
        cleaned = []
        for t, u in results:
            key = t.strip().lower()
            if key not in seen_titles:
                seen_titles.add(key)
                cleaned.append((t, u))
        return cleaned

    def _second_pass(self, df: pd.DataFrame) -> pd.DataFrame:
        """Light second pass on homepages to fill remaining quota."""
        logger.info("Running second pass to fill target (%d/%d)", len(df), self.target_total)
        existing = set(df["headline"].str.lower().str.strip())

        for domain, info in self.sources.items():
            if len(df) >= self.target_total:
                break
            html = self._get(info["base"])
            if not html:
                continue
            links = self._gather_links(info["base"], html)
            for link in list(links)[:300]:
                if len(df) >= self.target_total:
                    break
                title = self._headline_from_article(link)
                if title and title.lower().strip() not in existing:
                    new_row = pd.DataFrame([{
                        "headline": title, "url": link,
                        "source": domain, "category": info["category"],
                    }])
                    df = pd.concat([df, new_row], ignore_index=True)
                    existing.add(title.lower().strip())
                    time.sleep(0.15)
        return df

    # ── Helpers ──────────────────────────────────────────────

    def _get(self, url: str, timeout: int = 10) -> str | None:
        try:
            resp = self.session.get(url, timeout=timeout)
            time.sleep(random.uniform(*self.delay))
            return resp.text if resp.status_code == 200 else None
        except Exception:
            return None

    @staticmethod
    def _headline_from_html(html: str) -> str | None:
        soup = BeautifulSoup(html, "lxml")
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            return og["content"].strip()
        title = soup.find("title")
        if title:
            return title.get_text().strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text().strip()
        return None

    @staticmethod
    def _headline_from_article(url: str) -> str | None:
        if Article is None:
            return None
        try:
            art = Article(url)
            art.download()
            art.parse()
            title = art.title.strip()
            return title if title else None
        except Exception:
            return None

    @staticmethod
    def _gather_links(base_url: str, html: str) -> set:
        soup = BeautifulSoup(html, "lxml")
        links: set[str] = set()
        base_domain = urlparse(base_url).netloc
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith(("mailto:", "javascript:")):
                continue
            full = urljoin(base_url, href)
            if urlparse(full).netloc.endswith(base_domain):
                links.add(full.split("?")[0].rstrip("/"))
        return links

    @staticmethod
    def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
        df["_norm"] = df["headline"].str.lower().str.strip()
        df = df.drop_duplicates(subset=["_norm"]).drop(columns=["_norm"])
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    scraper = NewsScraper()
    scraper.run()
