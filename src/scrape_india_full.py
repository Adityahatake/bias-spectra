# src/scrape_india_full.py
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import pandas as pd
import time
import random
from tqdm import tqdm
from urllib.parse import urljoin, urlparse

# ------- Configuration -------
OUTPUT_CSV = "data/raw/india_news_raw.csv"
TARGET_TOTAL = 1500            # overall target headlines
PER_SOURCE_TARGET = None       # if set, overrides TARGET_TOTAL-based logic
MAX_PAGES_PER_SITE = 30        # max section pages to traverse per site (avoid infinite loops)
REQUEST_DELAY = (0.6, 1.5)     # random delay between requests (seconds)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; BiasBot/1.0; +https://example.com)"}

# Final list of sources and their labels (confirmed with NDTV = Center-Right)
SOURCES = {
    "thewire.in":      {"base": "https://thewire.in",      "category": "Left",         "sections": ["/politics", "/science-technology", "/society"]},
    "scroll.in":       {"base": "https://scroll.in",       "category": "Left",         "sections": ["/topic/politics", "/topic/social-issues"]},
    "thenewsminute.com":{"base": "https://www.thenewsminute.com","category":"Left",     "sections": ["/categories/politics", "/categories/national"]},
    "caravanmagazine.in":{"base":"https://caravanmagazine.in","category":"Left",     "sections": ["/", "/politics"]},

    "thehindu.com":    {"base": "https://www.thehindu.com", "category": "Left-Center",  "sections": ["/news", "/news/national"]},
    "indianexpress.com":{"base":"https://indianexpress.com","category":"Left-Center",  "sections": ["/section/india", "/section/politics"]},

    "indiatoday.in":   {"base": "https://www.indiatoday.in", "category": "Center",       "sections": ["/india", "/politics"]},
    "aajtak.in":       {"base": "https://www.aajtak.in",     "category": "Center",       "sections": ["/english", "/news"]},

    "ndtv.com":        {"base": "https://www.ndtv.com",     "category": "Center-Right", "sections": ["/latest", "/india", "/politics"]},
    "economictimes.indiatimes.com": {"base":"https://economictimes.indiatimes.com", "category":"Center-Right", "sections":["/policy", "/news/india"]},

    "timesnownews.com":{"base":"https://www.timesnownews.com", "category":"Right",       "sections":["/india", "/politics"]},
    "republicworld.com":{"base":"https://www.republicworld.com", "category":"Right",      "sections":["/national", "/politics"]},
    "zeenews.india.com":{"base":"https://zeenews.india.com", "category":"Right",         "sections":["/india", "/politics"]},
    "news18.com":      {"base":"https://www.news18.com",    "category":"Right",         "sections":["/news/india", "/news/politics"]},
}

# --------------------------------

def safe_get(url, session, timeout=10):
    try:
        r = session.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.text
    except Exception:
        return None
    return None

def extract_headline_with_article(url):
    try:
        art = Article(url)
        art.download()
        art.parse()
        title = art.title.strip()
        if title:
            return title
    except Exception:
        pass
    return None

def extract_headline_from_html(html):
    soup = BeautifulSoup(html, "lxml")
    # try common tags
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text().strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text().strip()
    return None

def gather_links_from_section(base_url, html):
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # ignore mailto, javascript
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        # join relative links
        full = urljoin(base_url, href)
        # keep same domain
        if urlparse(full).netloc.endswith(urlparse(base_url).netloc):
            links.add(full.split("?")[0].rstrip("/"))
    return links

def crawl_site(base, sections, label, per_site_target, session):
    seen = set()
    results = []
    to_visit = []
    for sec in sections:
        to_visit.append(urljoin(base, sec))
    pages_crawled = 0

    while to_visit and len(results) < per_site_target and pages_crawled < MAX_PAGES_PER_SITE:
        page = to_visit.pop(0)
        if page in seen:
            continue
        seen.add(page)
        pages_crawled += 1
        html = safe_get(page, session)
        time.sleep(random.uniform(*REQUEST_DELAY))
        if not html:
            continue
        # collect candidate article links
        links = gather_links_from_section(base, html)
        # try to extract headline from the page itself (some pages are single-article pages)
        headline = extract_headline_from_html(html)
        if headline:
            results.append((headline, page))
            # if we already reached target continue but still expand links
            if len(results) >= per_site_target:
                break
        # add new links to queue
        for l in links:
            if l not in seen and len(to_visit) < 500:
                to_visit.append(l)
        # from links, try to extract headlines using Article parser until we fill per_site_target
        for link in list(links)[:150]:  # limit attempts per page
            if len(results) >= per_site_target:
                break
            # skip likely non-article urls (tags, author pages, etc)
            if any(x in link for x in ["/video", "/gallery", "/photos", "/tag/", "/tags/", "/author/"]):
                continue
            title = extract_headline_with_article(link)
            time.sleep(0.2)
            if title:
                results.append((title, link))
    # dedupe titles
    cleaned = []
    seen_titles = set()
    for t, u in results:
        key = t.strip().lower()
        if key not in seen_titles:
            seen_titles.add(key)
            cleaned.append((t, u))
    return cleaned

def main():
    session = requests.Session()
    all_rows = []
    # decide per-site target
    sites = list(SOURCES.items())
    if PER_SOURCE_TARGET:
        per_site = PER_SOURCE_TARGET
    else:
        per_site = max(60, int((TARGET_TOTAL / max(1, len(sites))) + 0.5))  # at least 60 per site
    print(f"Target headlines per site ≈ {per_site}")
    for domain, info in sites:
        base = info["base"]
        cat = info["category"]
        sections = info.get("sections", ["/"])
        print(f"\nCrawling {domain} ({cat}) ...")
        try:
            rows = crawl_site(base, sections, cat, per_site, session)
            print(f"  → Found {len(rows)} candidate headlines")
            for title, url in rows:
                all_rows.append({"headline": title, "url": url, "source": domain, "category": cat})
        except Exception as e:
            print("  !! Error crawling", domain, e)

    # dedupe globally
    df = pd.DataFrame(all_rows)
    df['headline_norm'] = df['headline'].str.lower().str.strip()
    df = df.drop_duplicates(subset=['headline_norm'])
    df = df.drop(columns=['headline_norm'])
    current_total = len(df)
    print(f"\nTotal unique headlines scraped: {current_total}")

    # If we have fewer than TARGET_TOTAL, we will relax per-site limit and try limited second pass using Article on known site homepages
    if current_total < TARGET_TOTAL:
        print("Not enough headlines — performing light second pass on site homepages to add more.")
        for domain, info in sites:
            if len(df) >= TARGET_TOTAL:
                break
            base = info["base"]
            cat = info["category"]
            try:
                html = safe_get(base, session)
                links = gather_links_from_section(base, html or "")
                for l in list(links)[:300]:
                    if len(df) >= TARGET_TOTAL:
                        break
                    title = extract_headline_with_article(l)
                    if title and title.lower().strip() not in set(df['headline'].str.lower().str.strip()):
                        df = pd.concat([df, pd.DataFrame([{
                            "headline": title, "url": l, "source": domain, "category": cat
                        }])], ignore_index=True)
                        time.sleep(0.15)
            except Exception:
                pass
    # Final dedupe & save
    df = df.drop_duplicates(subset=['headline'])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved raw dataset: {OUTPUT_CSV}  (total {len(df)})")

if __name__ == "__main__":
    main()
