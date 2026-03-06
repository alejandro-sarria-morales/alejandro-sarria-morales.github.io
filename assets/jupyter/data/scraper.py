"""
scraper.py — Crawl and scrape El Espectador politics articles.

Usage:
    # Step 1: crawl article listing pages → article_urls.pkl
    python scraper.py crawl [--pages 2500] [--out article_urls.pkl] [--delay 0.1]

    # Step 2: scrape article text → elespectador_politics.pkl
    python scraper.py scrape [--urls article_urls.pkl] [--out elespectador_politics.pkl] [--delay 1.0]

    # Both steps in sequence
    python scraper.py all [--pages 2500] [--delay 0.5]
"""

import argparse
import pickle
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://www.elespectador.com"
ARCHIVE_URL = BASE_URL + "/archivo/politica/{page}/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Crawler: collect (date, url) pairs from archive listing pages
# ---------------------------------------------------------------------------

def crawl(num_pages: int, out_path: str, delay: float) -> list[tuple[str, str]]:
    """
    Crawl archive listing pages and collect article (date, url) pairs.
    Skips subscriber-only articles. Saves to *out_path*.
    Resumes from an existing file if found.
    """
    try:
        with open(out_path, "rb") as f:
            urls = pickle.load(f)
        print(f"Resuming: loaded {len(urls)} URLs from {out_path}")
    except FileNotFoundError:
        urls = []

    pages_to_crawl = range(1, num_pages + 1)
    errors = 0

    for page_num in tqdm(pages_to_crawl, desc="Crawling listing pages"):
        url = ARCHIVE_URL.format(page=page_num)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            errors += 1
            tqdm.write(f"  [page {page_num}] request error: {e}")
            continue

        soup = BeautifulSoup(resp.content, "html.parser")
        articles = soup.find_all("div", class_="Card-Container")

        for article in articles:
            if "suscriptores" in article.text.lower():
                continue
            h2s = article.find_all("h2")
            if not h2s:
                continue
            a_tag = h2s[0].find("a")
            if not a_tag or not a_tag.get("href"):
                continue
            href = a_tag["href"]
            article_url = href if href.startswith("http") else BASE_URL + href
            # Fix double-base URLs that can appear in archive pages
            article_url = article_url.replace(
                "https://www.elespectador.comhttps://", "https://"
            )

            date_tag = article.find("p", class_="Card-Datetime")
            date = date_tag.get_text(strip=True) if date_tag else ""

            urls.append((date, article_url))

        time.sleep(delay)

        # Checkpoint every 100 pages
        if page_num % 100 == 0:
            with open(out_path, "wb") as f:
                pickle.dump(urls, f)
            tqdm.write(f"  Checkpoint saved — {len(urls)} URLs so far")

    with open(out_path, "wb") as f:
        pickle.dump(urls, f)

    print(f"\nCrawl complete: {len(urls)} URLs saved to {out_path} ({errors} page errors)")
    return urls


# ---------------------------------------------------------------------------
# Scraper: fetch and parse article text
# ---------------------------------------------------------------------------

def scrape_article(url: str) -> tuple[str | None, str | None, str]:
    """
    Fetch a single article and return (title, body, status).
    status is one of: 'ok', 'paywall', 'no_title'.
    """
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    h1 = soup.find("h1", class_="Title ArticleHeader-Title")
    if not h1:
        return None, None, "no_title"

    title = h1.get_text(strip=True)

    paragraphs = soup.find_all(
        "p", class_=lambda c: c and "font--secondary" in c
    )
    # Skip paragraphs that are pure link lists (paywall teaser content)
    body = " ".join(
        p.get_text() for p in paragraphs if len(p.find_all("a")) == 0
    )

    if not body.strip():
        return title, "", "paywall"

    return title, body, "ok"


def scrape(urls_path: str, out_path: str, delay: float) -> pd.DataFrame:
    """
    Scrape articles listed in *urls_path* and save a DataFrame to *out_path*.
    Resumes if *out_path* already exists by skipping already-scraped URLs.
    """
    with open(urls_path, "rb") as f:
        urls: list[tuple[str, str]] = pickle.load(f)

    # Fix any leftover malformed URLs
    urls = [
        (date, url.replace("https://www.elespectador.comhttps://", "https://"))
        for date, url in urls
    ]

    # Resume support: build set of already-scraped URLs
    try:
        with open(out_path, "rb") as f:
            existing: pd.DataFrame = pickle.load(f)
        scraped_urls: set[str] = set(existing.get("url", []))
        rows = existing.to_dict("records")
        print(f"Resuming: {len(rows)} articles already scraped")
    except FileNotFoundError:
        scraped_urls = set()
        rows = []

    remaining = [(d, u) for d, u in urls if u not in scraped_urls]
    errors = {"paywall": 0, "no_title": 0, "request_error": 0}

    for i, (date, url) in enumerate(
        tqdm(remaining, desc="Scraping articles", total=len(remaining))
    ):
        try:
            title, body, status = scrape_article(url)
            if status == "ok":
                rows.append({"date": date, "url": url, "title": title, "body": body})
            else:
                errors[status] += 1
        except Exception as e:
            errors["request_error"] += 1
            tqdm.write(f"  Error on {url}: {e}")

        time.sleep(delay)

        # Checkpoint every 500 articles
        if (i + 1) % 500 == 0:
            df_check = pd.DataFrame(rows)
            with open(out_path, "wb") as f:
                pickle.dump(df_check, f)
            tqdm.write(f"  Checkpoint saved — {len(rows)} articles so far")

    arts = pd.DataFrame(rows).query("body != ''").reset_index(drop=True)

    with open(out_path, "wb") as f:
        pickle.dump(arts, f)

    print(f"\nScrape complete: {len(arts)} articles saved to {out_path}")
    print(f"Errors: {errors}")
    return arts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Crawl and scrape El Espectador politics articles."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # crawl subcommand
    p_crawl = sub.add_parser("crawl", help="Collect article URLs from listing pages")
    p_crawl.add_argument("--pages", type=int, default=2500, help="Number of archive pages to crawl (default: 2500)")
    p_crawl.add_argument("--out", default="article_urls.pkl", help="Output file for collected URLs")
    p_crawl.add_argument("--delay", type=float, default=0.1, help="Seconds to wait between page requests (default: 0.1)")

    # scrape subcommand
    p_scrape = sub.add_parser("scrape", help="Scrape article text from collected URLs")
    p_scrape.add_argument("--urls", default="article_urls.pkl", help="Input file with (date, url) pairs")
    p_scrape.add_argument("--out", default="elespectador_politics.pkl", help="Output file for scraped articles")
    p_scrape.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between article requests (default: 1.0)")

    # all subcommand
    p_all = sub.add_parser("all", help="Run crawl then scrape")
    p_all.add_argument("--pages", type=int, default=2500)
    p_all.add_argument("--delay", type=float, default=0.5)

    args = parser.parse_args()

    if args.command == "crawl":
        crawl(args.pages, args.out, args.delay)

    elif args.command == "scrape":
        scrape(args.urls, args.out, args.delay)

    elif args.command == "all":
        urls_path = "article_urls.pkl"
        out_path = "elespectador_politics.pkl"
        crawl(args.pages, urls_path, args.delay)
        scrape(urls_path, out_path, args.delay)


if __name__ == "__main__":
    main()
