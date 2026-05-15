"""Scrape SHL Individual Test Solutions catalog.

Output: data/catalog.json with one record per assessment:
  { "name", "url", "test_type", "description", 
    "remote_testing", "adaptive_irt", "languages", "assessment_length" }

The catalog is paginated as ?start=N&type=1 where N steps by 12.
We walk pages until a page returns zero rows. Then we fetch each detail
page concurrently to enrich the record with description + metadata.
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com"
LISTING_URL = BASE + "/solutions/products/product-catalog/?start={start}&type=1"
PAGE_SIZE = 12
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "catalog.json"


def fetch(url: str, retries: int = 3, timeout: int = 30) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.text
            last_err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(1 + attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def parse_listing(html: str) -> list[dict]:
    """Parse a single listing page. Returns list of partial records.

    The catalog page renders BOTH a 'Pre-packaged Job Solutions' table and
    an 'Individual Test Solutions' table. We must only ingest the latter,
    identified by the first <th> text of the table.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []

    target_table = None
    for tbl in soup.find_all("table"):
        th = tbl.find("th")
        if th and "Individual Test Solutions" in th.get_text():
            target_table = tbl
            break
    if target_table is None:
        return rows

    for tr in target_table.find_all("tr"):
        a = tr.select_one("a[href*='/products/product-catalog/view/']")
        if not a:
            continue
        name = a.get_text(strip=True)
        href = a.get("href", "")
        if not name or not href:
            continue
        url = urljoin(BASE, href)

        type_keys = [s.get_text(strip=True) for s in tr.select(".product-catalogue__key")]
        type_keys = [t for t in type_keys if re.fullmatch(r"[A-Z]", t)]

        tds = tr.find_all("td")
        remote = adaptive = None
        if len(tds) >= 3:
            remote = bool(tds[1].select_one(".-yes"))
            adaptive = bool(tds[2].select_one(".-yes"))

        rows.append(
            {
                "name": name,
                "url": url,
                "test_type": "".join(sorted(set(type_keys))),
                "remote_testing": remote,
                "adaptive_irt": adaptive,
            }
        )
    return rows


def parse_detail(html: str) -> dict:
    """Extract description + metadata from a detail page."""
    soup = BeautifulSoup(html, "html.parser")
    out: dict = {}

    # Description is usually inside a section with heading "Description".
    # The detail pages use h4 + p pattern.
    def section_text(heading: str) -> str | None:
        for h in soup.find_all(["h2", "h3", "h4"]):
            if h.get_text(strip=True).lower() == heading.lower():
                # Collect text from siblings until next heading.
                parts = []
                for sib in h.next_siblings:
                    if getattr(sib, "name", None) in ("h2", "h3", "h4"):
                        break
                    txt = getattr(sib, "get_text", lambda **_: "")(separator=" ", strip=True)
                    if txt:
                        parts.append(txt)
                return " ".join(parts).strip() or None
        return None

    out["description"] = section_text("Description") or ""
    out["job_levels"] = section_text("Job levels") or ""
    out["languages"] = section_text("Languages") or ""
    out["assessment_length"] = section_text("Assessment length") or ""

    # Try to capture test type letters from the detail page too, for accuracy.
    keys = [s.get_text(strip=True) for s in soup.select(".product-catalogue__key")]
    keys = [k for k in keys if re.fullmatch(r"[A-Z]", k)]
    if keys:
        out["test_type_detail"] = "".join(sorted(set(keys)))
    return out


def crawl_listings(max_pages: int = 80) -> list[dict]:
    seen_urls: set[str] = set()
    records: list[dict] = []
    for page_idx in range(max_pages):
        start = page_idx * PAGE_SIZE
        url = LISTING_URL.format(start=start)
        print(f"[listing] {url}", file=sys.stderr)
        html = fetch(url)
        page_rows = parse_listing(html)
        if not page_rows:
            print(f"[listing] empty page at start={start}, stopping", file=sys.stderr)
            break
        new_count = 0
        for row in page_rows:
            if row["url"] in seen_urls:
                continue
            seen_urls.add(row["url"])
            records.append(row)
            new_count += 1
        print(f"[listing] start={start} added {new_count} (total {len(records)})", file=sys.stderr)
        if new_count == 0:
            # Defensive: catalog can repeat last page; bail when no new items.
            break
        time.sleep(0.3)
    return records


def enrich_details(records: list[dict], max_workers: int = 8) -> None:
    def worker(rec: dict) -> tuple[dict, dict | None, str | None]:
        try:
            html = fetch(rec["url"])
            return rec, parse_detail(html), None
        except Exception as e:  # noqa: BLE001
            return rec, None, str(e)

    completed = 0
    total = len(records)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, r) for r in records]
        for fut in as_completed(futures):
            rec, detail, err = fut.result()
            completed += 1
            if err:
                print(f"[detail] {rec['name']}: ERROR {err}", file=sys.stderr)
                rec.setdefault("description", "")
                continue
            for k, v in detail.items():
                if k == "test_type_detail":
                    # Prefer detail page if listing missed letters.
                    if v and not rec.get("test_type"):
                        rec["test_type"] = v
                else:
                    rec[k] = v
            if completed % 25 == 0 or completed == total:
                print(f"[detail] {completed}/{total}", file=sys.stderr)


def main() -> None:
    records = crawl_listings()
    print(f"[scrape] crawled {len(records)} unique assessments", file=sys.stderr)
    enrich_details(records)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"[scrape] wrote {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
