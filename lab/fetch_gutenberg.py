"""Download the most popular English Project Gutenberg books as a clean natural-language corpus.

Strips the Gutenberg licence header/footer so only the book text remains.
The "top" list changes daily; to reproduce the research corpus exactly, use
lab/fetch_corpus.py, which downloads the pinned ID list from the manifest.

Usage: python3 lab/fetch_gutenberg.py [--max N]
"""
import argparse
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "corpus" / "gutenberg"
TOP = "https://www.gutenberg.org/browse/scores/top"
BOOK = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)
END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", re.I)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "jansloot-research/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def strip_book(text):
    """Book body without the Gutenberg licence, or None if not an English book with markers."""
    if not re.search(r"^Language: English", text[:5000], re.M):
        return None
    start, end = START.search(text), END.search(text)
    if not start or not end:
        return None
    return text[start.end():end.start()].strip().replace("\r\n", "\n") + "\n"


def download_book(book_id):
    """Download and strip one book; returns the saved text, or None if unusable."""
    return strip_book(fetch(BOOK.format(id=book_id)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = list(dict.fromkeys(re.findall(r'href="/ebooks/(\d+)"', fetch(TOP))))
    saved = 0
    for book_id in ids:
        if saved >= args.max:
            break
        path = OUT / f"{book_id}.txt"
        if path.exists():
            saved += 1
            continue
        try:
            body = download_book(book_id)
        except Exception as exc:
            print(f"{book_id}: skipped ({exc})")
            continue
        time.sleep(1)
        if body is None:
            continue
        path.write_text(body, encoding="utf-8")
        saved += 1
        print(f"{book_id}: {len(body):,} chars")
    print(f"{saved} books in {OUT}")


if __name__ == "__main__":
    main()
