"""E8c: download many English Project Gutenberg books for larger reference tables.

Selection: catalog rows with Type == Text and Language == en, minus the 10 test
books and the 100 books already in corpus/gutenberg. Order is a seeded shuffle
of the catalog, so the first N downloaded books form nested corpus tiers.
Downloads come from the gutenberg.pglaf.org mirror: 8 workers sharing a global
limit of 1.4 request starts per second; resumes where it stopped (state in
fetch_log.tsv).

Files are stored as plain text with the licence header/footer stripped,
exactly like lab/fetch_gutenberg.py. Obvious non-prose (huge data files) is
skipped by the size/word filter in clean().

Reproduce: --from-manifest re-downloads exactly the books in
lab/exp_table/manifest.json and verifies each sha256 (warns, doesn't stop, if
Gutenberg has changed a text since).

Usage:
  python3 lab/exp_table/fetch_big.py --target-mb 3000      grow the corpus
  python3 lab/exp_table/fetch_big.py --from-manifest       rebuild it from the manifest
"""
import argparse
import csv
import hashlib
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "corpus" / "gutenberg_big"
CATALOG = OUT / "pg_catalog.csv"
LOG = BIG_LOG = OUT / "fetch_log.tsv"
MANIFEST = Path(__file__).resolve().parent / "manifest.json"
MIRROR = "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt"
UA = "jansloot-research/0.1 (compression research; polite bulk download via mirror)"
TEST_IDS = {10, 46, 145, 408, 1232, 1952, 2701, 5197, 20203, 37683}
START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)
END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", re.I)
RATE = 1.4  # request starts per second, all workers together
WORKERS = 8
SEED = 20261022


def candidates():
    have = {int(p.stem) for p in (ROOT / "corpus" / "gutenberg").glob("*.txt")}
    with open(CATALOG, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["Type"] == "Text" and r["Language"] == "en"]
    ids = [int(r["Text#"]) for r in rows]
    ids = [i for i in ids if i not in TEST_IDS and i not in have]
    random.Random(SEED).shuffle(ids)
    return ids


def clean(text):
    start, end = START.search(text), END.search(text)
    if not start or not end:
        return None, "no-markers"
    body = text[start.end():end.start()].strip().replace("\r\n", "\n")
    if len(body) < 2_000:
        return None, "too-short"
    if len(body) > 15_000_000:
        return None, "too-big"
    words = re.findall(r"[A-Za-z]+", body[:200_000])
    if not words or sum(map(len, words)) / len(words) > 12 or len(words) < len(body[:200_000]) / 20:
        return None, "not-prose"
    return body + "\n", "ok"


class RateLimit:
    def __init__(self, rate):
        self.interval, self.next, self.lock = 1 / rate, time.monotonic(), threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            self.next = max(self.next, now)
            delay = self.next - now
            self.next += self.interval
        time.sleep(delay)


def fetch(book_id, limit):
    limit.wait()
    try:
        req = urllib.request.Request(MIRROR.format(id=book_id), headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        body, status = clean(raw)
    except Exception as exc:
        body, status = None, f"error:{type(exc).__name__}"
    size = 0
    if body is not None:
        path = OUT / f"{book_id}.txt"
        path.write_text(body, encoding="utf-8")
        size = path.stat().st_size
    return book_id, status, size


def from_manifest():
    """Re-download every manifest book that is missing or differs, verify hashes, rewrite the log."""
    books = json.loads(MANIFEST.read_text())["books"]
    OUT.mkdir(parents=True, exist_ok=True)
    limit = RateLimit(RATE)

    def ok(book):
        path = OUT / f"{book['id']}.txt"
        return path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == book["sha256"]

    todo = [b for b in books if not ok(b)]
    print(f"{len(books) - len(todo)} books present and verified, {len(todo)} to download", flush=True)
    changed = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        for book_id, status, size in pool.map(lambda b: fetch(b["id"], limit), todo):
            if status != "ok" or not ok(next(b for b in todo if b["id"] == book_id)):
                changed += 1
                print(f"WARNING: book {book_id} differs from the manifest ({status}); results may shift slightly")
    LOG.write_text("".join(f"{b['id']}\tok\t{b['bytes']}\n" for b in books))
    print(f"done; {changed} books differ from the manifest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mb", type=float, default=3000)
    ap.add_argument("--from-manifest", action="store_true")
    args = ap.parse_args()
    if args.from_manifest:
        from_manifest()
        return
    done = {}
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            book_id, status, size = line.split("\t")
            done[int(book_id)] = (status, int(size))
    total = sum(size for status, size in done.values() if status == "ok")
    todo = iter([i for i in candidates() if i not in done])
    limit = RateLimit(RATE)
    with open(LOG, "a") as log, ThreadPoolExecutor(WORKERS) as pool:
        running = {pool.submit(fetch, next(todo), limit) for _ in range(WORKERS * 2)}
        while running and total < args.target_mb * 1e6:
            finished = next(iter(running))
            for fut in list(running):
                if fut.done():
                    finished = fut
                    break
            else:
                time.sleep(0.05)
                continue
            running.remove(finished)
            book_id, status, size = finished.result()
            total += size
            log.write(f"{book_id}\t{status}\t{size}\n")
            log.flush()
            nxt = next(todo, None)
            if nxt is not None:
                running.add(pool.submit(fetch, nxt, limit))
        for fut in running:
            book_id, status, size = fut.result()
            log.write(f"{book_id}\t{status}\t{size}\n")
    print(f"total {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
