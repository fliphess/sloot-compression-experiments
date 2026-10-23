"""E8c: find training books that leak test-book text, so they can be excluded.

Two checks per candidate book:
  content  share of a test book's distinct 8-word shingles (lowercase a-z words)
           that also occur in the candidate; > 1% for any test book = leak.
           Reverse direction too: > 5% of the candidate's own shingles coming
           from one test book = leak (catches excerpts, e.g. one Bible chapter)
  title    catalog title/author matches a test book (other editions, parts of
           the same work); always excluded, whatever the content check says

Results are cached per book in lab/exp_table/leakage.tsv, so it can be rerun as
more books arrive.

Usage: python3 lab/exp_table/leakage.py [--procs 3]
"""
import argparse
import csv
import re
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lab"))
from steno import split_books  # noqa: E402

BIG = ROOT / "corpus" / "gutenberg_big"
CACHE = HERE / "leakage.tsv"
SHINGLE = 8
THRESHOLD = 0.01
REVERSE_THRESHOLD = 0.05
WORD = re.compile(r"[a-z]+")
MULT = np.uint64(0x9E3779B97F4A7C15)

# (test id, title regex, author regex or None)
TITLE_RULES = [
    (10, r"\bbible\b|king james", None),
    (46, r"christmas carol", r"dickens"),
    (145, r"middlemarch", None),
    (408, r"souls of black folk", None),
    (1232, r"^the prince\b|^il principe", r"machiavelli"),
    (1952, r"yellow wall", None),
    (2701, r"moby.dick|the whale", r"melville"),
    (5197, r"^my life", r"wagner"),
    (20203, r"autobiography", r"franklin"),
    (37683, r"chambers.*dictionary", None),
]

_test_vocab = None
_test_shingles = None


def word_ids(text, vocab):
    return np.array([vocab.get(w, 0) for w in WORD.findall(text.lower())], dtype=np.uint64)


def shingles(ids):
    """Distinct hashes of all 8-word windows that contain no unknown word (id 0)."""
    if len(ids) < SHINGLE:
        return np.empty(0, dtype=np.uint64)
    n = len(ids) - SHINGLE + 1
    h = np.zeros(n, dtype=np.uint64)
    for j in range(SHINGLE):
        h = h * MULT + ids[j:j + n]
    unknown = np.concatenate(([0], np.cumsum(ids == 0)))
    valid = (unknown[SHINGLE:] - unknown[:-SHINGLE]) == 0
    return np.unique(h[valid])


def init(test_vocab, test_shingles):
    global _test_vocab, _test_shingles
    _test_vocab, _test_shingles = test_vocab, test_shingles


def check(path):
    sh = shingles(word_ids(path.read_text(encoding="utf-8"), _test_vocab))
    shared = {tid: int(np.isin(tsh, sh, assume_unique=True).sum()) for tid, tsh in _test_shingles.items()}
    fracs = {tid: n / len(_test_shingles[tid]) for tid, n in shared.items()}
    rev = {tid: n / max(1, len(sh)) for tid, n in shared.items()}
    tid = max(fracs, key=fracs.get)
    rtid = max(rev, key=rev.get)
    return int(path.stem), tid, float(fracs[tid]), rtid, float(rev[rtid])


def catalog():
    with open(BIG / "pg_catalog.csv", newline="", encoding="utf-8") as f:
        return {int(r["Text#"]): (r["Title"].lower(), r["Authors"].lower()) for r in csv.DictReader(f)}


def title_match(book_id, cat):
    title, author = cat.get(book_id, ("", ""))
    for tid, t_re, a_re in TITLE_RULES:
        if re.search(t_re, title) and (a_re is None or re.search(a_re, author)):
            return tid
    return None


def load_cache():
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines()[1:]:
            out[int(line.split("\t")[0])] = line
    return out


def excluded_ids():
    """Book id -> reason, for every book the cache marks as excluded."""
    out = {}
    for line in load_cache().values():
        book_id, *_, excluded, reason = line.split("\t")
        if excluded == "1":
            out[int(book_id)] = reason
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=3)
    args = ap.parse_args()
    train, test = split_books()
    test_texts = {int(p.stem): p.read_text(encoding="utf-8") for p in test}
    vocab = {}
    for text in test_texts.values():
        for w in WORD.findall(text.lower()):
            vocab.setdefault(w, len(vocab) + 1)
    test_sh = {tid: shingles(word_ids(text, vocab)) for tid, text in test_texts.items()}

    cache = load_cache()
    big = sorted(BIG.glob("*.txt"), key=lambda p: int(p.stem))
    todo = [p for p in train + big if int(p.stem) not in cache]
    cat = catalog()
    print(f"{len(cache)} cached, {len(todo)} to check", flush=True)
    with get_context("fork").Pool(args.procs, initializer=init, initargs=(vocab, test_sh)) as pool:
        for book_id, tid, frac, rtid, rfrac in pool.imap_unordered(check, todo, chunksize=4):
            tmatch = title_match(book_id, cat)
            reasons = []
            if frac > THRESHOLD:
                reasons.append(f"content:{frac:.1%} of test {tid}")
            if rfrac > REVERSE_THRESHOLD:
                reasons.append(f"excerpt:{rfrac:.1%} of book is test {rtid}")
            if tmatch is not None:
                reasons.append(f"title:test {tmatch}")
            cache[book_id] = "\t".join([str(book_id), str(tid), f"{frac:.5f}", str(rtid), f"{rfrac:.5f}", str(tmatch or ""),
                                        "1" if reasons else "0", "; ".join(reasons)])
    lines = sorted(cache.values(), key=lambda l: int(l.split("\t")[0]))
    CACHE.write_text("id\tclosest_test\tshare_of_test\tclosest_rev\tshare_of_book\ttitle_match\texcluded\treason\n" + "\n".join(lines) + "\n")
    print(f"{sum(l.split(chr(9))[6] == '1' for l in lines)} of {len(lines)} books excluded")


if __name__ == "__main__":
    main()
