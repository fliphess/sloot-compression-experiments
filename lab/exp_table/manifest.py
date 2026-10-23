"""E8c: write lab/exp_table/manifest.json, the tracked record of the big corpus.

  books     every successfully downloaded book in download order: Gutenberg ID,
            sha256 and byte size of the stripped text as saved
  excluded  test-leak exclusions from leakage.tsv: ID and reason
  skipped   per status (no-markers, not-prose, too-big, errors): the IDs that
            were tried but not saved

Only books already checked by leakage.py are listed (the list stops at the
first unchecked download), so run leakage.py right before this.

Table tiers are prefixes of `books` minus `excluded` (see build_table.py), so
manifest.json + fetch_big.py --from-manifest rebuild everything from scratch.

Usage: python3 lab/exp_table/manifest.py
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from fetch_big import BIG_LOG, MIRROR, OUT, SEED  # noqa: E402
from leakage import excluded_ids, load_cache  # noqa: E402

MANIFEST = HERE / "manifest.json"


def main():
    books, skipped = [], defaultdict(list)
    checked = load_cache()
    for line in BIG_LOG.read_text().splitlines():
        book_id, status, size = line.split("\t")
        if status == "ok" and int(book_id) not in checked:
            break  # stop at the first book leakage.py hasn't checked yet: keeps the list a clean prefix
        if status == "ok":
            data = (OUT / f"{book_id}.txt").read_bytes()
            books.append({"id": int(book_id), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        else:
            skipped[status].append(int(book_id))
    manifest = {
        "description": "English Project Gutenberg books for E8c tables; text stored with licence header/footer "
                       "stripped (fetch_big.clean). Tiers = the 90 codec-v1 training books, then these books "
                       "in order, minus 'excluded'.",
        "source": MIRROR,
        "selection": f"pg_catalog.csv Type=Text, Language=en, minus test and corpus/gutenberg IDs, shuffled with seed {SEED}",
        "books": books,
        "excluded": [{"id": i, "reason": r} for i, r in sorted(excluded_ids().items())],
        "skipped": {k: sorted(v) for k, v in sorted(skipped.items())},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=0))
    print(f"{len(books)} books ({sum(b['bytes'] for b in books) / 1e6:.0f} MB), "
          f"{len(manifest['excluded'])} excluded, {sum(map(len, skipped.values()))} skipped -> {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
