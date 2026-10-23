"""Download and rebuild the research corpus, verified against lab/corpus_manifest.json.

corpus/ is not in git; this script recreates it exactly:
  enwik8        Hutter Prize benchmark (first 100 MB of English Wikipedia, 2006)
  gutenberg     the pinned 100 Project Gutenberg books (licence text stripped)
  samples       slices and derived files used by E1-E4

Every file is checked against the SHA-256 in the manifest. The deliverable is
the method to build a table, not one specific table, so source drift is fine:
Project Gutenberg occasionally corrects its texts, and a changed or vanished
book is a warning. Tables built from it get a different fingerprint, and
results can differ slightly from the log. enwik8 and the samples are fixed
recipes and must always match. --strict turns every difference into an error.

Usage:
  python3 lab/fetch_corpus.py [--strict]           download what's missing, build samples, verify
  python3 lab/fetch_corpus.py verify [--strict]    only verify existing files
  python3 lab/fetch_corpus.py freeze     write the manifest from the current corpus (maintainers)
"""
import gzip
import hashlib
import json
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

from fetch_gutenberg import download_book

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
DOWNLOADS = CORPUS / "downloads"
MANIFEST = Path(__file__).resolve().parent / "corpus_manifest.json"
MB = 1_000_000

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
CPYTHON_URL = "https://www.python.org/ftp/python/3.15.0/Python-3.15.0.tgz"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "jansloot-research/0.1"})
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as out:
        while block := resp.read(1 << 20):
            out.write(block)
    tmp.rename(path)


# --- samples: deterministic recipes -------------------------------------------

def python_source():
    """Top-level Lib/*.py of the pinned CPython release, in byte order of the name, first 1 MB."""
    tgz = DOWNLOADS / CPYTHON_URL.rsplit("/", 1)[1]
    download(CPYTHON_URL, tgz)
    with tarfile.open(tgz) as tar:  # read members in memory, nothing is extracted
        members = sorted((m for m in tar.getmembers()
                          if m.isfile() and m.name.count("/") == 2 and "/Lib/" in m.name and m.name.endswith(".py")),
                         key=lambda m: m.name.encode())
        out = bytearray()
        for m in members:
            out += tar.extractfile(m).read()
            if len(out) >= MB:
                break
    return bytes(out[:MB])


def samples(enwik8):
    return {
        "enwik_1M.txt": lambda: enwik8()[:MB],
        "enwik_10M.txt": lambda: enwik8()[:10 * MB],
        "enwik_1M.txt.gz": lambda: gzip.compress(enwik8()[:MB], compresslevel=6, mtime=0),
        "pysrc_1M.txt": python_source,
        "random_1M.bin": lambda: hashlib.shake_256(b"jansloot random sample").digest(MB),
    }


# --- commands -----------------------------------------------------------------

def files():
    """(name, path) of every corpus file the manifest covers."""
    yield "enwik8", CORPUS / "enwik8"
    for path in sorted((CORPUS / "gutenberg").glob("*.txt"), key=lambda p: int(p.stem)):
        yield f"gutenberg/{path.name}", path
    for name in samples(None):
        yield f"samples/{name}", CORPUS / "samples" / name


def freeze():
    manifest = {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                for name, path in files() if path.exists()}
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(manifest)} files")


def verify(manifest, strict=False):
    """True if the corpus is usable: fixed recipes match, and (strict) every source matches too."""
    ok = drift = broken = 0
    for name, entry in manifest.items():
        path = CORPUS / name
        if not path.exists():
            problem = "MISSING "
        elif sha256(path) != entry["sha256"]:
            problem = f"CHANGED  (expected {entry['bytes']:,} bytes, got {path.stat().st_size:,})"
        else:
            ok += 1
            continue
        if name.startswith("gutenberg/") and not strict:
            print(f"warning  {name} {problem.strip()}: source drift, tables built now get a new fingerprint")
            drift += 1
        else:
            print(f"ERROR    {name} {problem.strip()}")
            broken += 1
    print(f"{ok} ok, {drift} drifted sources (warnings), {broken} errors")
    return broken == 0


def fetch(manifest, strict=False):
    enwik8_path = CORPUS / "enwik8"
    if not enwik8_path.exists():
        zip_path = DOWNLOADS / "enwik8.zip"
        download(ENWIK8_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:  # read the one expected member, nothing else
            enwik8_path.write_bytes(z.read("enwik8"))

    cache = {}

    def enwik8():
        if "data" not in cache:
            cache["data"] = enwik8_path.read_bytes()
        return cache["data"]

    (CORPUS / "samples").mkdir(parents=True, exist_ok=True)
    for name, recipe in samples(enwik8).items():
        path = CORPUS / "samples" / name
        if not path.exists():
            path.write_bytes(recipe())
            print(f"built samples/{name}")

    (CORPUS / "gutenberg").mkdir(parents=True, exist_ok=True)
    for name in manifest:
        if not name.startswith("gutenberg/"):
            continue
        path = CORPUS / name
        if path.exists():
            continue
        book_id = path.stem
        try:
            body = download_book(book_id)
        except Exception as exc:
            print(f"FAILED   {name}: {exc}")
            continue
        if body is None:
            print(f"FAILED   {name}: no longer an English book with Gutenberg markers")
        else:
            path.write_text(body, encoding="utf-8")
            print(f"fetched  {name}")
        time.sleep(1)  # be polite to Project Gutenberg
    return verify(manifest, strict)


def main():
    args = [a for a in sys.argv[1:] if a != "--strict"]
    strict = "--strict" in sys.argv[1:]
    cmd = args[0] if args else "fetch"
    if cmd == "freeze":
        freeze()
        return
    manifest = json.loads(MANIFEST.read_text())
    if cmd == "verify":
        sys.exit(0 if verify(manifest, strict) else 1)
    if cmd == "fetch":
        sys.exit(0 if fetch(manifest, strict) else 1)
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
