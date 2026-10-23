"""E8c: build codec-v1 tables from large corpora, in chunks and in parallel.

Produces exactly the array format (and, for the same input, the same arrays
and fingerprint) as codec.FrozenTable.build, without holding all tokens in
memory:
  pass 1  token frequencies, per group of books in worker processes
  pass 2  token ids per book (workers, in order), then n-gram counts per
          chunk of ~CHUNK tokens, merged into sorted accumulators; n-grams
          crossing book boundaries are counted, as in FrozenTable.build

Training text order: the 90 codec-v1 training books (steno.split_books), then
the books of lab/exp_table/manifest.json in order, minus its "excluded" list
(from leakage.py), until --mb is reached. Run leakage.py and manifest.py first. Tiers are
therefore nested: a bigger tier contains every smaller one.

--prune N drops order-1/2 n-grams seen fewer than N times (0 = keep all).

Memory (external build): token ids are produced in bounded batches of books;
each chunk's n-gram counts are written to disk as sorted runs, split into
PARTS key ranges (by the oldest word of the n-gram, balanced by token mass).
Each range is then merged on its own, and the final arrays are assembled in
memory-mapped files under tables/exp_table_tmp_*/ (removed afterwards).
Concatenated in order, the ranges are the globally sorted arrays.

Usage: python3 lab/exp_table/build_table.py --mb 300 --out tables/exp_table_300mb.npz
"""
import argparse
import hashlib
import json
import os
import resource
import shutil
import sys
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "lab"))
sys.path.insert(0, str(HERE))
from codec import MAX_TOTAL, SPELL_END, UNK, FrozenTable  # noqa: E402
from steno import TOKEN, split_books  # noqa: E402

BIG = ROOT / "corpus" / "gutenberg_big"
CHUNK = 20_000_000
BATCH = 64  # books per pass-2 batch, bounds buffered results
PARTS = 32
MAX_BITS = 21  # 3 ids of 21 bits fit in one int64 key

_vocab = None


def training_files(mb):
    train, _ = split_books()
    files = list(train)
    manifest = json.loads((HERE / "manifest.json").read_text())
    excluded = {e["id"] for e in manifest["excluded"]}
    for book in manifest["books"]:
        if book["id"] not in excluded:
            files.append(BIG / f"{book['id']}.txt")
    out, total = [], 0
    for path in files:
        if total >= mb * 1e6:
            break
        out.append(path)
        total += path.stat().st_size
    return out, total


def count_group(paths):
    c = Counter()
    for p in paths:
        c.update(TOKEN.findall(p.read_text(encoding="utf-8")))
    return c


def ids_for(path):
    get = _vocab.get
    return np.fromiter((get(t, UNK) for t in TOKEN.findall(path.read_text(encoding="utf-8"))), dtype=np.int32)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS: bytes


def merge_into(acc, keys, counts):
    """Add sorted-unique (keys, counts) into the sorted-unique accumulator."""
    if acc is None:
        return keys, counts.astype(np.uint32)
    akeys, acounts = acc
    idx = np.searchsorted(akeys, keys)
    found = idx < len(akeys)
    found[found] = akeys[idx[found]] == keys[found]
    np.add.at(acounts, idx[found], counts[found].astype(np.uint32))
    new = ~found
    return np.insert(akeys, idx[new], keys[new]), np.insert(acounts, idx[new], counts[new].astype(np.uint32))


def main():
    global _vocab
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--prune", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    files, total_bytes = training_files(args.mb)
    print(f"{len(files)} books, {total_bytes / 1e6:.0f} MB", flush=True)
    ctx = get_context("fork")

    groups = [files[i::30] for i in range(30)]
    freq = Counter()
    with ctx.Pool(args.procs) as pool:
        for c in pool.imap_unordered(count_group, groups):
            freq.update(c)
    kept = sorted((t for t, c in freq.items() if c >= 2), key=lambda t: (-freq[t], t.encode()))
    vocab = ["<UNK>"] + kept
    _vocab = {t: i for i, t in enumerate(vocab)}
    b = len(vocab).bit_length()
    assert b <= MAX_BITS, f"vocab {len(vocab)} needs {b} bits"
    print(f"pass 1: {len(vocab):,} words, {time.time() - t0:.0f}s, peak {rss_gb():.1f} GB", flush=True)

    spell = np.zeros((257, 257), dtype=np.uint32)
    rows, cols = [], []
    for tok, c in freq.items():
        if c == 1:
            seq = [SPELL_END] + list(tok.encode()) + [SPELL_END]
            rows.extend(seq[:-1])
            cols.extend(seq[1:])
    np.add.at(spell, (np.array(rows), np.array(cols)), 1)
    mass = np.array([0] + [freq[t] for t in kept], dtype=np.int64)
    mass[UNK] = sum(1 for c in freq.values() if c == 1)
    del freq, rows, cols
    cum = np.cumsum(mass)
    bounds = np.unique(np.searchsorted(cum, cum[-1] * np.arange(1, PARTS) / PARTS))
    key_bounds = {k: bounds.astype(np.int64) << (b * k) for k in (1, 2)}

    tmp = ROOT / "tables" / f"exp_table_tmp_{Path(args.out).stem}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    nparts = len(bounds) + 1
    runs = {k: [[] for _ in range(nparts)] for k in (1, 2)}
    uni = np.zeros(len(vocab), dtype=np.int64)
    carry = np.empty(0, dtype=np.int64)
    pending, pending_len, run_no = [], 0, 0

    def flush():
        """Count one chunk's n-grams and write them to disk as sorted runs, one per key range."""
        nonlocal carry, pending, pending_len, run_no
        ids = np.concatenate([carry] + pending)
        uni[:] += np.bincount(ids[len(carry):], minlength=len(vocab))
        for k in (1, 2):
            first = max(len(carry), k)  # first new token that has k predecessors
            key = ids[first:].copy()
            for j in range(1, k + 1):
                key |= ids[first - j:len(ids) - j] << (b * j)
            keys, counts = np.unique(key, return_counts=True)
            del key
            edges = [0, *np.searchsorted(keys, key_bounds[k]).tolist(), len(keys)]
            for p in range(nparts):
                lo, hi = edges[p], edges[p + 1]
                if hi > lo:
                    path = tmp / f"run_k{k}_p{p:02d}_{run_no:05d}"
                    np.save(f"{path}_keys.npy", keys[lo:hi])
                    np.save(f"{path}_cnts.npy", counts[lo:hi].astype(np.uint32))
                    runs[k][p].append(path)
        carry = ids[-2:]
        pending, pending_len = [], 0
        run_no += 1

    with ctx.Pool(args.procs) as pool:
        for i in range(0, len(files), BATCH):
            for arr in pool.map(ids_for, files[i:i + BATCH], chunksize=4):
                pending.append(arr.astype(np.int64))
                pending_len += len(arr)
            if pending_len >= CHUNK:
                flush()
                print(f"  {int(uni.sum()):,} tokens, {run_no} runs, {time.time() - t0:.0f}s, "
                      f"peak {rss_gb():.1f} GB", flush=True)
        if pending:
            flush()
    n_tokens = int(uni.sum())
    assert n_tokens < MAX_TOTAL, f"{n_tokens} tokens exceed the coder limit"
    print(f"pass 2: {n_tokens:,} tokens, {time.time() - t0:.0f}s, peak {rss_gb():.1f} GB", flush=True)

    # merge: one key range at a time; contexts never straddle a range (the range key is the
    # context's oldest word), so each range yields a contiguous piece of the final arrays
    pruned = {1: 0, 2: 0}
    pieces = {k: [] for k in (1, 2)}
    for k in (1, 2):
        offset = 0
        for p in range(nparts):
            if not runs[k][p]:
                continue
            keys = np.concatenate([np.load(f"{r}_keys.npy") for r in runs[k][p]])
            counts = np.concatenate([np.load(f"{r}_cnts.npy") for r in runs[k][p]])
            for r in runs[k][p]:
                Path(f"{r}_keys.npy").unlink()
                Path(f"{r}_cnts.npy").unlink()
            order = np.argsort(keys, kind="stable")
            keys, counts = keys[order], counts[order]
            del order
            first = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
            counts = np.add.reduceat(counts, first)  # uint32: every total stays < 2^30
            keys = keys[first]
            if args.prune:
                keep = counts >= args.prune
                pruned[k] += int((~keep).sum())
                keys, counts = keys[keep], counts[keep]
            if len(keys) == 0:
                continue
            ctxkeys = keys >> b  # sorted, because keys are
            cfirst = np.flatnonzero(np.concatenate(([True], ctxkeys[1:] != ctxkeys[:-1])))
            base = tmp / f"piece_k{k}_p{p:02d}"
            np.save(f"{base}_ctx.npy", ctxkeys[cfirst])
            np.save(f"{base}_start.npy", (cfirst + offset).astype(np.uint32))
            np.save(f"{base}_sym.npy", (keys & ((1 << b) - 1)).astype(np.uint32))
            np.save(f"{base}_cnt.npy", counts.astype(np.uint32))
            pieces[k].append(base)
            offset += len(keys)
            del keys, counts, ctxkeys, cfirst
        print(f"  merged order {k}: {offset:,} entries, peak {rss_gb():.1f} GB", flush=True)

    arrays = {
        "vocab_blob": np.frombuffer(b"".join(t.encode() for t in vocab), dtype=np.uint8),
        "vocab_lens": np.array([len(t.encode()) for t in vocab], dtype=np.uint32),
        "uni": uni.astype(np.uint32),
        "spell": spell,
    }
    for k in (1, 2):
        for part, extra in (("ctx", 0), ("start", 1), ("sym", 0), ("cnt", 0)):
            files_k = [f"{base}_{part}.npy" for base in pieces[k]]
            lengths = [np.load(f, mmap_mode="r").shape[0] for f in files_k]
            first = np.load(files_k[0], mmap_mode="r")
            dest = np.lib.format.open_memmap(tmp / f"final_c{k}_{part}.npy", mode="w+", dtype=first.dtype,
                                             shape=(sum(lengths) + extra,))
            pos = 0
            for f, n in zip(files_k, lengths):
                dest[pos:pos + n] = np.load(f, mmap_mode="r")
                pos += n
                Path(f).unlink()
            arrays[f"c{k}_{part}"] = dest
        # the closing start offset = number of entries; set it now that sym is complete
        arrays[f"c{k}_start"][-1] = len(arrays[f"c{k}_sym"])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.parent / f".partial_{out.name}"  # readers never see a half-written table
    np.savez_compressed(partial, **arrays)
    os.replace(partial, out)
    fingerprint = fingerprint_of(arrays)
    entries = {k: len(arrays[f"c{k}_sym"]) for k in (1, 2)}
    del arrays, dest
    shutil.rmtree(tmp)
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS reports bytes
    meta = {
        "table": args.out, "books": len(files), "train_mb": round(total_bytes / 1e6, 1),
        "tokens": n_tokens, "vocab": len(vocab), "bits": b,
        "entries_o1": entries[1], "entries_o2": entries[2],
        "prune": args.prune, "pruned_o1": pruned[1], "pruned_o2": pruned[2],
        "disk_mb": round(out.stat().st_size / 1e6, 1), "build_s": round(time.time() - t0),
        "peak_rss_gb_main": round(peak_gb, 2), "fingerprint": fingerprint.hex(),
        "book_ids": [int(p.stem) for p in files],
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / (Path(args.out).stem + ".build.json")).write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "book_ids"}))


def fingerprint_of(arrays, step=16_000_000):
    """Same value as FrozenTable's fingerprint, computed in slices to bound memory."""
    digest = hashlib.sha256()
    for name in FrozenTable.ARRAYS:
        digest.update(name.encode())
        flat = np.asarray(arrays[name]).reshape(-1)
        for i in range(0, len(flat), step):
            digest.update(np.ascontiguousarray(flat[i:i + step]).astype("<i8").tobytes())
    return digest.digest()[:4]


if __name__ == "__main__":
    main()
