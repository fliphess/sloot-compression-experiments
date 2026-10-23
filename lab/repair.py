"""v0: recursive pattern replacement (Re-Pair style grammar compression).

Each round counts every adjacent symbol pair, picks the most frequent pairs,
gives each a new symbol in the table and replaces all occurrences. Repeat
until no pair occurs twice. Symbols 0..255 are bytes, 256+ are table rules.

Per round we report two honest costs of "table + sequence":
  fixed    every symbol uses ceil(log2(alphabet)) bits -- the literal
           "replace with one reference" idea
  adaptive adaptive order-0 arithmetic coding (Laplace estimator); an
           achievable code length, not just a theoretical bound

Usage: python3 lab/repair.py FILE [--batch N]
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np


def adaptive_bits(symbols, alphabet):
    """Exact code length of adaptive Laplace arithmetic coding over `alphabet`."""
    n = len(symbols)
    if n == 0:
        return 0.0
    counts = np.bincount(symbols)
    counts = counts[counts > 0]
    nats = (math.lgamma(n + alphabet) - math.lgamma(alphabet)
            - sum(math.lgamma(c + 1) for c in counts.tolist()))
    return nats / math.log(2)


def choose_pairs(seq, batch):
    """Most frequent pairs (count >= 2) whose symbols don't overlap each other."""
    keys = (seq[:-1].astype(np.int64) << 32) | seq[1:].astype(np.int64)
    uniq, counts = np.unique(keys, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    chosen, used = [], set()
    for i in order:
        if counts[i] < 2 or len(chosen) >= batch:
            break
        a, b = int(uniq[i] >> 32), int(uniq[i] & 0xFFFFFFFF)
        if a in used or b in used:
            continue
        used.update((a, b))
        chosen.append((a, b))
    return keys, chosen


def replace_pairs(seq, keys, chosen, first_symbol):
    """Replace every non-overlapping occurrence of the chosen pairs."""
    chosen_keys = np.array([(a << 32) | b for a, b in chosen], dtype=np.int64)
    sort = np.argsort(chosen_keys)
    chosen_keys = chosen_keys[sort]
    match = np.isin(keys, chosen_keys)
    # Pairs share no symbols, so matches only overlap in runs like "aaaa" of a
    # self-pair (a, a): keep every other match, counted from the start of the run.
    idx = np.arange(len(match))
    starts = match & ~np.concatenate(([False], match[:-1]))
    run_start = np.maximum.accumulate(np.where(starts, idx, 0))
    keep = np.flatnonzero(match & ((idx - run_start) % 2 == 0))
    out = seq.copy()
    out[keep] = first_symbol + sort[np.searchsorted(chosen_keys, keys[keep])]
    alive = np.ones(len(seq), dtype=bool)
    alive[keep + 1] = False
    return out[alive]


def compress(data, batch, report):
    seq = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
    rules = []  # rules[k] = (a, b) defines symbol 256 + k
    rnd = 0
    while len(seq) > 1:
        keys, chosen = choose_pairs(seq, batch)
        if not chosen:
            break
        seq = replace_pairs(seq, keys, chosen, 256 + len(rules))
        rules.extend(chosen)
        rnd += 1
        report(rnd, seq, rules)
    return seq, rules


def decompress(seq, rules):
    expansion = [bytes([i]) for i in range(256)]
    for a, b in rules:
        expansion.append(expansion[a] + expansion[b])
    return b"".join(expansion[s] for s in seq.tolist())


def costs(seq, rules, split=False):
    alphabet = 256 + len(rules)
    width = math.ceil(math.log2(alphabet))
    rule_stream = np.array(rules, dtype=np.int64).ravel()
    fixed = (len(seq) + 2 * len(rules)) * width
    table, sequence = adaptive_bits(rule_stream, alphabet), adaptive_bits(seq, alphabet)
    if split:
        return table, sequence
    return fixed, table + sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--batch", type=int, default=1000, help="max new rules per round")
    args = ap.parse_args()
    data = Path(args.file).read_bytes()
    n = len(data)
    best = {"fixed": (math.inf, 0), "adaptive": (math.inf, 0)}

    print(f"{args.file}: {n} bytes")
    print(f"{'round':>6}{'rules':>9}{'sequence':>11}{'fixed bpc':>11}{'adaptive bpc':>14}")

    def report(rnd, seq, rules):
        fixed, adaptive = costs(seq, rules)
        for name, bits in (("fixed", fixed), ("adaptive", adaptive)):
            if bits < best[name][0]:
                best[name] = (bits, rnd)
        if adaptive <= best["adaptive"][0]:
            best["split"] = (len(rules), *costs(seq, rules, split=True))
        if rnd in (1, 2, 5) or rnd % 10 == 0 and math.log10(rnd).is_integer() or rnd % 500 == 0 or rnd in (20, 50, 200):
            print(f"{rnd:>6}{len(rules):>9}{len(seq):>11}{fixed / n:>11.3f}{adaptive / n:>14.3f}")

    t0 = time.time()
    seq, rules = compress(data, args.batch, report)
    elapsed = time.time() - t0
    fixed, adaptive = costs(seq, rules)
    print(f"{'final':>6}{len(rules):>9}{len(seq):>11}{fixed / n:>11.3f}{adaptive / n:>14.3f}")
    assert decompress(seq, rules) == data, "round trip failed"
    print(f"\nround trip OK, {elapsed:.1f}s")
    rules_at_best, table, sequence = best.pop("split")
    for name, (bits, rnd) in best.items():
        print(f"best {name:<9} {bits / n:.3f} bpc = {bits / 8:,.0f} bytes (round {rnd})")
    print(f"  at best adaptive: table {table / 8:,.0f} bytes ({rules_at_best} rules), "
          f"sequence {sequence / 8:,.0f} bytes")


if __name__ == "__main__":
    main()
