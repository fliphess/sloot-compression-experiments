"""Steno in overdrive: a frozen word-level n-gram table + arithmetic coding.

Text is split losslessly into tokens: a word with its leading space (" house")
or a single other character. The table is built once from training books:
  - vocabulary: every token seen at least twice; rarer tokens become ESC
  - n-gram counts: how often each token follows the previous 1 or 2 tokens
Coding a token costs -log2(p) bits (what an arithmetic coder achieves), with p
from Witten-Bell interpolation of the unigram, bigram and trigram counts. An
ESC token is followed by its spelling, coded byte by byte with an order-1
letter model.

Modes:
  static    the table is never changed: classic steno, pure lookup
  adaptive  a per-document layer on top of the frozen table learns from the
            document while coding it (the decoder does the same), so names
            and new words get cheap fast; the table itself is never modified

Usage: python3 lab/steno.py
"""
import math
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ROOT / "corpus" / "gutenberg"
TOKEN = re.compile(r" ?[A-Za-z]+|[^A-Za-z]")
ESC = 0
ORDER = 2  # max context length in tokens: 2 = trigram
LOG2 = math.log(2)


def split_books():
    paths = sorted(BOOKS.glob("*.txt"), key=lambda p: int(p.stem))
    test = paths[::10]
    train = [p for p in paths if p not in test]
    return train, test


def spell_pairs(raw):
    prev = 256
    for b in list(raw) + [256]:  # 256 = end of token
        yield prev, b
        prev = b


class Table:
    """The frozen reference table. Built once, never modified afterwards."""

    def __init__(self, train_texts):
        tokens = [t for text in train_texts for t in TOKEN.findall(text)]
        freq = Counter(tokens)
        self.vocab = {"<ESC>": ESC}
        for tok, c in freq.most_common():
            if c >= 2:
                self.vocab[tok] = len(self.vocab)
        self.bits = (len(self.vocab) + 1_000_000).bit_length()  # room for per-document words
        ids = np.array([self.vocab.get(t, ESC) for t in tokens], dtype=np.int64)

        self.N = len(ids)
        self.uni = dict(enumerate(np.bincount(ids).tolist()))
        self.ngram, self.ctx_total, self.ctx_types = {}, {}, {}
        for k in range(1, ORDER + 1):
            key = ids[k:].copy()
            for j in range(1, k + 1):
                key |= ids[k - j:len(ids) - j] << (self.bits * j)
            uniq, counts = np.unique(key, return_counts=True)
            cu, inverse = np.unique(uniq >> self.bits, return_inverse=True)
            self.ngram[k] = dict(zip(uniq.tolist(), counts.tolist()))
            self.ctx_total[k] = dict(zip(cu.tolist(), np.bincount(inverse, weights=counts).astype(np.int64).tolist()))
            self.ctx_types[k] = dict(zip(cu.tolist(), np.bincount(inverse).tolist()))

        # letter model for spelling escaped tokens, trained on the rare training tokens
        self.spell, self.spell_total = defaultdict(int), defaultdict(int)
        for tok, c in freq.items():
            if c == 1:
                for prev, b in spell_pairs(tok.encode()):
                    self.spell[prev << 9 | b] += 1
                    self.spell_total[prev] += 1
        self.spell, self.spell_total = dict(self.spell), dict(self.spell_total)

    def entries(self):
        return len(self.vocab) + sum(len(d) for d in self.ngram.values())


class Session:
    """Codes one document. Counts are frozen table + what this document taught so far."""

    def __init__(self, table, adaptive):
        self.t = table
        self.adaptive = adaptive
        self.vocab = {}  # words first seen in this document
        self.N = 0
        self.uni = defaultdict(int)
        self.ngram = {k: defaultdict(int) for k in range(1, ORDER + 1)}
        self.ctx_total = {k: defaultdict(int) for k in range(1, ORDER + 1)}
        self.ctx_types = {k: defaultdict(int) for k in range(1, ORDER + 1)}
        self.spell = defaultdict(int)
        self.spell_total = defaultdict(int)

    def lookup(self, tok):
        return self.t.vocab.get(tok, self.vocab.get(tok, ESC))

    def context(self, history, k):
        ctx = 0
        for j in range(1, k + 1):
            ctx |= history[-j] << (self.t.bits * (j - 1))
        return ctx

    def token_bits(self, w, history):
        """Bits to code token id w after history, for every model order 0..ORDER."""
        t = self.t
        p = (t.uni.get(w, 0) + self.uni[w]) / (t.N + self.N)
        out = [-math.log(p) / LOG2]
        for k in range(1, ORDER + 1):
            if len(history) >= k:
                ctx = self.context(history, k)
                total = t.ctx_total[k].get(ctx, 0) + self.ctx_total[k][ctx]
                if total:
                    types = t.ctx_types[k].get(ctx, 0) + self.ctx_types[k][ctx]
                    key = ctx << t.bits | w
                    count = t.ngram[k].get(key, 0) + self.ngram[k][key]
                    p = (count + types * p) / (total + types)
            out.append(-math.log(p) / LOG2)
        return out

    def spell_bits(self, raw):
        t = self.t
        return -sum(math.log((t.spell.get(p << 9 | b, 0) + self.spell[p << 9 | b] + 1)
                             / (t.spell_total.get(p, 0) + self.spell_total[p] + 257))
                    for p, b in spell_pairs(raw)) / LOG2

    def learn(self, tok, w, history):
        """After coding: update the document layer (decoder does exactly the same)."""
        self.uni[w] += 1
        self.N += 1
        for k in range(1, ORDER + 1):
            if len(history) >= k:
                ctx = self.context(history, k)
                key = ctx << self.t.bits | w
                if self.t.ngram[k].get(key, 0) + self.ngram[k][key] == 0:
                    self.ctx_types[k][ctx] += 1
                self.ngram[k][key] += 1
                self.ctx_total[k][ctx] += 1
        if w == ESC:
            for prev, b in spell_pairs(tok.encode()):
                self.spell[prev << 9 | b] += 1
                self.spell_total[prev] += 1
            new = len(self.t.vocab) + len(self.vocab)
            self.vocab[tok] = new
            self.uni[new] += 1
            self.N += 1


def encode_cost(table, text, adaptive):
    """Total bits per model order, bits spent spelling escapes, number of escapes."""
    session = Session(table, adaptive)
    bits = [0.0] * (ORDER + 1)
    spell = 0.0
    escapes = 0
    history = []
    for tok in TOKEN.findall(text):
        w = session.lookup(tok)
        for k, b in enumerate(session.token_bits(w, history)):
            bits[k] += b
        if w == ESC:
            escapes += 1
            spell += session.spell_bits(tok.encode())
        if adaptive:
            session.learn(tok, w, history)
        history = (history + [w])[-ORDER:]
    return [b + spell for b in bits], spell, escapes


def compressed_size(cmd, data):
    # zstd --patch-from needs a real input file (it reads the size up front)
    with tempfile.NamedTemporaryFile() as f:
        f.write(data)
        f.flush()
        return len(subprocess.run(cmd + [f.name], capture_output=True, check=True).stdout)


def main():
    train, test = split_books()
    train_texts = [p.read_text(encoding="utf-8") for p in train]
    print(f"train: {len(train)} books, {sum(map(len, train_texts)) / 1e6:.0f}M chars; test: {len(test)} unseen books")
    table = Table(train_texts)
    print(f"table: {len(table.vocab):,} words, {table.entries():,} entries total\n")

    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "train.txt"
        ref.write_text("".join(train_texts), encoding="utf-8")
        baselines = {
            "xz -9e": ["xz", "-9e", "-c"],
            "zstd -19": ["zstd", "-19", "-c", "-q"],
            "zstd +ref": ["zstd", "-19", "--long=31", f"--patch-from={ref}", "-c", "-q"],
        }
        total_bytes = 0
        sums = defaultdict(float)
        cols = list(baselines) + [f"static o{k}" for k in range(ORDER + 1)] + [f"adapt o{k}" for k in range(ORDER + 1)]
        print(f"{'book':<8}{'bytes':>10}" + "".join(f"{c:>11}" for c in cols) + f"{'esc%':>7}{'spell%':>8}")
        for path in test:
            text = path.read_text(encoding="utf-8")
            data = text.encode()
            total_bytes += len(data)
            row = {name: 8 * compressed_size(cmd, data) for name, cmd in baselines.items()}
            static, _, _ = encode_cost(table, text, adaptive=False)
            adaptive, spell, escapes = encode_cost(table, text, adaptive=True)
            for k in range(ORDER + 1):
                row[f"static o{k}"] = static[k]
                row[f"adapt o{k}"] = adaptive[k]
            for c in cols:
                sums[c] += row[c]
            ntok = len(TOKEN.findall(text))
            print(f"{path.stem:<8}{len(data):>10,}" + "".join(f"{row[c] / len(data):>11.3f}" for c in cols)
                  + f"{100 * escapes / ntok:>6.1f}%{100 * spell / adaptive[-1]:>7.1f}%", flush=True)
        print(f"{'TOTAL':<8}{total_bytes:>10,}" + "".join(f"{sums[c] / total_bytes:>11.3f}" for c in cols))
    print("\nbpc = bits per byte; o0 = single words, o1 = +1 word context, o2 = +2 words context")
    print("esc% = share of tokens not in the table; spell% = share of adaptive o2 bits spent spelling them")


if __name__ == "__main__":
    main()
