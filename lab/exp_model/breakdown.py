"""Where do the bits go? Ideal code length per category for one variant.

Categories: hit@k (token coded in an order-k context), esc@k (escapes paid
leaving order k), order0, spell (UNK spelling), case, wrap.

Usage: python3 lab/exp_model/breakdown.py VARIANT [SIZE]
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from model import SPELL_END, UNK, Config, State, contexts, events, render, table_for  # noqa: E402
from short_docs import snippets  # noqa: E402
from steno import split_books  # noqa: E402


def bits(low, high, total):
    return -math.log2((high - low) / total)


def cum_bits(cum, sym):
    return bits(int(cum[sym - 1]) if sym else 0, int(cum[sym]), int(cum[-1]))


def main():
    cfg = Config.parse(sys.argv[1])
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
    table = table_for(cfg)
    _, test = split_books()
    acc = defaultdict(float)
    hits = defaultdict(int)
    raw = 0
    for p in test:
        for snip in snippets(p.read_text(encoding="utf-8"), size):
            raw += len(snip.encode())
            nodes, st = {}, State()
            for tok, cls, wrapped in events(snip, cfg):
                w = table.index.get(tok, UNK)
                k = cfg.order + 1
                for syms, freqs, esc in contexts(table, nodes):
                    k -= 1
                    while k > 0 and nodes.get(k) is None:
                        k -= 1
                    if freqs is None:
                        ex = syms
                        uni = table.uni.copy()
                        uni[ex] = 0
                        acc["order0"] += -math.log2(uni[w] / uni.sum())
                        hits[0] += 1
                        break
                    total = int(freqs.sum()) + esc
                    i = int(np.searchsorted(syms, w))
                    if i < syms.size and syms[i] == w:
                        acc[f"hit@{k}"] += -math.log2(int(freqs[i]) / total)
                        hits[k] += 1
                        break
                    acc[f"esc@{k}"] += -math.log2(esc / total)
                if w == UNK:
                    p2 = p1 = SPELL_END
                    for b in list(tok.encode()) + [SPELL_END]:
                        acc["spell"] += cum_bits(table.spell_cum2(p2, p1), b)
                        p2, p1 = p1, b
                if cls is not None:
                    acc["case"] += cum_bits(table.case_cum_for(w, st), cls)
                    st.last_cls = cls
                if wrapped is not None:
                    end, rel = st.wrap_ctx(tok)
                    acc["wrap"] += cum_bits(np.cumsum(table.wrap_freq[end, rel]), wrapped)
                    st.wrap_seen(end, wrapped)
                st.emit(render(tok, cls, wrapped))
                nodes = table.advance(nodes, w)
    total = sum(acc.values())
    ntok = sum(hits.values())
    print(f"{cfg}: {size} B snippets, ideal {total / raw:.3f} bpc")
    for name in sorted(acc):
        print(f"  {name:<8}{100 * acc[name] / total:6.1f}%  {acc[name] / raw:.3f} bpc")
    print("  tokens found at order: " + ", ".join(f"{k}: {100 * hits[k] / ntok:.1f}%" for k in sorted(hits, reverse=True)))


if __name__ == "__main__":
    main()
