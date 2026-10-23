"""Per-token ideal bits of the E8b adaptive n-gram codec (d3-sum-w32) on the 200 short snippets.

Writes results/table_tokens.json: for every snippet, a list of [byte offset at
the end of the token, cumulative ideal bits so far]. mix.py uses it to switch
between the table and the neural model at token boundaries.

Usage: python3 lab/exp_tinymodel/table_bits.py   (system Python, CPU only)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "exp_adaptive")]

import run as e8b  # noqa: E402
from adaptive import UNK, DocState, Encoder, encode_spelling, encode_token  # noqa: E402
from codec import FrozenTable  # noqa: E402
from short_docs import SIZES, snippets  # noqa: E402
from steno import TOKEN, split_books  # noqa: E402


def token_bits(table, cfg, text):
    st, enc, hist, out, pos = DocState(table, cfg), Encoder(), [], [], 0
    for tok in TOKEN.findall(text):
        w = st.lookup(tok)
        encode_token(enc, st, w, hist)
        if w == UNK:
            encode_spelling(enc, st, tok.encode())
        hist = (hist + [st.learn(tok, w, hist)])[-3:]
        pos += len(tok.encode())
        out.append([pos, enc.ideal])
    return out


def main():
    table = FrozenTable.load()
    cfg = e8b.make_configs()["d3-sum-w32"]
    _, test = split_books()
    texts = [p.read_text(encoding="utf-8") for p in test]
    res = {str(size): [token_bits(table, cfg, s) for text in texts for s in snippets(text, size)] for size in SIZES}
    (HERE / "results" / "table_tokens.json").write_text(json.dumps(res))
    for size, docs in res.items():
        print(size, f"{sum(d[-1][1] for d in docs) / 8 / len(docs):.0f} B ideal avg")


if __name__ == "__main__":
    main()
