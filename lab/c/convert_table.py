"""E11: convert a codec-v1 FrozenTable (.npz) into a flat little-endian file that C can mmap.

Format "STNCTAB1" (all integers little-endian, every section 64-byte aligned):

  header, 512 bytes
    0   char[8]  magic "STNCTAB1"
    8   u32      format version (1)
    12  u32      header size (512)
    16  u8[4]    table fingerprint (= Python FrozenTable.fingerprint), then 4 zero bytes
    24  u64      V            vocab size incl. UNK
    32  u64      b            bit length of V (context key = h[-1] | h[-2] << b)
    40  u64      total0       sum of uni
    48  u64      n_ctx1       number of order-1 contexts
    56  u64      n_ent1       number of order-1 (context, symbol) entries
    64  u64      n_ctx2
    72  u64      n_ent2
    80  u64      blob_len     bytes in vocab_blob
    88  u64      hash_slots   size of the vocab hash (power of two)
    96  u64      n_sections (22)
    104 u64      flags: bit 0 = "nested": every order-2 follower (a, b) -> s is also an order-1 follower
                 b -> s (always true for tables built from one token stream; the converter checks it)
    112 u64      c2_hash_bits
    128 { u64 offset, u64 bytes } x n_sections, in the order of SECTIONS below

  sections (name, element type, count); "helper" = derived, not part of the fingerprint
    vocab_blob  u8   blob_len    token bytes, id order (id 0 = "<UNK>")
    vocab_lens  u32  V
    vocab_off   u64  V+1         helper: byte offset of token i in vocab_blob
    uni         u32  V
    uni_cum     u64  V           helper: inclusive prefix sum of uni
    spell       u32  257*257     row-major [prev][next]
    spell_cum   u32  257*257     helper: per row, inclusive prefix sum of (spell + 1)
    c1_ctx      i64  n_ctx1      sorted context keys
    c1_start    u32  n_ctx1+1
    c1_sym      u32  n_ent1      ascending within a context
    c1_cnt      u32  n_ent1
    c1_cum      u32  n_ent1      helper: inclusive prefix sum of cnt within each context
    c2_ctx .. c2_cum             same for order 2
    vocab_hash  u32  hash_slots  helper: open addressing, slot = FNV-1a-64(token bytes) & (slots-1),
                                 linear probing, value = token id (0 = empty; UNK is not inserted)
    c1_unicum   u32  n_ent1      helper: inclusive prefix sum of uni[c1_sym] within each order-1 context
                                 (order-0 exclusion of a whole order-1 context in O(log n))
    c2_c1cum    u32  n_ent2      helper: for order-2 context (a, b), inclusive prefix sum over its followers s
                                 of the order-1 count of s in context b (0 if absent). Lets the order-1 level
                                 exclude a whole order-2 context in O(log n) instead of O(n).
    c1_index    u32  V           helper: order-1 context index + 1 of token id i (0 = no such context)
    c2_hash     u32  2^bits + overflow   helper: order-2 context index + 1 by key (0 = empty);
                                 home slot = (key * 0x9E3779B97F4A7C15 mod 2^64) >> (64 - bits), linear probing
                                 upward without wrap-around (the overflow tail holds the spill); a lookup
                                 scans from the home slot to the first empty slot, comparing c2_ctx[index]

The fingerprint is computed here exactly as FrozenTable does (SHA-256 over name + values as <i8 for the 12
fingerprint arrays, first 4 bytes), in slices to bound memory; `steno verify` recomputes it in C.

Usage: python3 lab/c/convert_table.py tables/en-gutenberg-v1.npz tables/c_en-gutenberg-v1.bin
"""
import hashlib
import struct
import sys
import time
from pathlib import Path

import numpy as np

FP_ARRAYS = ["vocab_blob", "vocab_lens", "uni", "spell",
             "c1_ctx", "c1_start", "c1_sym", "c1_cnt",
             "c2_ctx", "c2_start", "c2_sym", "c2_cnt"]
SECTIONS = ["vocab_blob", "vocab_lens", "vocab_off", "uni", "uni_cum", "spell", "spell_cum",
            "c1_ctx", "c1_start", "c1_sym", "c1_cnt", "c1_cum",
            "c2_ctx", "c2_start", "c2_sym", "c2_cnt", "c2_cum", "vocab_hash", "c1_unicum", "c2_c1cum", "c1_index", "c2_hash"]
DTYPES = {"vocab_blob": "<u1", "vocab_lens": "<u4", "vocab_off": "<u8", "uni": "<u4", "uni_cum": "<u8",
          "spell": "<u4", "spell_cum": "<u4", "vocab_hash": "<u4", "c1_unicum": "<u4", "c2_c1cum": "<u4", "c1_index": "<u4", "c2_hash": "<u4"}
for _k in (1, 2):
    DTYPES.update({f"c{_k}_ctx": "<i8", f"c{_k}_start": "<u4", f"c{_k}_sym": "<u4",
                   f"c{_k}_cnt": "<u4", f"c{_k}_cum": "<u4"})
HEADER = 512
STEP = 16_000_000
MAX_TOTAL = 1 << 30


def fnv1a64(data):
    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.f.write(b"\0" * HEADER)
        self.sections = {}

    def write(self, name, arr):
        dtype = np.dtype(DTYPES[name])
        pos = self.f.tell()
        pad = -pos % 64
        self.f.write(b"\0" * pad)
        start = self.f.tell()
        flat = np.asarray(arr).reshape(-1)
        for i in range(0, len(flat), STEP):
            part = flat[i:i + STEP]
            conv = part.astype(dtype)
            assert np.array_equal(conv.astype(np.int64), part.astype(np.int64)), f"{name} overflows {dtype}"
            self.f.write(conv.tobytes())
        self.sections[name] = (start, self.f.tell() - start)


def digest_update(digest, name, arr):
    digest.update(name.encode())
    flat = np.asarray(arr).reshape(-1)
    for i in range(0, len(flat), STEP):
        digest.update(np.ascontiguousarray(flat[i:i + STEP]).astype("<i8").tobytes())


def within_context_cum(start, cnt):
    """Inclusive prefix sum of cnt inside each context [start[i], start[i+1])."""
    start = start.astype(np.int64)
    n = len(cnt)
    totals = np.add.reduceat(cnt.astype(np.int64), start[:-1]) if n else np.zeros(0, np.int64)
    totals[np.diff(start) == 0] = 0  # reduceat quirk for empty groups (none expected)
    assert totals.size == 0 or int(totals.max()) < MAX_TOTAL
    base = np.concatenate(([0], np.cumsum(totals)[:-1]))  # global exclusive sum at each context start
    out = np.empty(n, dtype=np.uint32)
    carry = 0
    for i in range(0, n, STEP):
        c = cnt[i:i + STEP].astype(np.int64)
        g = carry + np.cumsum(c)
        carry = int(g[-1])
        ctx = np.searchsorted(start, np.arange(i, i + len(c)), side="right") - 1
        out[i:i + len(c)] = (g - base[ctx]).astype(np.uint32)
    return out


def check_sorted_within(start, sym):
    boundary = np.zeros(len(sym) + 1, dtype=bool)
    boundary[start.astype(np.int64)] = True
    for i in range(0, len(sym), STEP):
        part = sym[i:i + STEP + 1].astype(np.int64)
        bad = np.diff(part) <= 0
        assert not np.any(bad & ~boundary[i + 1:i + len(part)]), "symbols not ascending within a context"


def order2_counts_in_order1(c1, c2_ctx, c2_start, c2_sym, b):
    """Order-1 count of each order-2 follower in the order-1 context of the context's last token."""
    c1_ctx, c1_start, c1_sym, c1_cnt = c1
    c1_owner = np.repeat(c1_ctx, np.diff(c1_start))
    key1 = (c1_owner << b) | c1_sym.astype(np.int64)  # globally ascending
    del c1_owner
    assert np.all(np.diff(key1) > 0)
    mask = (1 << b) - 1
    n = len(c2_sym)
    out = np.empty(n, dtype=np.uint32)
    found_total = 0
    starts = c2_start.astype(np.int64)
    for i in range(0, n, STEP):
        idx = np.arange(i, min(n, i + STEP))
        owner = np.searchsorted(starts, idx, side="right") - 1
        last = c2_ctx[owner] & mask
        key = (last << b) | c2_sym[idx].astype(np.int64)
        g = np.searchsorted(key1, key)
        gc = np.minimum(g, len(key1) - 1)
        found = key1[gc] == key
        out[idx] = np.where(found, c1_cnt[gc], 0).astype(np.uint32)
        found_total += int(found.sum())
    return out, found_total / max(1, n)


def context_hash(keys):
    """Linear-probing hash of the sorted context keys, built in one vectorized pass (see format)."""
    n = len(keys)
    bits = max(4, (2 * n - 1).bit_length())
    home = ((keys.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(64 - bits)).astype(np.int64)
    order = np.argsort(home, kind="stable")
    h = home[order]
    i = np.arange(n, dtype=np.int64)
    slot = i + np.maximum.accumulate(h - i)  # slot_i = max(h_i, slot_{i-1} + 1)
    table = np.zeros(max(1 << bits, int(slot[-1]) + 2 if n else 0), dtype=np.uint32)
    table[slot] = (order + 1).astype(np.uint32)
    return table, bits


def convert(src, dst):
    t0 = time.time()
    digest = hashlib.sha256()
    w = Writer(dst)
    meta = {}
    with np.load(src) as data:
        blob = data["vocab_blob"]
        lens = data["vocab_lens"]
        V = len(lens)
        b = V.bit_length()
        digest_update(digest, "vocab_blob", blob)
        digest_update(digest, "vocab_lens", lens)
        w.write("vocab_blob", blob)
        w.write("vocab_lens", lens)
        off = np.concatenate(([0], np.cumsum(lens, dtype=np.int64)))
        w.write("vocab_off", off)
        slots = 1 << max(4, (2 * V - 1).bit_length())
        table = np.zeros(slots, dtype=np.uint32)
        raw = blob.tobytes()
        offl = off.tolist()
        mask = slots - 1
        for i in range(1, V):
            s = fnv1a64(raw[offl[i]:offl[i + 1]]) & mask
            while table[s]:
                s = (s + 1) & mask
            table[s] = i
        meta.update(V=V, b=b, blob_len=len(blob), hash_slots=slots)
        del blob, raw, offl
        print(f"  vocab {V:,} ({time.time() - t0:.0f}s)", flush=True)

        uni = data["uni"]
        digest_update(digest, "uni", uni)
        w.write("uni", uni)
        uni_cum = np.cumsum(uni, dtype=np.int64)
        meta["total0"] = int(uni_cum[-1])
        assert meta["total0"] < MAX_TOTAL
        w.write("uni_cum", uni_cum)
        del uni_cum

        spell = data["spell"]
        assert spell.shape == (257, 257)
        digest_update(digest, "spell", spell)
        w.write("spell", spell)
        spell_cum = np.cumsum(spell.astype(np.int64) + 1, axis=1)
        assert int(spell_cum[:, -1].max()) < MAX_TOTAL
        w.write("spell_cum", spell_cum)
        del spell, spell_cum

        for k in (1, 2):
            ctx = data[f"c{k}_ctx"]
            assert ctx.dtype == np.int64 and np.all(np.diff(ctx) > 0), "contexts not ascending"
            digest_update(digest, f"c{k}_ctx", ctx)
            w.write(f"c{k}_ctx", ctx)
            meta[f"n_ctx{k}"] = len(ctx)
            if k == 1:
                index = np.zeros(V, dtype=np.uint32)
                index[ctx] = np.arange(1, len(ctx) + 1, dtype=np.uint32)
                w.write("c1_index", index)
                del index
            else:
                table2, meta["c2_hash_bits"] = context_hash(ctx)
                w.write("c2_hash", table2)
                del table2
            del ctx
            start = data[f"c{k}_start"]
            assert start[0] == 0 and np.all(np.diff(start.astype(np.int64)) > 0)
            digest_update(digest, f"c{k}_start", start)
            w.write(f"c{k}_start", start)
            sym = data[f"c{k}_sym"]
            assert int(start[-1]) == len(sym)
            assert len(sym) == 0 or int(sym.max()) < V
            check_sorted_within(start, sym)
            digest_update(digest, f"c{k}_sym", sym)
            w.write(f"c{k}_sym", sym)
            meta[f"n_ent{k}"] = len(sym)
            del sym
            cnt = data[f"c{k}_cnt"]
            assert len(cnt) == 0 or int(cnt.min()) >= 1
            digest_update(digest, f"c{k}_cnt", cnt)
            w.write(f"c{k}_cnt", cnt)
            w.write(f"c{k}_cum", within_context_cum(start, cnt))
            del cnt
            if k == 1:
                sym = data["c1_sym"]
                w.write("c1_unicum", within_context_cum(start, uni[sym]))
                c1 = (data["c1_ctx"], start.astype(np.int64), sym, data["c1_cnt"])
                del sym
            else:
                c1cnt, nested = order2_counts_in_order1(c1, data["c2_ctx"], start, data["c2_sym"], b)
                w.write("c2_c1cum", within_context_cum(start, c1cnt))
                meta["nested"] = nested
                print(f"  order-2 entries present in their order-1 context: {nested:.6%}", flush=True)
                del c1cnt, c1
            del start
            print(f"  order {k} done ({time.time() - t0:.0f}s)", flush=True)
        w.write("vocab_hash", table)
        del uni

    fp = digest.digest()[:4]
    hdr = bytearray(HEADER)
    hdr[0:8] = b"STNCTAB1"
    struct.pack_into("<II", hdr, 8, 1, HEADER)
    hdr[16:20] = fp
    struct.pack_into("<QQ", hdr, 104, 1 if meta["nested"] == 1.0 else 0, meta["c2_hash_bits"])
    struct.pack_into("<10Q", hdr, 24, meta["V"], meta["b"], meta["total0"], meta["n_ctx1"], meta["n_ent1"],
                     meta["n_ctx2"], meta["n_ent2"], meta["blob_len"], meta["hash_slots"], len(SECTIONS))
    for i, name in enumerate(SECTIONS):
        struct.pack_into("<QQ", hdr, 128 + 16 * i, *w.sections[name])
    w.f.seek(0)
    w.f.write(bytes(hdr))
    w.f.close()
    print(f"{src} -> {dst}: fingerprint {fp.hex()}, {Path(dst).stat().st_size / 1e6:.1f} MB, "
          f"{time.time() - t0:.0f}s", flush=True)
    return fp


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
