"""E10: tiny byte-level transformer, shared data, model and evaluation code.

Tokenisation: raw UTF-8 bytes (256 symbols) plus BOS = 256 at the start of every
document. No OOV, no tokenizer to ship. The output head is tied to the input
embedding, so all parameters are in the counts below.

Evaluation = ideal code length sum(-log2 p(byte)) with the context reset per
document; documents longer than the window use a sliding window (512 bytes,
stride 256), so every byte after the first window sees >= 256 bytes of context.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LAB = Path(__file__).resolve().parent.parent
ROOT = LAB.parent
sys.path.insert(0, str(LAB))

from short_docs import SIZES, snippets  # noqa: E402
from steno import split_books  # noqa: E402

BOS = 256
VOCAB = 257
CTX = 512
STRIDE = 256
CONTAINER = 7  # bytes: mode + fingerprint + varint length + coder flush, as in codec v1
CKPT = ROOT / "tables"

CONFIGS = {  # name: (d_model, layers, heads)
    "1m": (128, 4, 4),
    "5m": (256, 6, 8),
    "20m": (512, 6, 8),
}


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.out = nn.Linear(4 * d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class ByteLM(nn.Module):
    def __init__(self, d, layers, heads):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(CTX, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.quant_head = False
        for name, p in self.named_parameters():  # GPT-2 style init
            if p.dim() >= 2:
                std = 0.02 / math.sqrt(2 * layers) if name.endswith(("proj.weight", "out.weight")) else 0.02
                nn.init.normal_(p, std=std)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for block in self.blocks:
            x = block(x)
        h = self.ln(x)
        if self.quant_head:
            h = fake_quant(h, dim=-1)
        return h @ self.emb.weight.T  # tied output head

    def params(self):
        return sum(p.numel() for p in self.parameters())

    def flops_per_byte(self):
        """Inference multiply-adds x2 per byte with a KV cache, at full context."""
        d = self.emb.weight.shape[1]
        layers = len(self.blocks)
        matmul = layers * 12 * d * d + d * VOCAB
        attention = layers * 2 * CTX * d
        return 2 * (matmul + attention)


def build(name):
    return ByteLM(*CONFIGS[name])


def train_bytes():
    train, _ = split_books()
    return np.frombuffer(b"".join(p.read_bytes() for p in train), dtype=np.uint8)


def test_docs():
    """{'1000': [bytes, ...], ..., 'books': [bytes, ...]} -- the shared E6/E7 test set."""
    _, test = split_books()
    texts = [p.read_text(encoding="utf-8") for p in test]
    docs = {str(size): [s.encode() for text in texts for s in snippets(text, size)] for size in SIZES}
    docs["books"] = [p.read_bytes() for p in test]
    return docs


def windows(data):
    """(inputs, targets, score-from index) per sliding window over [BOS] + data."""
    x = np.concatenate(([BOS], np.frombuffer(data, dtype=np.uint8).astype(np.int64)))
    n = len(data)
    out = []
    s = 0
    while True:
        inp = x[s:s + CTX]
        tgt = x[s + 1:s + 1 + CTX]
        out.append((inp[:len(tgt)], tgt, 0 if s == 0 else CTX - STRIDE))
        if s + CTX >= n:
            return out
        s += STRIDE


@torch.no_grad()
def doc_bits(model, docs, device, batch=64):
    """Ideal code length in bits for each document."""
    model.eval()
    items = [(i, w) for i, d in enumerate(docs) for w in windows(d)]
    bits = np.zeros(len(docs))
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        t = max(len(w[1]) for _, w in chunk)
        inp = torch.full((len(chunk), t), BOS, dtype=torch.long)
        tgt = torch.zeros((len(chunk), t), dtype=torch.long)
        mask = torch.zeros((len(chunk), t), dtype=torch.bool)
        for j, (_, (a, b, frm)) in enumerate(chunk):
            inp[j, :len(a)] = torch.from_numpy(a)
            tgt[j, :len(b)] = torch.from_numpy(b)
            mask[j, frm:len(b)] = True
        logp = F.log_softmax(model(inp.to(device)).float(), dim=-1)
        nll = -logp.gather(2, tgt.to(device).unsqueeze(2)).squeeze(2) / math.log(2)
        per = (nll * mask.to(device)).sum(1).cpu().numpy()
        for j, (i, _) in enumerate(chunk):
            bits[i] += per[j]
    return bits


def summarize(model, device, docs):
    """Short docs: avg bytes per size (ceil(bits/8) + container) and bpc; books: total bpc."""
    res = {}
    for key, ds in docs.items():
        bits = doc_bits(model, ds, device)
        raw = sum(len(d) for d in ds)
        res[key] = {"bpc": float(bits.sum() / raw),
                    "avg_bytes": float(np.mean([math.ceil(b / 8) + CONTAINER for b in bits])),
                    "per_doc_bits": bits.tolist()}
    return res


# --- fake int8 quantisation ---------------------------------------------------

def fake_quant(x, dim):
    """Symmetric int8 quantise-dequantise with one scale per slice along `dim`."""
    scale = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8) / 127
    return (x / scale).round().clamp(-127, 127) * scale


class QuantLinear(nn.Module):
    """int8 weights (per output channel) and int8 activations (per token row)."""

    def __init__(self, lin):
        super().__init__()
        self.weight = nn.Parameter(fake_quant(lin.weight.data, dim=1), requires_grad=False)
        self.bias = lin.bias

    def forward(self, x):
        return F.linear(fake_quant(x, dim=-1), self.weight, self.bias)


def quantized(model):
    import copy
    q = copy.deepcopy(model)
    for block in q.blocks:
        for name in ("qkv", "proj", "fc", "out"):
            setattr(block, name, QuantLinear(getattr(block, name)))
    q.emb.weight.data = fake_quant(q.emb.weight.data, dim=1)  # also the tied output head
    q.pos.weight.data = fake_quant(q.pos.weight.data, dim=1)
    q.quant_head = True
    return q


@torch.no_grad()
def byte_bits(model, docs, device, batch=64):
    """Per-byte ideal bits for each document (same windows as doc_bits)."""
    model.eval()
    items = [(i, w) for i, d in enumerate(docs) for w in windows(d)]
    out = [[] for _ in docs]
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        t = max(len(w[1]) for _, w in chunk)
        inp = torch.full((len(chunk), t), BOS, dtype=torch.long)
        tgt = torch.zeros((len(chunk), t), dtype=torch.long)
        for j, (_, (a, b, _)) in enumerate(chunk):
            inp[j, :len(a)] = torch.from_numpy(a)
            tgt[j, :len(b)] = torch.from_numpy(b)
        logp = F.log_softmax(model(inp.to(device)).float(), dim=-1)
        nll = (-logp.gather(2, tgt.to(device).unsqueeze(2)).squeeze(2) / math.log(2)).cpu().numpy()
        for j, (i, (_, b, frm)) in enumerate(chunk):
            out[i].append(nll[j, frm:len(b)])
    return [np.concatenate(parts) for parts in out]
