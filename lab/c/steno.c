/* E11: C11 port of the steno codec (lab/codec.py v1 static, mode 1) and the adaptive
 * per-document layer (lab/exp_adaptive/adaptive.py, config d3-sum-w32, mode 2).
 * Bit-identical to the Python reference. No dependencies beyond libc + POSIX mmap.
 *
 *   steno compress   [-s|-a] [-u] [-v] TABLE IN OUT [IN OUT ...]   (-s static mode 1 = default, -a adaptive)
 *   steno decompress [-u] [-v] TABLE IN OUT [IN OUT ...]           (mode taken from the file)
 *   steno info TABLE                                               header + load time
 *   steno verify TABLE                                             recompute the SHA-256 fingerprint
 *
 * -u: E9 chain variant (lab/exp_chain/chain.py count_unk): a spelled word also counts as UNK at
 *     order 0. Not used by adaptive.py; both write mode byte 2, so the reader must know which.
 * -v: per file timing on stderr (coding only, table load excluded).
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define UNK 0
#define SPELL_END 256
#define TOPV 0xFFFFFFFFull
#define HALF (1ull << 31)
#define QUARTER (1ull << 30)
#define MAX_TOTAL QUARTER
#define MAX_NEW (1u << 19)
#define MODE_STORED 0
#define MODE_STATIC 1
#define MODE_ADAPTIVE 2

static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "steno: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}
static void *xcalloc(size_t n, size_t s) {
    void *p = calloc(n ? n : 1, s ? s : 1);
    if (!p) die("out of memory");
    return p;
}
static void *xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) die("out of memory");
    return p;
}
static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint64_t fnv1a64(const uint8_t *p, size_t n) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 0x100000001B3ull;
    return h;
}

/* ---------------------------------------------------------------- table (mmap) */

enum { S_BLOB, S_LENS, S_OFF, S_UNI, S_UNICUM, S_SPELL, S_SPELLCUM,
       S_C1CTX, S_C1START, S_C1SYM, S_C1CNT, S_C1CUM,
       S_C2CTX, S_C2START, S_C2SYM, S_C2CNT, S_C2CUM, S_HASH, S_C1UNICUM, S_C2C1CUM, S_C1INDEX, S_C2HASH, N_SECTIONS };

typedef struct {
    const uint8_t *base;
    size_t size;
    uint8_t fp[4];
    uint64_t V, b, total0, blob_len, hash_slots;
    uint64_t nctx[3], nent[3];
    uint64_t sec_off[N_SECTIONS], sec_len[N_SECTIONS];
    const uint8_t *blob;
    const uint32_t *lens;
    const uint64_t *voff;
    const uint32_t *uni;
    const uint64_t *uni_cum;
    const uint32_t *spell, *spell_cum;
    const int64_t *ctx[3];
    const uint32_t *start[3], *sym[3], *cnt[3], *cum[3];
    const uint32_t *hash;
    const uint32_t *unicum1, *c1cum2, *c1index, *c2hash;
    uint64_t c2bits, c2slots;
    int nested;
} Table;

static uint64_t rd64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }  /* LE host */

static void table_open(Table *t, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("%s: %s", path, strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0) die("%s: %s", path, strerror(errno));
    t->size = (size_t)st.st_size;
    if (t->size < 512) die("%s: too small", path);
    void *m = mmap(NULL, t->size, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) die("%s: mmap: %s", path, strerror(errno));
    close(fd);
    /* lookups are random: no read-ahead / fault-around (keeps first-touch cost and RSS proportional to use) */
    if (!getenv("STENO_NO_MADVISE")) posix_madvise(m, t->size, POSIX_MADV_RANDOM);
    t->base = m;
    const uint8_t *h = t->base;
    uint32_t ver, hsize;
    memcpy(&ver, h + 8, 4);
    memcpy(&hsize, h + 12, 4);
    if (memcmp(h, "STNCTAB1", 8) || ver != 1 || hsize != 512) die("%s: not a STNCTAB1 v1 table", path);
    memcpy(t->fp, h + 16, 4);
    t->V = rd64(h + 24); t->b = rd64(h + 32); t->total0 = rd64(h + 40);
    t->nctx[1] = rd64(h + 48); t->nent[1] = rd64(h + 56);
    t->nctx[2] = rd64(h + 64); t->nent[2] = rd64(h + 72);
    t->blob_len = rd64(h + 80); t->hash_slots = rd64(h + 88);
    if (rd64(h + 96) != N_SECTIONS) die("%s: unexpected section count", path);
    for (int i = 0; i < N_SECTIONS; i++) {
        t->sec_off[i] = rd64(h + 128 + 16 * i);
        t->sec_len[i] = rd64(h + 136 + 16 * i);
        if (t->sec_off[i] + t->sec_len[i] > t->size || t->sec_off[i] % 8) die("%s: bad section %d", path, i);
    }
#define SEC(i, type) ((const type *)(t->base + t->sec_off[i]))
    uint64_t V = t->V;
    uint64_t want[N_SECTIONS] = {t->blob_len, 4 * V, 8 * (V + 1), 4 * V, 8 * V, 4 * 257 * 257, 4 * 257 * 257,
                                 8 * t->nctx[1], 4 * (t->nctx[1] + 1), 4 * t->nent[1], 4 * t->nent[1], 4 * t->nent[1],
                                 8 * t->nctx[2], 4 * (t->nctx[2] + 1), 4 * t->nent[2], 4 * t->nent[2], 4 * t->nent[2],
                                 4 * t->hash_slots, 4 * t->nent[1], 4 * t->nent[2], 4 * V, 0};
    t->c2bits = rd64(h + 112);
    t->c2slots = t->sec_len[S_C2HASH] / 4;
    if (t->c2bits < 4 || t->c2bits > 40 || t->c2slots < (1ull << t->c2bits)) die("%s: bad c2_hash", path);
    want[S_C2HASH] = t->sec_len[S_C2HASH];
    for (int i = 0; i < N_SECTIONS; i++)
        if (t->sec_len[i] != want[i]) die("%s: section %d has %llu bytes, expected %llu", path, i,
                                          (unsigned long long)t->sec_len[i], (unsigned long long)want[i]);
    if (V < 1 || V >= (1ull << 31) || t->total0 >= MAX_TOTAL || (t->hash_slots & (t->hash_slots - 1)))
        die("%s: bad header", path);
    t->blob = SEC(S_BLOB, uint8_t);
    t->lens = SEC(S_LENS, uint32_t);
    t->voff = SEC(S_OFF, uint64_t);
    t->uni = SEC(S_UNI, uint32_t);
    t->uni_cum = SEC(S_UNICUM, uint64_t);
    t->spell = SEC(S_SPELL, uint32_t);
    t->spell_cum = SEC(S_SPELLCUM, uint32_t);
    t->ctx[1] = SEC(S_C1CTX, int64_t); t->start[1] = SEC(S_C1START, uint32_t);
    t->sym[1] = SEC(S_C1SYM, uint32_t); t->cnt[1] = SEC(S_C1CNT, uint32_t); t->cum[1] = SEC(S_C1CUM, uint32_t);
    t->ctx[2] = SEC(S_C2CTX, int64_t); t->start[2] = SEC(S_C2START, uint32_t);
    t->sym[2] = SEC(S_C2SYM, uint32_t); t->cnt[2] = SEC(S_C2CNT, uint32_t); t->cum[2] = SEC(S_C2CUM, uint32_t);
    t->hash = SEC(S_HASH, uint32_t);
    t->unicum1 = SEC(S_C1UNICUM, uint32_t);
    t->c1cum2 = SEC(S_C2C1CUM, uint32_t);
    t->nested = (int)(rd64(h + 104) & 1);
    if (getenv("STENO_NO_NESTED")) t->nested = 0;  /* test hook: force the general exclusion path */
    t->c1index = SEC(S_C1INDEX, uint32_t);
    t->c2hash = SEC(S_C2HASH, uint32_t);
#undef SEC
}

static uint32_t table_lookup(const Table *t, const uint8_t *p, size_t n) {
    uint64_t mask = t->hash_slots - 1, s = fnv1a64(p, n) & mask;
    for (;;) {
        uint32_t id = t->hash[s];
        if (!id) return UNK;
        if (t->lens[id] == n && memcmp(t->blob + t->voff[id], p, n) == 0) return id;
        s = (s + 1) & mask;
    }
}

/* table context of order k for the (UNK-mapped) history; returns 1 and the entry range if it exists */
static int table_group(const Table *t, int k, const uint32_t *th, int hlen, uint64_t *lo, uint64_t *hi) {
    int64_t key = 0;
    for (int j = 1; j <= k; j++) key |= (int64_t)th[hlen - j] << (t->b * (j - 1));
    uint64_t a;
    if (k == 1) {
        uint32_t v = t->c1index[key];
        if (!v) return 0;
        a = v - 1;
    } else {
        uint64_t s = ((uint64_t)key * 0x9E3779B97F4A7C15ull) >> (64 - t->c2bits);
        for (;; s++) {
            if (s >= t->c2slots) return 0;
            uint32_t v = t->c2hash[s];
            if (!v) return 0;
            if (t->ctx[2][v - 1] == key) { a = v - 1; break; }
        }
    }
    *lo = t->start[k][a];
    *hi = t->start[k][a + 1];
    return 1;
}

/* ---------------------------------------------------------------- arithmetic coder */

typedef struct {
    uint64_t low, high, pending;
    uint8_t *buf;
    size_t cap, nbits;
} Encoder;

static void enc_init(Encoder *e) {
    e->low = 0; e->high = TOPV; e->pending = 0;
    e->cap = 1 << 16; e->buf = xcalloc(e->cap, 1); e->nbits = 0;
}
static inline void put_bit(Encoder *e, int bit) {
    if ((e->nbits >> 3) >= e->cap) {
        e->buf = xrealloc(e->buf, e->cap * 2);
        memset(e->buf + e->cap, 0, e->cap);
        e->cap *= 2;
    }
    if (bit) e->buf[e->nbits >> 3] |= (uint8_t)(0x80 >> (e->nbits & 7));
    e->nbits++;
}
static inline void emit(Encoder *e, int bit) {
    put_bit(e, bit);
    for (; e->pending; e->pending--) put_bit(e, !bit);
}
static void enc_encode(Encoder *e, uint64_t lo, uint64_t hi, uint64_t total) {
    if (!(lo < hi && hi <= total && total <= MAX_TOTAL))
        die("encode: bad interval [%llu, %llu) of %llu", (unsigned long long)lo, (unsigned long long)hi,
            (unsigned long long)total);
    uint64_t span = e->high - e->low + 1;
    e->high = e->low + span * hi / total - 1;
    e->low = e->low + span * lo / total;
    for (;;) {
        if (e->high < HALF) {
            emit(e, 0);
        } else if (e->low >= HALF) {
            emit(e, 1); e->low -= HALF; e->high -= HALF;
        } else if (e->low >= QUARTER && e->high < HALF + QUARTER) {
            e->pending++; e->low -= QUARTER; e->high -= QUARTER;
        } else break;
        e->low = 2 * e->low;
        e->high = 2 * e->high + 1;
    }
}
/* returns the code bytes (trailing zero bytes removed) */
static size_t enc_finish(Encoder *e) {
    e->pending++;
    emit(e, e->low < QUARTER ? 0 : 1);
    size_t n = (e->nbits + 7) >> 3;
    while (n && e->buf[n - 1] == 0) n--;
    return n;
}

typedef struct {
    const uint8_t *data;
    size_t nbits, pos;
    uint64_t low, high, value;
} Decoder;

static inline int get_bit(Decoder *d) {
    int bit = d->pos < d->nbits ? (d->data[d->pos >> 3] >> (7 - (d->pos & 7))) & 1 : 0;
    d->pos++;
    return bit;
}
static void dec_init(Decoder *d, const uint8_t *data, size_t n) {
    d->data = data; d->nbits = n * 8; d->pos = 0;
    d->low = 0; d->high = TOPV; d->value = 0;
    for (int i = 0; i < 32; i++) d->value = 2 * d->value + get_bit(d);
}
static uint64_t dec_target(Decoder *d, uint64_t total) {
    if (total == 0 || total > MAX_TOTAL) die("decode: bad total %llu", (unsigned long long)total);
    uint64_t span = d->high - d->low + 1;
    return ((d->value - d->low + 1) * total - 1) / span;
}
static void dec_consume(Decoder *d, uint64_t lo, uint64_t hi, uint64_t total) {
    uint64_t span = d->high - d->low + 1;
    d->high = d->low + span * hi / total - 1;
    d->low = d->low + span * lo / total;
    for (;;) {
        if (d->high < HALF) {
        } else if (d->low >= HALF) {
            d->low -= HALF; d->high -= HALF; d->value -= HALF;
        } else if (d->low >= QUARTER && d->high < HALF + QUARTER) {
            d->low -= QUARTER; d->high -= QUARTER; d->value -= QUARTER;
        } else break;
        d->low = 2 * d->low;
        d->high = 2 * d->high + 1;
        d->value = 2 * d->value + get_bit(d);
    }
}

/* ---------------------------------------------------------------- config */

typedef struct { char kind; int k; int64_t w; } LevelSpec;
typedef struct {
    int mode, nlev;
    LevelSpec lev[4];
    int64_t w0, ws;
    int new_words, count_unk;
    int doc_order[4], ndoc;  /* sorted distinct k of M/D levels */
} Config;

static void config_finish(Config *c) {
    c->ndoc = 0;
    for (int k = 1; k <= 3; k++)
        for (int i = 0; i < c->nlev; i++)
            if ((c->lev[i].kind == 'M' || c->lev[i].kind == 'D') && c->lev[i].k == k) {
                c->doc_order[c->ndoc++] = k;
                break;
            }
}
static Config config_static(void) {
    Config c = {MODE_STATIC, 2, {{'T', 2, 0}, {'T', 1, 0}}, 0, 0, 0, 0, {0}, 0};
    config_finish(&c);
    return c;
}
static Config config_d3_sum_w32(int count_unk) {
    Config c = {MODE_ADAPTIVE, 3, {{'D', 3, 1}, {'M', 2, 32}, {'M', 1, 32}}, 32, 1, 1, count_unk, {0}, 0};
    config_finish(&c);
    return c;
}

/* ---------------------------------------------------------------- document state */

typedef struct { int k; uint64_t lo, hi; } Seg;  /* table entries [lo, hi) of order k; k = 0: none */

typedef struct {
    uint32_t key[3];
    int k;
    uint32_t n, cap;
    uint32_t *sym;   /* ascending */
    int64_t *cnt;
    int64_t *tpos;   /* position inside the table segment of the same (UNK-mapped) context, or -1 */
    int has_tab;
    uint64_t tlo, thi;
} DocCtx;

typedef struct {
    const Table *t;
    const Config *cfg;
    uint32_t V;
    size_t nsize;            /* V + MAX_NEW */
    int64_t *doc_uni, *fen;  /* fen: Fenwick tree, 1-based, nsize + 1 */
    int64_t doc_total;
    int64_t *doc_spell;      /* 257 x 257 */
    /* doc contexts: open addressing hash -> index into ctxs */
    DocCtx *ctxs;
    size_t nctxs, capctxs;
    uint32_t *chash;
    size_t cslots;
    /* new words */
    uint32_t n_new;
    uint8_t *arena;
    size_t arena_len, arena_cap;
    size_t *new_off;         /* n_new + 1 offsets */
    uint32_t *new_len;
    uint32_t *nhash;         /* value = new index + 1 */
    size_t nslots;
    /* excluded set X = Xs ∪ Tb (disjoint): Xs sorted ids, Tb one table segment kept virtual */
    uint32_t *Xs, *Xs2;
    size_t nXs, capXs;
    Seg Tb;
    uint8_t *xs_found;       /* per Xs entry: found in the current level's table segment */
    uint32_t *tb_res;        /* Tb symbols not in the current level's table segment */
    size_t ntb_res, capT;
    /* sorted table ids with doc_uni > 0 (for the order-0 exclusion of a table segment) */
    uint32_t *U;
    size_t nU, capU;
} Doc;

static void doc_init(Doc *d, const Table *t, const Config *cfg) {
    memset(d, 0, sizeof *d);
    d->t = t; d->cfg = cfg; d->V = (uint32_t)t->V;
    d->nsize = (size_t)t->V + MAX_NEW;
    if (cfg->w0) {
        d->doc_uni = xcalloc(d->nsize, sizeof(int64_t));
        d->fen = xcalloc(d->nsize + 1, sizeof(int64_t));
    }
    d->doc_spell = xcalloc(257 * 257, sizeof(int64_t));
    d->cslots = 1024;
    d->chash = xcalloc(d->cslots, sizeof(uint32_t));
    d->nslots = 1024;
    d->nhash = xcalloc(d->nslots, sizeof(uint32_t));
    d->new_off = xmalloc(sizeof(size_t));
    d->new_off[0] = 0;
}
static void doc_free(Doc *d) {
    free(d->doc_uni); free(d->fen); free(d->doc_spell);
    for (size_t i = 0; i < d->nctxs; i++) { free(d->ctxs[i].sym); free(d->ctxs[i].cnt); free(d->ctxs[i].tpos); }
    free(d->ctxs); free(d->chash); free(d->arena); free(d->new_off); free(d->new_len); free(d->nhash);
    free(d->Xs); free(d->Xs2); free(d->xs_found); free(d->tb_res); free(d->U);
}

static inline uint64_t ctx_hash(int k, const uint32_t *key) {
    uint64_t h = (uint64_t)k * 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < k; i++) {
        h ^= key[i] + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
        h *= 0xFF51AFD7ED558CCDull;
    }
    return h ^ (h >> 29);
}

/* key = last k actual history ids, oldest first (matches tuple(hist[-k:])) */
static DocCtx *doc_ctx_find(Doc *d, int k, const uint32_t *key, int create) {
    uint64_t mask = d->cslots - 1, s = ctx_hash(k, key) & mask;
    for (;;) {
        uint32_t v = d->chash[s];
        if (!v) break;
        DocCtx *c = &d->ctxs[v - 1];
        if (c->k == k && memcmp(c->key, key, 4 * k) == 0) return c;
        s = (s + 1) & mask;
    }
    if (!create) return NULL;
    if (d->nctxs == d->capctxs) {
        d->capctxs = d->capctxs ? 2 * d->capctxs : 1024;
        d->ctxs = xrealloc(d->ctxs, d->capctxs * sizeof(DocCtx));
    }
    DocCtx *c = &d->ctxs[d->nctxs++];
    memset(c, 0, sizeof *c);
    c->k = k;
    memcpy(c->key, key, 4 * k);
    if (k <= 2) {
        uint32_t th[2];
        for (int i = 0; i < k; i++) th[i] = key[i] < d->V ? key[i] : UNK;
        c->has_tab = table_group(d->t, k, th, k, &c->tlo, &c->thi);
    }
    d->chash[s] = (uint32_t)d->nctxs;
    if (2 * d->nctxs > d->cslots) {  /* grow + rehash */
        size_t ns = d->cslots * 2;
        uint32_t *nh = xcalloc(ns, sizeof(uint32_t));
        for (size_t i = 0; i < d->nctxs; i++) {
            uint64_t q = ctx_hash(d->ctxs[i].k, d->ctxs[i].key) & (ns - 1);
            while (nh[q]) q = (q + 1) & (ns - 1);
            nh[q] = (uint32_t)(i + 1);
        }
        free(d->chash);
        d->chash = nh;
        d->cslots = ns;
    }
    return c;
}

static int64_t lower_bound_u32(const uint32_t *a, int64_t lo, int64_t hi, uint32_t v) {
    while (lo < hi) {
        int64_t m = lo + (hi - lo) / 2;
        if (a[m] < v) lo = m + 1; else hi = m;
    }
    return lo;
}

static void doc_ctx_add(Doc *d, DocCtx *c, uint32_t w) {
    int64_t i = lower_bound_u32(c->sym, 0, c->n, w);
    if (i < c->n && c->sym[i] == w) { c->cnt[i]++; return; }
    if (c->n == c->cap) {
        c->cap = c->cap ? 2 * c->cap : 4;
        c->sym = xrealloc(c->sym, c->cap * 4);
        c->cnt = xrealloc(c->cnt, c->cap * 8);
        c->tpos = xrealloc(c->tpos, c->cap * 8);
    }
    memmove(c->sym + i + 1, c->sym + i, (c->n - i) * 4);
    memmove(c->cnt + i + 1, c->cnt + i, (c->n - i) * 8);
    memmove(c->tpos + i + 1, c->tpos + i, (c->n - i) * 8);
    c->sym[i] = w;
    c->cnt[i] = 1;
    c->tpos[i] = -1;
    if (c->has_tab && w < d->V) {
        const uint32_t *ts = d->t->sym[c->k];
        int64_t p = lower_bound_u32(ts, (int64_t)c->tlo, (int64_t)c->thi, w);
        if (p < (int64_t)c->thi && ts[p] == w) c->tpos[i] = p - (int64_t)c->tlo;
    }
    c->n++;
}

static inline void fen_add(Doc *d, size_t i, int64_t delta) {
    for (i++; i <= d->nsize; i += i & (~i + 1)) d->fen[i] += delta;
}
static inline int64_t fen_prefix(const Doc *d, size_t i) {  /* sum of entries 0..i-1 */
    int64_t s = 0;
    for (; i > 0; i -= i & (~i + 1)) s += d->fen[i];
    return s;
}

static uint32_t new_lookup(const Doc *d, const uint8_t *p, size_t n) {
    if (!d->n_new) return UNK;
    uint64_t mask = d->nslots - 1, s = fnv1a64(p, n) & mask;
    for (;;) {
        uint32_t v = d->nhash[s];
        if (!v) return UNK;
        uint32_t j = v - 1;
        if (d->new_len[j] == n && memcmp(d->arena + d->new_off[j], p, n) == 0) return d->V + j;
        s = (s + 1) & mask;
    }
}
static void new_insert_hash(uint32_t *h, size_t slots, uint64_t hv, uint32_t val) {
    uint64_t s = hv & (slots - 1);
    while (h[s]) s = (s + 1) & (slots - 1);
    h[s] = val;
}
static uint32_t new_add(Doc *d, const uint8_t *p, size_t n) {
    if (d->n_new >= MAX_NEW) die("more than %u new words in one document", MAX_NEW);
    if (d->arena_len + n > d->arena_cap) {
        d->arena_cap = (d->arena_cap + n) * 2;
        d->arena = xrealloc(d->arena, d->arena_cap);
    }
    memcpy(d->arena + d->arena_len, p, n);
    uint32_t j = d->n_new++;
    d->new_off = xrealloc(d->new_off, (d->n_new + 1) * sizeof(size_t));
    d->new_len = xrealloc(d->new_len, d->n_new * sizeof(uint32_t));
    d->new_off[j] = d->arena_len;
    d->new_len[j] = (uint32_t)n;
    d->arena_len += n;
    d->new_off[j + 1] = d->arena_len;
    if (2 * d->n_new > d->nslots) {
        size_t ns = d->nslots * 2;
        uint32_t *nh = xcalloc(ns, sizeof(uint32_t));
        for (uint32_t i = 0; i < j; i++)
            new_insert_hash(nh, ns, fnv1a64(d->arena + d->new_off[i], d->new_len[i]), i + 1);
        free(d->nhash);
        d->nhash = nh;
        d->nslots = ns;
    }
    new_insert_hash(d->nhash, d->nslots, fnv1a64(p, n), j + 1);
    return d->V + j;
}

static void doc_uni_inc(Doc *d, uint32_t w) {
    if (d->doc_uni[w]++ == 0 && w < d->V) {
        int64_t i = lower_bound_u32(d->U, 0, (int64_t)d->nU, w);
        if (d->nU == d->capU) { d->capU = d->capU ? 2 * d->capU : 1024; d->U = xrealloc(d->U, d->capU * 4); }
        memmove(d->U + i + 1, d->U + i, (d->nU - i) * 4);
        d->U[i] = w;
        d->nU++;
    }
    fen_add(d, w, 1);
}

/* learning after each token; identical in encoder and decoder. Returns the history id. */
static uint32_t doc_learn(Doc *d, const uint8_t *tok, size_t n, uint32_t w, const uint32_t *hist, int hlen) {
    const Config *cfg = d->cfg;
    int spelled = w == UNK;
    if (cfg->count_unk && spelled && cfg->w0) {
        doc_uni_inc(d, UNK);
        d->doc_total++;
    }
    if (spelled && cfg->new_words) w = new_add(d, tok, n);
    if (cfg->ws && spelled) {
        int prev = SPELL_END;
        for (size_t i = 0; i <= n; i++) {
            int b = i < n ? tok[i] : SPELL_END;
            d->doc_spell[prev * 257 + b]++;
            prev = b;
        }
    }
    if (cfg->w0) doc_uni_inc(d, w);
    d->doc_total++;
    for (int i = 0; i < cfg->ndoc; i++) {
        int k = cfg->doc_order[i];
        if (hlen >= k) doc_ctx_add(d, doc_ctx_find(d, k, hist + hlen - k, 1), w);
    }
    return w;
}

/* ---------------------------------------------------------------- levels */

/* One PPM level after exclusion. Candidate order: table segment (ascending id, excluded entries get
 * width 0), then document-only symbols (ascending id). */
typedef struct {
    Seg seg;
    const uint32_t *tsym, *tcum;  /* table segment */
    int64_t tn;
    /* adjustments on table positions (sorted, unique): doc additions (+) and exclusions (-count) */
    int64_t na, capa;
    int64_t *apos, *adelta, *apre;  /* apre: inclusive prefix of adelta */
    uint8_t *arem;
    int64_t *rtmp;                  /* scratch: removal positions */
    int64_t nrem;
    /* virtual removal of a whole order-2 segment nested in this order-1 segment */
    const uint32_t *vsym, *vcum;    /* its symbols and inclusive prefix of their order-1 counts */
    int64_t vn;
    /* document-only part (not in the table segment, not excluded) */
    int64_t nd, capd;
    uint32_t *dsym;
    int64_t *dcnt, *dcum;           /* dcum inclusive */
    int64_t *dpos, *dval;           /* scratch: doc additions on table positions */
    int64_t tsum, esc, total;
    const DocCtx *doc;
} Level;

static void lev_reserve(Level *L, int64_t na, int64_t nd) {
    if (na > L->capa) {
        L->capa = na * 2;
        L->apos = xrealloc(L->apos, L->capa * 8);
        L->adelta = xrealloc(L->adelta, L->capa * 8);
        L->apre = xrealloc(L->apre, L->capa * 8);
        L->rtmp = xrealloc(L->rtmp, L->capa * 8);
        L->arem = xrealloc(L->arem, L->capa);
    }
    if (nd > L->capd) {
        L->capd = nd * 2;
        L->dsym = xrealloc(L->dsym, L->capd * 4);
        L->dcnt = xrealloc(L->dcnt, L->capd * 8);
        L->dcum = xrealloc(L->dcum, L->capd * 8);
        L->dpos = xrealloc(L->dpos, L->capd * 8);
        L->dval = xrealloc(L->dval, L->capd * 8);
    }
}
static void doc_reserve(Doc *d, size_t nxs, size_t ntb) {
    if (nxs > d->capXs) {
        d->capXs = nxs * 2;
        d->Xs = xrealloc(d->Xs, d->capXs * 4);
        d->Xs2 = xrealloc(d->Xs2, d->capXs * 4);
        d->xs_found = xrealloc(d->xs_found, d->capXs);
    }
    if (ntb > d->capT) {
        d->capT = ntb * 2;
        d->tb_res = xrealloc(d->tb_res, d->capT * 4);
    }
}
static inline int64_t tcum_at(const Level *L, int64_t p) { return p < 0 ? 0 : (int64_t)L->tcum[p]; }
/* sum of adjustments at positions < p */
static inline int64_t adj_before(const Level *L, int64_t p) {
    int64_t lo = 0, hi = L->na;
    while (lo < hi) {
        int64_t m = (lo + hi) / 2;
        if (L->apos[m] < p) lo = m + 1; else hi = m;
    }
    return lo ? L->apre[lo - 1] : 0;
}
/* order-1 mass of the virtually removed order-2 symbols < v */
static inline int64_t vbefore(const Level *L, uint64_t v) {
    if (!L->vn) return 0;
    int64_t i = lower_bound_u32(L->vsym, 0, L->vn, v > UINT32_MAX ? UINT32_MAX : (uint32_t)v);
    if (v > UINT32_MAX) i = L->vn;
    return i ? (int64_t)L->vcum[i - 1] : 0;
}
/* smallest i in [from, n) with a[i] >= v (exponential search from `from`) */
static inline int64_t gallop_u32(const uint32_t *a, int64_t from, int64_t n, uint32_t v) {
    if (from >= n || a[from] >= v) return from;
    int64_t bound = 1;
    while (from + bound < n && a[from + bound] < v) bound *= 2;
    int64_t hi = from + bound < n ? from + bound : n;
    return lower_bound_u32(a, from + bound / 2 + 1, hi, v);
}
static inline int in_sorted(const uint32_t *a, int64_t *from, int64_t n, uint32_t v) {
    int64_t i = gallop_u32(a, *from, n, v);
    *from = i;
    return i < n && a[i] == v;
}
static inline int advance_has(const uint32_t *a, int64_t *from, int64_t n, uint32_t v, int linear) {
    if (linear) {
        int64_t i = *from;
        while (i < n && a[i] < v) i++;
        *from = i;
    } else {
        *from = gallop_u32(a, *from, n, v);
    }
    return *from < n && a[*from] == v;
}
static inline const uint32_t *seg_sym(const Table *t, Seg s) { return t->sym[s.k] + s.lo; }

/* Build level `spec`; returns 0 if the level doesn't exist or has no candidates left. */
static int build_level(Doc *d, Level *L, const LevelSpec *spec, const uint32_t *hist, const uint32_t *th, int hlen) {
    const Table *t = d->t;
    int k = spec->k;
    if (hlen < k) return 0;
    L->tn = 0; L->na = 0; L->nrem = 0; L->nd = 0; L->doc = NULL; L->seg.k = 0; L->vn = 0;
    uint64_t tlo = 0, thi = 0;
    int has_tab = 0;
    if (spec->kind == 'T' || spec->kind == 'M') has_tab = table_group(t, k, th, hlen, &tlo, &thi);
    if (spec->kind != 'T') {
        L->doc = doc_ctx_find(d, k, hist + hlen - k, 0);
        if (L->doc && L->doc->n == 0) L->doc = NULL;
    }
    if (spec->kind == 'T' && !has_tab) return 0;
    if (spec->kind == 'D' && !L->doc) return 0;
    if (spec->kind == 'M' && !has_tab && !L->doc) return 0;
    if (has_tab) {
        L->seg = (Seg){k, tlo, thi};
        L->tsym = t->sym[k] + tlo;
        L->tcum = t->cum[k] + tlo;
        L->tn = (int64_t)(thi - tlo);
    }
    const DocCtx *c = L->doc;
    int64_t nc = c ? c->n : 0, W = spec->w;
    int64_t ntb = d->Tb.k ? (int64_t)(d->Tb.hi - d->Tb.lo) : 0;
    lev_reserve(L, nc + (int64_t)d->nXs + ntb + 1, nc + 1);
    doc_reserve(d, d->nXs + 1, (size_t)ntb + 1);
    d->ntb_res = 0;
    /* removal positions: excluded symbols inside this table segment */
    int64_t ra = 0;
    if (L->tn) {
        int64_t *rx = L->rtmp, nrx = 0, *rt = L->apre, nrt = 0;  /* apre is free until the prefix pass */
        int64_t from = 0;
        for (size_t i = 0; i < d->nXs; i++) {
            int f = in_sorted(L->tsym, &from, L->tn, d->Xs[i]);
            d->xs_found[i] = (uint8_t)f;
            if (f) rx[nrx++] = from;
        }
        if (ntb) {
            const uint32_t *ts = seg_sym(t, d->Tb);
            if (d->Tb.k == 2 && k == 1 && t->nested) {
                /* the order-2 segment of the same history nests in this order-1 segment: remove it
                   virtually, its order-1 mass comes from the c2_c1cum helper */
                L->vsym = ts;
                L->vcum = t->c1cum2 + d->Tb.lo;
                L->vn = ntb;
            } else {
                from = 0;
                for (int64_t e = 0; e < ntb; e++) {
                    if (in_sorted(L->tsym, &from, L->tn, ts[e])) rt[nrt++] = from;
                    else d->tb_res[d->ntb_res++] = ts[e];
                }
            }
        }
        /* merge the two sorted, disjoint position lists (via adelta, rebuilt below) into rtmp */
        int64_t i = 0, j = 0;
        int64_t *mr = L->adelta;
        while (i < nrx || j < nrt) {
            if (j >= nrt || (i < nrx && rx[i] < rt[j])) mr[ra++] = rx[i++];
            else mr[ra++] = rt[j++];
        }
        memcpy(L->rtmp, mr, ra * 8);
    }
    int64_t *rp = L->rtmp;
    /* one pass over the document context: membership in Xs and Tb by linear merge (or exponential
       search when the other list is much longer); table symbols give doc additions, others doc-only */
    int use_tpos = spec->kind == 'M' && has_tab;
    int64_t nadd = 0, nd = 0, dsum = 0;
    if (c) {
        const uint32_t *ts = ntb ? seg_sym(t, d->Tb) : NULL;
        int64_t nxs = (int64_t)d->nXs, xf = 0, tf = 0;
        int linx = nxs <= 8 * nc + 64, lint = ntb <= 8 * nc + 64;
        for (int64_t i = 0; i < nc; i++) {
            uint32_t v = c->sym[i];
            if (nxs && advance_has(d->Xs, &xf, nxs, v, linx)) continue;
            if (ntb && advance_has(ts, &tf, ntb, v, lint)) continue;
            if (use_tpos && c->tpos[i] >= 0) {
                L->dpos[nadd] = c->tpos[i];
                L->dval[nadd++] = W * c->cnt[i];
            } else {
                L->dsym[nd] = v;
                L->dcnt[nd] = W * c->cnt[i];
                dsum += L->dcnt[nd];
                L->dcum[nd++] = dsum;
            }
        }
    }
    L->nd = nd;
    /* merge removals (rtmp) with doc additions by table position; the two sets are disjoint */
    int64_t na = 0, ri = 0, ci = 0;
    while (ri < ra || ci < nadd) {
        if (ci >= nadd || (ri < ra && rp[ri] < L->dpos[ci])) {
            int64_t pr = rp[ri++];
            L->apos[na] = pr;
            L->adelta[na] = -(tcum_at(L, pr) - tcum_at(L, pr - 1));
            L->arem[na++] = 1;
        } else {
            L->apos[na] = L->dpos[ci];
            L->adelta[na] = L->dval[ci++];
            L->arem[na++] = 0;
        }
    }
    L->na = na;
    L->nrem = ra + L->vn;
    int64_t s = 0;
    for (int64_t i = 0; i < na; i++) { s += L->adelta[i]; L->apre[i] = s; }
    L->tsum = (L->tn ? (int64_t)L->tcum[L->tn - 1] : 0) + s - (L->vn ? (int64_t)L->vcum[L->vn - 1] : 0);
    L->esc = (L->tn - L->nrem) + L->nd;
    if (L->esc == 0) return 0;
    L->total = L->tsum + (L->nd ? L->dcum[L->nd - 1] : 0) + L->esc;
    return 1;
}

/* after an escape: X = X ∪ all symbols of the level. A level with a table segment becomes the new Tb;
 * everything else excluded goes to Xs (kept disjoint from Tb). */
static void merge_excluded(Doc *d, const Level *L) {
    const uint32_t *a = d->Xs, *b = d->tb_res, *c = L->dsym;
    size_t na = d->nXs, nb = L->tn ? d->ntb_res : 0, nc = (size_t)L->nd;
    doc_reserve(d, na + nb + nc + 1, 0);
    a = d->Xs;
    int filt = L->tn != 0;  /* drop Xs entries that are inside the new Tb */
    size_t i = 0, j = 0, k = 0, n = 0;
    uint32_t *out = d->Xs2;
    for (;;) {
        while (filt && i < na && d->xs_found[i]) i++;
        uint32_t v = UINT32_MAX;
        int src = -1;
        if (i < na && a[i] < v) { v = a[i]; src = 0; }
        if (j < nb && b[j] < v) { v = b[j]; src = 1; }
        if (k < nc && c[k] < v) { v = c[k]; src = 2; }
        if (src < 0) break;
        if (src == 0) i++; else if (src == 1) j++; else k++;
        out[n++] = v;
    }
    d->Xs2 = d->Xs;
    d->Xs = out;
    d->nXs = n;
    if (L->tn) d->Tb = L->seg;
}

/* ---------------------------------------------------------------- order 0 */

typedef struct {
    int64_t *epre;    /* inclusive prefix of the Xs widths */
    size_t cap;
    uint32_t *tdsym;  /* Tb symbols with a document count, ascending */
    int64_t *tdpre;   /* inclusive prefix of w0 * doc_uni over tdsym */
    size_t ntd, captd;
    int64_t total;
} Order0;

static inline int64_t width0(const Doc *d, uint32_t w) {
    int64_t u = w < d->V ? (int64_t)d->t->uni[w] : 0;
    return d->cfg->w0 ? u + d->cfg->w0 * d->doc_uni[w] : u;
}
static void order0_prepare(Doc *d, Order0 *o) {
    const Table *t = d->t;
    if (d->Tb.k && d->Tb.k != 1) {  /* no order-0 helper for this segment: materialize it into Xs */
        Level tmp = {0};
        tmp.tn = 1;
        d->ntb_res = 0;
        size_t ntb = d->Tb.hi - d->Tb.lo;
        doc_reserve(d, d->nXs + ntb + 1, ntb + 1);
        memcpy(d->tb_res, seg_sym(t, d->Tb), ntb * 4);
        d->ntb_res = ntb;
        memset(d->xs_found, 0, d->nXs);
        merge_excluded(d, &tmp);
        d->Tb.k = 0;
    }
    if (d->nXs > o->cap) {
        o->cap = d->nXs * 2;
        o->epre = xrealloc(o->epre, o->cap * 8);
    }
    int64_t s = 0;
    for (size_t i = 0; i < d->nXs; i++) { s += width0(d, d->Xs[i]); o->epre[i] = s; }
    o->ntd = 0;
    if (d->Tb.k == 1) {
        s += t->unicum1[d->Tb.hi - 1];
        if (d->cfg->w0 && d->nU) {
            const uint32_t *ts = seg_sym(t, d->Tb);
            int64_t n = (int64_t)(d->Tb.hi - d->Tb.lo);
            size_t need = (size_t)(n < (int64_t)d->nU ? n : (int64_t)d->nU) + 1;
            if (need > o->captd) {
                o->captd = need * 2;
                o->tdsym = xrealloc(o->tdsym, o->captd * 4);
                o->tdpre = xrealloc(o->tdpre, o->captd * 8);
            }
            int64_t ds = 0;
            if ((int64_t)d->nU * 16 < n) {  /* few document symbols: look them up in the segment */
                int64_t from = 0;
                for (size_t i = 0; i < d->nU && from < n; i++)
                    if (in_sorted(ts, &from, n, d->U[i])) {
                        ds += d->cfg->w0 * d->doc_uni[d->U[i]];
                        o->tdsym[o->ntd] = d->U[i];
                        o->tdpre[o->ntd++] = ds;
                    }
            } else {
                for (int64_t e = 0; e < n; e++) {
                    int64_t du = d->doc_uni[ts[e]];
                    if (du) {
                        ds += d->cfg->w0 * du;
                        o->tdsym[o->ntd] = ts[e];
                        o->tdpre[o->ntd++] = ds;
                    }
                }
            }
            s += ds;
        }
    }
    o->total = (int64_t)t->total0 + d->cfg->w0 * d->doc_total - s;
}
static int64_t cum_before(const Doc *d, const Order0 *o, uint64_t w) {
    const Table *t = d->t;
    int64_t tab = (w > 0 && w <= d->V) ? (int64_t)t->uni_cum[w - 1] : (w > d->V ? (int64_t)t->total0 : 0);
    int64_t doc = d->cfg->w0 ? d->cfg->w0 * fen_prefix(d, w) : 0;
    int64_t excl = 0;
    uint32_t w32 = w > UINT32_MAX ? UINT32_MAX : (uint32_t)w;
    if (d->nXs) {
        int64_t i = lower_bound_u32(d->Xs, 0, (int64_t)d->nXs, w32);
        excl += i ? o->epre[i - 1] : 0;
    }
    if (d->Tb.k == 1) {
        int64_t p = lower_bound_u32(t->sym[1], (int64_t)d->Tb.lo, (int64_t)d->Tb.hi, w32);
        excl += p > (int64_t)d->Tb.lo ? (int64_t)t->unicum1[p - 1] : 0;
        if (o->ntd) {
            int64_t i = lower_bound_u32(o->tdsym, 0, (int64_t)o->ntd, w32);
            excl += i ? o->tdpre[i - 1] : 0;
        }
    }
    return tab + doc - excl;
}

/* ---------------------------------------------------------------- token coding */

typedef struct {
    Doc doc;
    Level L;
    Order0 o0;
} Model;

static void encode_token(Model *m, Encoder *e, uint32_t w, const uint32_t *hist, const uint32_t *th, int hlen) {
    Doc *d = &m->doc;
    Level *L = &m->L;
    d->nXs = 0;
    d->Tb.k = 0;
    for (int li = 0; li < d->cfg->nlev; li++) {
        if (!build_level(d, L, &d->cfg->lev[li], hist, th, hlen)) continue;
        int64_t p = L->tn ? lower_bound_u32(L->tsym, 0, L->tn, w) : 0;
        if (p < L->tn && L->tsym[p] == w) {
            int64_t ai = 0, hi2 = L->na;  /* adjustment at p? */
            while (ai < hi2) { int64_t mm = (ai + hi2) / 2; if (L->apos[mm] < p) ai = mm + 1; else hi2 = mm; }
            int64_t before = ai ? L->apre[ai - 1] : 0;
            int64_t delta = (ai < L->na && L->apos[ai] == p) ? L->adelta[ai] : 0;
            if (ai < L->na && L->apos[ai] == p && L->arem[ai]) die("internal: coding an excluded symbol");
            int64_t vb = vbefore(L, w);
            int64_t lo = tcum_at(L, p - 1) + before - vb;
            int64_t hi = (int64_t)L->tcum[p] + before + delta - vb;
            enc_encode(e, lo, hi, L->total);
            return;
        }
        if (L->nd) {
            int64_t q = lower_bound_u32(L->dsym, 0, L->nd, w);
            if (q < L->nd && L->dsym[q] == w) {
                int64_t lo = L->tsum + (q ? L->dcum[q - 1] : 0);
                enc_encode(e, lo, lo + L->dcnt[q], L->total);
                return;
            }
        }
        enc_encode(e, L->total - L->esc, L->total, L->total);
        merge_excluded(d, L);
    }
    order0_prepare(d, &m->o0);
    int64_t lo = cum_before(d, &m->o0, w);
    enc_encode(e, lo, lo + width0(d, w), m->o0.total);
}

static uint32_t decode_token(Model *m, Decoder *dc, const uint32_t *hist, const uint32_t *th, int hlen) {
    Doc *d = &m->doc;
    Level *L = &m->L;
    d->nXs = 0;
    d->Tb.k = 0;
    for (int li = 0; li < d->cfg->nlev; li++) {
        if (!build_level(d, L, &d->cfg->lev[li], hist, th, hlen)) continue;
        int64_t total = L->total;
        int64_t target = (int64_t)dec_target(dc, total);
        if (target >= total - L->esc) {
            dec_consume(dc, total - L->esc, total, total);
            merge_excluded(d, L);
            continue;
        }
        if (target < L->tsum) {
            /* smallest p with tcum[p] + adj(<= p) > target */
            int64_t lo = 0, hi = L->tn - 1;
            while (lo < hi) {
                int64_t mid = (lo + hi) / 2;
                if ((int64_t)L->tcum[mid] + adj_before(L, mid + 1) - vbefore(L, (uint64_t)L->tsym[mid] + 1) > target)
                    hi = mid;
                else
                    lo = mid + 1;
            }
            int64_t vb = vbefore(L, L->tsym[lo]);
            int64_t c_lo = tcum_at(L, lo - 1) + adj_before(L, lo) - vb;
            int64_t c_hi = (int64_t)L->tcum[lo] + adj_before(L, lo + 1) - vb;
            dec_consume(dc, c_lo, c_hi, total);
            return L->tsym[lo];
        }
        int64_t lo = 0, hi = L->nd - 1;
        while (lo < hi) {
            int64_t mid = (lo + hi) / 2;
            if (L->tsum + L->dcum[mid] > target) hi = mid; else lo = mid + 1;
        }
        int64_t c_lo = L->tsum + (lo ? L->dcum[lo - 1] : 0);
        dec_consume(dc, c_lo, c_lo + L->dcnt[lo], total);
        return L->dsym[lo];
    }
    order0_prepare(d, &m->o0);
    int64_t total = m->o0.total;
    int64_t target = (int64_t)dec_target(dc, total);
    int64_t lo = 0, hi = (int64_t)d->V + d->n_new - 1;
    while (lo < hi) {
        int64_t mid = (lo + hi) / 2;
        if (cum_before(d, &m->o0, mid + 1) > target) hi = mid; else lo = mid + 1;
    }
    int64_t c_lo = cum_before(d, &m->o0, lo);
    dec_consume(dc, c_lo, c_lo + width0(d, (uint32_t)lo), total);
    return (uint32_t)lo;
}

/* spelling row value at symbol x: table cum (Laplace) + ws * doc prefix */
static void spell_encode(Model *m, Encoder *e, const uint8_t *raw, size_t n) {
    Doc *d = &m->doc;
    int prev = SPELL_END;
    for (size_t i = 0; i <= n; i++) {
        int b = i < n ? raw[i] : SPELL_END;
        const uint32_t *row = d->t->spell_cum + prev * 257;
        int64_t lo = b ? row[b - 1] : 0, hi = row[b], total = row[256];
        if (d->cfg->ws) {
            const int64_t *ds = d->doc_spell + prev * 257;
            int64_t s = 0, slo = 0, shi = 0;
            for (int x = 0; x < 257; x++) {
                s += ds[x];
                if (x == b - 1) slo = s;
                if (x == b) shi = s;
            }
            lo += d->cfg->ws * slo; hi += d->cfg->ws * shi; total += d->cfg->ws * s;
        }
        enc_encode(e, lo, hi, total);
        prev = b;
    }
}

static size_t spell_decode(Model *m, Decoder *dc, uint8_t **buf, size_t *cap) {
    Doc *d = &m->doc;
    int prev = SPELL_END;
    size_t n = 0;
    int64_t row[257];
    for (;;) {
        const uint32_t *tr = d->t->spell_cum + prev * 257;
        if (d->cfg->ws) {
            const int64_t *ds = d->doc_spell + prev * 257;
            int64_t s = 0;
            for (int x = 0; x < 257; x++) { s += ds[x]; row[x] = tr[x] + d->cfg->ws * s; }
        } else {
            for (int x = 0; x < 257; x++) row[x] = tr[x];
        }
        int64_t total = row[256];
        int64_t target = (int64_t)dec_target(dc, total);
        int b = 0;
        while (row[b] <= target) b++;
        dec_consume(dc, b ? row[b - 1] : 0, row[b], total);
        if (b == SPELL_END) return n;
        if (n == *cap) { *cap = *cap ? 2 * *cap : 64; *buf = xrealloc(*buf, *cap); }
        (*buf)[n++] = (uint8_t)b;
        prev = b;
    }
}

/* ---------------------------------------------------------------- tokenizer + UTF-8 */

/* strict UTF-8 check, same acceptance as Python's bytes.decode("utf-8") */
static int valid_utf8(const uint8_t *s, size_t n) {
    size_t i = 0;
    while (i < n) {
        uint8_t c = s[i];
        if (c < 0x80) { i++; continue; }
        size_t len;
        uint8_t lo = 0x80, hi = 0xBF;
        if (c >= 0xC2 && c <= 0xDF) len = 2;
        else if (c == 0xE0) { len = 3; lo = 0xA0; }
        else if (c >= 0xE1 && c <= 0xEC) len = 3;
        else if (c == 0xED) { len = 3; hi = 0x9F; }
        else if (c >= 0xEE && c <= 0xEF) len = 3;
        else if (c == 0xF0) { len = 4; lo = 0x90; }
        else if (c >= 0xF1 && c <= 0xF3) len = 4;
        else if (c == 0xF4) { len = 4; hi = 0x8F; }
        else return 0;
        if (i + len > n) return 0;
        if (s[i + 1] < lo || s[i + 1] > hi) return 0;
        for (size_t j = 2; j < len; j++)
            if (s[i + j] < 0x80 || s[i + j] > 0xBF) return 0;
        i += len;
    }
    return 1;
}
static inline int is_alpha(uint8_t c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
/* length of the token at s[i] (input is valid UTF-8): " ?[A-Za-z]+|[^A-Za-z]" */
static size_t next_token(const uint8_t *s, size_t n, size_t i) {
    size_t j = i;
    if (s[j] == ' ' && j + 1 < n && is_alpha(s[j + 1])) j++;
    if (is_alpha(s[j])) {
        while (j < n && is_alpha(s[j])) j++;
        return j - i;
    }
    uint8_t c = s[i];
    return c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
}

/* ---------------------------------------------------------------- container */

typedef struct { uint8_t *p; size_t n, cap; } Buf;
static void buf_put(Buf *b, const void *src, size_t n) {
    if (b->n + n > b->cap) { b->cap = (b->n + n) * 2 + 64; b->p = xrealloc(b->p, b->cap); }
    memcpy(b->p + b->n, src, n);
    b->n += n;
}
static void buf_byte(Buf *b, uint8_t v) { buf_put(b, &v, 1); }
static void buf_varint(Buf *b, uint64_t n) {
    for (;;) {
        buf_byte(b, (uint8_t)((n & 0x7F) | (n > 0x7F ? 0x80 : 0)));
        n >>= 7;
        if (!n) return;
    }
}

static void model_init(Model *m, const Table *t, const Config *cfg) {
    memset(m, 0, sizeof *m);
    doc_init(&m->doc, t, cfg);
}
static void model_free(Model *m) {
    doc_free(&m->doc);
    Level *L = &m->L;
    free(L->apos); free(L->adelta); free(L->apre); free(L->arem); free(L->rtmp);
    free(L->dsym); free(L->dcnt); free(L->dcum); free(L->dpos); free(L->dval);
    free(m->o0.epre); free(m->o0.tdsym); free(m->o0.tdpre);
}

static inline void push_hist(uint32_t *hist, uint32_t *th, int *hlen, uint32_t id, uint32_t V) {
    if (*hlen == 3) {
        hist[0] = hist[1]; hist[1] = hist[2];
        th[0] = th[1]; th[1] = th[2];
        *hlen = 2;
    }
    hist[*hlen] = id;
    th[*hlen] = id < V ? id : UNK;
    (*hlen)++;
}

static void compress(const Table *t, const Config *cfg, const uint8_t *data, size_t n, Buf *out) {
    out->n = 0;
    if (!valid_utf8(data, n)) {
        buf_byte(out, MODE_STORED);
        buf_put(out, data, n);
        return;
    }
    Model m;
    model_init(&m, t, cfg);
    Encoder e;
    enc_init(&e);
    uint32_t hist[3], th[3];
    int hlen = 0;
    uint64_t ntok = 0;
    for (size_t i = 0; i < n;) {
        size_t len = next_token(data, n, i);
        const uint8_t *tok = data + i;
        uint32_t w = table_lookup(t, tok, len);
        if (w == UNK && cfg->new_words) w = new_lookup(&m.doc, tok, len);
        encode_token(&m, &e, w, hist, th, hlen);
        if (w == UNK) spell_encode(&m, &e, tok, len);
        uint32_t hid = cfg->mode == MODE_STATIC && !cfg->ndoc && !cfg->w0 && !cfg->ws
                           ? w : doc_learn(&m.doc, tok, len, w, hist, hlen);
        push_hist(hist, th, &hlen, hid, (uint32_t)t->V);
        ntok++;
        i += len;
    }
    size_t code = enc_finish(&e);
    buf_byte(out, (uint8_t)cfg->mode);
    buf_put(out, t->fp, 4);
    buf_varint(out, ntok);
    buf_put(out, e.buf, code);
    free(e.buf);
    model_free(&m);
    if (out->n >= 1 + n) {
        out->n = 0;
        buf_byte(out, MODE_STORED);
        buf_put(out, data, n);
    }
}

static void decompress(const Table *t, int count_unk, const uint8_t *blob, size_t n, Buf *out) {
    out->n = 0;
    if (n == 0) die("empty input");
    if (blob[0] == MODE_STORED) { buf_put(out, blob + 1, n - 1); return; }
    Config cfg;
    if (blob[0] == MODE_STATIC) cfg = config_static();
    else if (blob[0] == MODE_ADAPTIVE) cfg = config_d3_sum_w32(count_unk);
    else die("unknown mode %d", blob[0]);
    if (n < 5 || memcmp(blob + 1, t->fp, 4))
        die("made with table %02x%02x%02x%02x, this table is %02x%02x%02x%02x", n > 1 ? blob[1] : 0,
            n > 2 ? blob[2] : 0, n > 3 ? blob[3] : 0, n > 4 ? blob[4] : 0, t->fp[0], t->fp[1], t->fp[2], t->fp[3]);
    uint64_t count = 0;
    size_t pos = 5;
    for (int shift = 0;; shift += 7) {
        if (pos >= n || shift > 63) die("truncated token count");
        uint8_t byte = blob[pos++];
        count |= (uint64_t)(byte & 0x7F) << shift;
        if (!(byte & 0x80)) break;
    }
    Decoder dc;
    dec_init(&dc, blob + pos, n - pos);
    Model m;
    model_init(&m, t, &cfg);
    uint32_t hist[3], th[3];
    int hlen = 0;
    uint8_t *sp = NULL;
    size_t spcap = 0;
    for (uint64_t i = 0; i < count; i++) {
        uint32_t w = decode_token(&m, &dc, hist, th, hlen);
        const uint8_t *tok;
        size_t len;
        if (w == UNK) {
            len = spell_decode(&m, &dc, &sp, &spcap);
            tok = sp;
        } else if (w < t->V) {
            tok = t->blob + t->voff[w];
            len = t->lens[w];
        } else {
            uint32_t j = w - (uint32_t)t->V;
            if (j >= m.doc.n_new) die("corrupt input: symbol %u", w);
            tok = m.doc.arena + m.doc.new_off[j];
            len = m.doc.new_len[j];
        }
        buf_put(out, tok, len);
        uint32_t hid = cfg.mode == MODE_STATIC ? w : doc_learn(&m.doc, out->p + out->n - len, len, w, hist, hlen);
        push_hist(hist, th, &hlen, hid, (uint32_t)t->V);
    }
    free(sp);
    model_free(&m);
}

/* ---------------------------------------------------------------- SHA-256 (fingerprint check) */

typedef struct { uint32_t h[8]; uint8_t buf[64]; size_t blen; uint64_t total; } Sha256;
static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
static void sha_block(Sha256 *s, const uint8_t *p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROR(w[i - 15], 7) ^ ROR(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = ROR(w[i - 2], 17) ^ ROR(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = s->h[0], b = s->h[1], c = s->h[2], d = s->h[3], e = s->h[4], f = s->h[5], g = s->h[6], h = s->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + (ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + w[i];
        uint32_t t2 = (ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d; s->h[4] += e; s->h[5] += f; s->h[6] += g; s->h[7] += h;
}
static void sha_init(Sha256 *s) {
    static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(s->h, iv, sizeof iv);
    s->blen = 0; s->total = 0;
}
static void sha_update(Sha256 *s, const uint8_t *p, size_t n) {
    s->total += n;
    if (s->blen) {
        while (n && s->blen < 64) { s->buf[s->blen++] = *p++; n--; }
        if (s->blen == 64) { sha_block(s, s->buf); s->blen = 0; }
    }
    while (n >= 64) { sha_block(s, p); p += 64; n -= 64; }
    while (n) { s->buf[s->blen++] = *p++; n--; }
}
static void sha_final(Sha256 *s, uint8_t out[32]) {
    uint64_t bits = s->total * 8;
    uint8_t pad = 0x80, z = 0;
    sha_update(s, &pad, 1);
    while (s->blen != 56) sha_update(s, &z, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; i++) len[i] = (uint8_t)(bits >> (56 - 8 * i));
    sha_update(s, len, 8);
    for (int i = 0; i < 8; i++) { out[4 * i] = s->h[i] >> 24; out[4 * i + 1] = s->h[i] >> 16; out[4 * i + 2] = s->h[i] >> 8; out[4 * i + 3] = s->h[i]; }
}
/* values of a section widened to little-endian int64 */
static void sha_section(Sha256 *s, const char *name, const Table *t, int sec, int elem) {
    sha_update(s, (const uint8_t *)name, strlen(name));
    const uint8_t *p = t->base + t->sec_off[sec];
    uint64_t n = t->sec_len[sec] / elem;
    int64_t tmp[4096];
    for (uint64_t i = 0; i < n;) {
        int m = 0;
        for (; m < 4096 && i < n; m++, i++) {
            if (elem == 1) tmp[m] = p[i];
            else if (elem == 4) { uint32_t v; memcpy(&v, p + 4 * i, 4); tmp[m] = v; }
            else memcpy(&tmp[m], p + 8 * i, 8);
        }
        sha_update(s, (const uint8_t *)tmp, (size_t)m * 8);  /* little-endian host */
    }
}
static int table_verify(const Table *t, uint8_t fp_out[4]) {
    Sha256 s;
    sha_init(&s);
    sha_section(&s, "vocab_blob", t, S_BLOB, 1);
    sha_section(&s, "vocab_lens", t, S_LENS, 4);
    sha_section(&s, "uni", t, S_UNI, 4);
    sha_section(&s, "spell", t, S_SPELL, 4);
    sha_section(&s, "c1_ctx", t, S_C1CTX, 8);
    sha_section(&s, "c1_start", t, S_C1START, 4);
    sha_section(&s, "c1_sym", t, S_C1SYM, 4);
    sha_section(&s, "c1_cnt", t, S_C1CNT, 4);
    sha_section(&s, "c2_ctx", t, S_C2CTX, 8);
    sha_section(&s, "c2_start", t, S_C2START, 4);
    sha_section(&s, "c2_sym", t, S_C2SYM, 4);
    sha_section(&s, "c2_cnt", t, S_C2CNT, 4);
    uint8_t dg[32];
    sha_final(&s, dg);
    memcpy(fp_out, dg, 4);
    return memcmp(dg, t->fp, 4) == 0;
}

/* ---------------------------------------------------------------- CLI */

static uint8_t *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) die("%s: %s", path, strerror(errno));
    size_t cap = 1 << 16, len = 0;
    uint8_t *p = xmalloc(cap);
    for (;;) {
        if (len == cap) { cap *= 2; p = xrealloc(p, cap); }
        size_t r = fread(p + len, 1, cap - len, f);
        len += r;
        if (r == 0) break;
    }
    fclose(f);
    *n = len;
    return p;
}
static void write_file(const char *path, const uint8_t *p, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) die("%s: %s", path, strerror(errno));
    if (n && fwrite(p, 1, n, f) != n) die("%s: write failed", path);
    fclose(f);
}

static void usage(void) {
    fprintf(stderr,
            "usage: steno compress [-s|-a] [-u] [-v] TABLE IN OUT [IN OUT ...]\n"
            "       steno decompress [-u] [-v] TABLE IN OUT [IN OUT ...]\n"
            "       steno info TABLE\n"
            "       steno verify TABLE\n");
    exit(2);
}

int main(int argc, char **argv) {
    if (argc < 3) usage();
    const char *cmd = argv[1];
    int adaptive = 0, count_unk = 0, verbose = 0, ai = 2;
    for (; ai < argc && argv[ai][0] == '-' && argv[ai][1]; ai++) {
        if (!strcmp(argv[ai], "-a")) adaptive = 1;
        else if (!strcmp(argv[ai], "-s")) adaptive = 0;
        else if (!strcmp(argv[ai], "-u")) count_unk = 1;
        else if (!strcmp(argv[ai], "-v")) verbose = 1;
        else usage();
    }
    if (ai >= argc) usage();
    double t0 = now();
    Table t;
    table_open(&t, argv[ai++]);
    double tload = now() - t0;
    if (!strcmp(cmd, "info")) {
        printf("fingerprint %02x%02x%02x%02x V %llu b %llu total0 %llu ctx1 %llu ent1 %llu ctx2 %llu ent2 %llu "
               "file %.1f MB, open+mmap %.3f ms\n", t.fp[0], t.fp[1], t.fp[2], t.fp[3],
               (unsigned long long)t.V, (unsigned long long)t.b, (unsigned long long)t.total0,
               (unsigned long long)t.nctx[1], (unsigned long long)t.nent[1], (unsigned long long)t.nctx[2],
               (unsigned long long)t.nent[2], t.size / 1e6, tload * 1e3);
        return 0;
    }
    if (!strcmp(cmd, "verify")) {
        uint8_t fp[4];
        int ok = table_verify(&t, fp);
        printf("stored %02x%02x%02x%02x computed %02x%02x%02x%02x %s (%.1f s)\n", t.fp[0], t.fp[1], t.fp[2], t.fp[3],
               fp[0], fp[1], fp[2], fp[3], ok ? "OK" : "MISMATCH", now() - t0);
        return ok ? 0 : 1;
    }
    int comp = !strcmp(cmd, "compress");
    if (!comp && strcmp(cmd, "decompress")) usage();
    if ((argc - ai) % 2 || ai >= argc) usage();
    Config cfg = adaptive ? config_d3_sum_w32(count_unk) : config_static();
    Buf out = {0};
    for (; ai < argc; ai += 2) {
        size_t n;
        uint8_t *in = read_file(argv[ai], &n);
        double t1 = now();
        if (comp) compress(&t, &cfg, in, n, &out);
        else decompress(&t, count_unk, in, n, &out);
        double dt = now() - t1;
        write_file(argv[ai + 1], out.p, out.n);
        if (verbose)
            fprintf(stderr, "%s %s %zu -> %zu bytes %.6f s\n", comp ? "compress" : "decompress", argv[ai], n, out.n, dt);
        free(in);
    }
    free(out.p);
    return 0;
}
