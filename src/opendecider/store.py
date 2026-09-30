"""Tier 2 state store: build a decision state per request from a large record collection
(docs/roadmap_designs.md §3, Tier 2).

One SQLite file, no service:
  chunks      one row per chunk (id, source, ts, key_path, text, n_tokens, pinned, is_json); rowid is the join key
  chunks_fts  FTS5 external-content index over (id, key_path, text); ranked with bm25(). If FTS5 is not compiled
              in (or use_fts=False) a vectorised numpy BM25 over the same fields is used instead.
  vecs        rowid -> unit-normalised float16 vector BLOB; searched by matmul against an in-memory cache.

Retrieval: every query text (question prompts, option texts, instructions) gets a lexical and a dense ranked
list; lists are fused with reciprocal rank fusion (k0=60), per text, then across texts by max (default) or sum. Selection: pinned chunks first, then greedy MMR
over the fused candidates until the token budget is spent. Assembly orders the non-pinned chunks by a STABLE key
(ts, then id), never by score, so rank cannot leak as a position signal and consecutive requests share prefixes.

Embeddings come from a pluggable `embed_fn(list[str]) -> (n, d) array`; this module has no model dependency.
`hash_embedder` is a deterministic placeholder (hashed bag of words) for tests and plumbing only.
"""
from __future__ import annotations

import json
import re
import sqlite3
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

RRF_K0 = 60
_TOK = re.compile(r"[^\W_]+")  # mirrors FTS5 unicode61: letters/digits are token chars, '_' and punctuation split

EmbedFn = Callable[[list], np.ndarray]
CountFn = Callable[[str], int]


def default_count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def record_text(rec: Any) -> str:
    """Same serialisation as formatting.state_text uses for JSON states."""
    if isinstance(rec, str):
        return rec
    return json.dumps(rec, ensure_ascii=False, separators=(",", ": "))


def tokenize(text: str) -> list:
    return _TOK.findall(text.lower())


def fts5_available() -> bool:
    try:
        c = sqlite3.connect(":memory:")
        c.execute("CREATE VIRTUAL TABLE t USING fts5(a)")
        c.close()
        return True
    except sqlite3.OperationalError:
        return False


def query_terms(text: str) -> list:
    """[(term, tokens)]: every word token, plus the adjacency phrase of multi-token words (C-48213 -> "c 48213")."""
    terms, seen = [], set()
    for w in text.split():
        toks = tokenize(w)
        for t in ([" ".join(toks)] if len(toks) > 1 else []) + toks:
            if t not in seen:
                seen.add(t)
                terms.append((t, t.split()))
    return terms


def prune_terms(terms: list, df: dict, max_df: float) -> list:
    """Drop terms whose every token has document frequency > max_df (near-zero BM25 idf, but they make OR queries
    touch most of the index). If everything would go, keep the rarest term."""
    keep = [t for t in terms if any(df.get(x, 0) <= max_df for x in t[1])]
    if not keep and terms:
        keep = [min(terms, key=lambda t: min(df.get(x, 0) for x in t[1]))]
    return keep


def fts_query(text: str, max_terms: int = 64, df: dict | None = None, max_df: float | None = None) -> str | None:
    """Sanitise free text into a safe FTS5 MATCH expression: every term is a quoted phrase of word tokens, OR'd.
    Quotes, colons, parens, '*', '-', and bare AND/OR/NOT/NEAR never reach the FTS5 parser unquoted."""
    terms = query_terms(text)
    if df is not None and max_df is not None:
        terms = prune_terms(terms, df, max_df)
    if not terms:
        return None
    return " OR ".join('"' + t + '"' for t, _ in terms[:max_terms])


# ---- placeholder embedder ------------------------------------------------------------------------------

_HASH_CACHE: dict = {}


def _hash_feat(f: str, dim: int):
    h = _HASH_CACHE.get((f, dim))
    if h is None:
        x = zlib.crc32(f.encode("utf-8"))
        h = _HASH_CACHE[(f, dim)] = (x % dim, 1.0 if (x >> 31) & 1 else -1.0)
    return h


def hash_embedder(dim: int = 256, prefix: int = 5) -> EmbedFn:
    """Deterministic hashed bag of words + word prefixes (a crude stemmer). PLACEHOLDER ONLY: no semantics beyond
    shared stems; real dense vectors come from the backbone later."""
    def embed(texts):
        rows, cols, sign = [], [], []
        for i, t in enumerate(texts):
            for w in tokenize(t):
                for f in (w, "p:" + w[:prefix]) if len(w) > prefix else (w, "p:" + w):
                    c, s = _hash_feat(f, dim)
                    rows.append(i); cols.append(c); sign.append(s)
        out = np.zeros((len(texts), dim), np.float32)
        np.add.at(out, (np.asarray(rows, np.int64), np.asarray(cols, np.int64)), np.asarray(sign, np.float32))
        return out
    return embed


# ---- results -------------------------------------------------------------------------------------------

@dataclass
class SearchResult:
    ids: list                 # fused order, best first
    scores: np.ndarray        # RRF scores (descending)
    bm25_rank: np.ndarray     # best 1-based rank over lexical lists; 0 = not retrieved lexically
    dense_rank: np.ndarray    # same for dense lists
    rowids: np.ndarray = field(repr=False, default=None)


@dataclass
class Selection:
    ids: list                 # pinned first (pinned order), then MMR pick order
    n_pinned: int
    tokens: int               # total n_tokens of selected chunks (pinned may exceed the budget on their own)
    n_candidates: int


def _norm_queries(query_texts) -> tuple:
    texts, w = [], []
    for q in query_texts:
        t, wt = (q, 1.0) if isinstance(q, str) else (q[0], float(q[1]))
        if t and t.strip():
            texts.append(t); w.append(wt)
    return texts, np.asarray(w, np.float64)


class StateStore:
    """SQLite-backed chunk store with hybrid (BM25 + dense) retrieval, MMR selection and stable assembly.

    Chunk dict: {id, text, source?, ts?, key_path?, pinned?, is_json?}. `pinned` is 0 (not pinned) or a positive
    priority; pinned chunks are always selected, ordered by (pinned, ts, id).
    """

    def __init__(self, path: str = ":memory:", embed_fn: EmbedFn | None = None,
                 count_tokens: CountFn | None = None, use_fts: bool | None = None):
        self.path = path
        self.embed_fn = embed_fn
        self.count_tokens = count_tokens or default_count_tokens
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL" if path != ":memory:" else "PRAGMA journal_mode=MEMORY")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, source TEXT, ts REAL, key_path TEXT,
                text TEXT NOT NULL, n_tokens INTEGER NOT NULL, pinned INTEGER NOT NULL DEFAULT 0,
                is_json INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX IF NOT EXISTS chunks_pinned ON chunks(pinned) WHERE pinned > 0;
            CREATE TABLE IF NOT EXISTS vecs(rowid INTEGER PRIMARY KEY, v BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);""")
        has_fts = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='chunks_fts'").fetchone() is not None
        self.use_fts = (fts5_available() if use_fts is None else bool(use_fts)) and fts5_available()
        if has_fts and not self.use_fts:  # fallback writes would leave it stale; rebuilt if reopened with FTS5
            self.db.execute("DROP TABLE IF EXISTS chunks_vocab")
            self.db.execute("DROP TABLE chunks_fts")
        if self.use_fts and not has_fts:
            self.db.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(id, key_path, text, "
                            "content='chunks', content_rowid='rowid')")
            self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        if self.use_fts:
            self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vocab USING fts5vocab(chunks_fts, 'row')")
        self.db.commit()
        self.fts_weights = (4.0, 2.0, 1.0)  # bm25 column weights: id, key_path, text
        self.max_df, self.prune_min_n = 0.1, 1000  # query-term pruning (df fraction), only on stores >= min_n chunks
        self._idx = None    # in-memory arrays (rowid-sorted): rowid, id, ts, n_tokens, pinned, vec, has_vec
        self._bm25 = None   # numpy BM25 fallback index

    # ---- writes ----------------------------------------------------------------------------------------

    @property
    def dim(self) -> int | None:
        r = self.db.execute("SELECT v FROM meta WHERE k='dim'").fetchone()
        return int(r[0]) if r else None

    def __len__(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def add(self, chunks: Iterable[dict], embed_fn: EmbedFn | None = None, count_tokens: CountFn | None = None,
            batch_size: int = 4096) -> int:
        """Upsert chunks (same id replaces text/metadata/vector; rowid is kept). Returns the number written."""
        embed_fn = embed_fn or self.embed_fn
        count = count_tokens or self.count_tokens
        chunks = list(chunks)
        for s in range(0, len(chunks), batch_size):
            self._add_batch(chunks[s:s + batch_size], embed_fn, count)
        self.db.commit()
        self._idx = self._bm25 = None
        return len(chunks)

    def _add_batch(self, chunks, embed_fn, count):
        rows = []
        for c in chunks:
            if "id" not in c or "text" not in c:
                raise ValueError("chunk needs 'id' and 'text'")
            t = str(c["text"])
            ts = c.get("ts")
            rows.append((str(c["id"]), c.get("source"), None if ts is None else float(ts), c.get("key_path"), t,
                         int(count(t)), int(c.get("pinned") or 0), int(bool(c.get("is_json", False)))))
        ids = [r[0] for r in rows]
        if len(set(ids)) != len(ids):  # last write wins within a batch
            last = {r[0]: r for r in rows}
            rows = list(last.values()); ids = list(last.keys())
        cur = self.db.cursor()
        if self.use_fts:  # external content: remove old index entries before the row changes
            old = cur.execute("SELECT rowid, id, key_path, text FROM chunks WHERE id IN "
                              "(SELECT value FROM json_each(?))", (json.dumps(ids),)).fetchall()
            cur.executemany("INSERT INTO chunks_fts(chunks_fts, rowid, id, key_path, text) "
                            "VALUES('delete', ?, ?, ?, ?)", old)
        cur.executemany(
            "INSERT INTO chunks(id, source, ts, key_path, text, n_tokens, pinned, is_json) VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET source=excluded.source, ts=excluded.ts, key_path=excluded.key_path, "
            "text=excluded.text, n_tokens=excluded.n_tokens, pinned=excluded.pinned, is_json=excluded.is_json",
            rows)
        rid = dict(cur.execute("SELECT id, rowid FROM chunks WHERE id IN (SELECT value FROM json_each(?))",
                               (json.dumps(ids),)).fetchall())
        rowids = [rid[i] for i in ids]
        if self.use_fts:
            cur.executemany("INSERT INTO chunks_fts(rowid, id, key_path, text) VALUES(?,?,?,?)",
                            [(rowids[j], r[0], r[3], r[4]) for j, r in enumerate(rows)])
        if embed_fn is not None:
            v = np.asarray(embed_fn([r[4] for r in rows]), np.float32)
            if v.ndim != 2 or v.shape[0] != len(rows):
                raise ValueError(f"embed_fn returned {v.shape} for {len(rows)} texts")
            d = self.dim
            if d is None:
                cur.execute("INSERT INTO meta(k, v) VALUES('dim', ?)", (str(v.shape[1]),))
            elif d != v.shape[1]:
                raise ValueError(f"embedding dim {v.shape[1]} != store dim {d}")
            v = (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)).astype(np.float16)
            cur.executemany("INSERT OR REPLACE INTO vecs(rowid, v) VALUES(?, ?)",
                            [(rowids[j], v[j].tobytes()) for j in range(len(rows))])
        else:  # text changed without a new vector: drop the stale one
            cur.executemany("DELETE FROM vecs WHERE rowid=?", [(r,) for r in rowids])

    def add_records(self, records: Sequence[Any], prefix: str = "records", start: int = 0, id_key: str | None = "id",
                    ts_key: str | None = "ts", source: str | None = None, pinned: int = 0,
                    embed_fn: EmbedFn | None = None, count_tokens: CountFn | None = None) -> list:
        """One chunk per record. key_path = f"{prefix}[{start+i}]"; text = compact JSON (formatting.state_text's
        serialisation) for non-strings, the string itself otherwise. id = str(rec[id_key]) when the record is a dict
        carrying it, else the key_path. ts = rec[ts_key] when numeric. Returns the chunk ids."""
        chunks = []
        for i, r in enumerate(records):
            kp = f"{prefix}[{start + i}]"
            isd = isinstance(r, dict)
            cid = str(r[id_key]) if isd and id_key and id_key in r else kp
            ts = r.get(ts_key) if isd and ts_key else None
            chunks.append({"id": cid, "text": record_text(r), "key_path": kp, "source": source, "pinned": pinned,
                           "ts": ts if isinstance(ts, (int, float)) and not isinstance(ts, bool) else None,
                           "is_json": not isinstance(r, str)})
        self.add(chunks, embed_fn=embed_fn, count_tokens=count_tokens)
        return [c["id"] for c in chunks]

    # ---- in-memory index -------------------------------------------------------------------------------

    def _index(self) -> dict:
        if self._idx is not None:
            return self._idx
        rows = self.db.execute("SELECT rowid, id, ts, n_tokens, pinned FROM chunks ORDER BY rowid").fetchall()
        rowid, ids, ts, ntok, pin = (list(c) for c in zip(*rows)) if rows else ([], [], [], [], [])
        idx = {"rowid": np.asarray(rowid, np.int64), "id": np.asarray(ids, object),
               "ts": np.asarray([np.nan if t is None else t for t in ts], np.float64),
               "ntok": np.asarray(ntok, np.int64), "pinned": np.asarray(pin, np.int64)}
        n, d = len(rowid), self.dim
        vec, has = None, np.zeros(n, bool)
        if d is not None and n:
            vr = self.db.execute("SELECT rowid, v FROM vecs ORDER BY rowid").fetchall()
            if vr:
                pos = np.searchsorted(idx["rowid"], np.asarray([r[0] for r in vr], np.int64))
                vec = np.zeros((n, d), np.float16)
                vec[pos] = np.frombuffer(b"".join(r[1] for r in vr), np.float16).reshape(len(vr), d)
                has[pos] = True
        idx["vec"], idx["has_vec"] = vec, has
        self._idx = idx
        return idx

    def _pos(self, rowids: np.ndarray) -> np.ndarray:
        return np.searchsorted(self._index()["rowid"], rowids)

    def _pos_of_ids(self, ids: Sequence[str]) -> np.ndarray:
        """Index positions of chunk ids (unknown ids raise)."""
        if not len(ids):
            return np.zeros(0, np.int64)
        got = dict(self.db.execute("SELECT id, rowid FROM chunks WHERE id IN (SELECT value FROM json_each(?))",
                                   (json.dumps(list(ids)),)).fetchall())
        miss = [i for i in ids if i not in got]
        if miss:
            raise KeyError(f"unknown chunk ids: {miss[:5]}")
        return self._pos(np.asarray([got[i] for i in ids], np.int64))

    # ---- lexical ---------------------------------------------------------------------------------------

    def _max_df(self) -> float | None:
        n = len(self._index()["rowid"])
        return None if self.max_df is None or n < self.prune_min_n else self.max_df * n

    def _lexical(self, texts: list, n: int) -> list:
        """Per query text: array of index positions, best first."""
        if not self.use_fts:
            return [self._bm25_np(t, n) for t in texts]
        w, thr, df = self.fts_weights, self._max_df(), None
        if thr is not None:
            toks = sorted({x for t in texts for x in tokenize(t)})
            df = dict(self.db.execute("SELECT term, doc FROM chunks_vocab WHERE term IN "
                                      "(SELECT value FROM json_each(?))", (json.dumps(toks),)).fetchall())
        sql = (f"SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
               f"ORDER BY bm25(chunks_fts, {w[0]}, {w[1]}, {w[2]}) LIMIT ?")
        out = []
        for t in texts:
            q = fts_query(t, df=df, max_df=thr)
            r = self.db.execute(sql, (q, n)).fetchall() if q else []
            out.append(self._pos(np.asarray([x[0] for x in r], np.int64)))
        return out

    def _bm25_index(self) -> dict:
        if self._bm25 is not None:
            return self._bm25
        idx = self._index()
        rows = self.db.execute("SELECT id, COALESCE(key_path, ''), text FROM chunks ORDER BY rowid").fetchall()
        vocab: dict = {}
        term, doc, fw, dl = [], [], [], np.zeros(len(rows), np.float64)
        cw = self.fts_weights  # column weights, as in FTS5 bm25(): weighted tf over id, key_path, text
        for i, r in enumerate(rows):  # index build, not a request path
            for c in range(3):
                toks = tokenize(r[c])
                dl[i] += len(toks)
                term.extend(vocab.setdefault(t, len(vocab)) for t in toks)
                doc.extend([i] * len(toks)); fw.extend([cw[c]] * len(toks))
        n1 = max(len(rows), 1)
        term, doc = np.asarray(term, np.int64), np.asarray(doc, np.int64)
        key, inv = np.unique(term * n1 + doc, return_inverse=True)  # sorted by (term, doc)
        tf = np.bincount(inv.ravel(), weights=np.asarray(fw, np.float64), minlength=len(key))
        kt, kd = key // n1, key % n1
        ptr = np.searchsorted(kt, np.arange(len(vocab) + 1))
        n = len(rows)
        df = np.diff(ptr).astype(np.float64)
        self._bm25 = {"vocab": vocab, "ptr": ptr, "doc": kd, "tf": tf, "dl": dl,
                      "avgdl": dl.mean() if n else 1.0,
                      "idf": np.maximum(np.log((n - df + 0.5) / (df + 0.5)), 1e-6), "n": n}
        assert n == len(idx["rowid"])
        return self._bm25

    def _bm25_np(self, text: str, n: int, k1: float = 1.2, b: float = 0.75) -> np.ndarray:
        ix = self._bm25_index()
        toks = sorted(set(tokenize(text)))
        thr = self._max_df()
        if thr is not None:
            df = {t: int(ix["ptr"][ix["vocab"][t] + 1] - ix["ptr"][ix["vocab"][t]]) for t in toks if t in ix["vocab"]}
            toks = [t for t, _ in prune_terms([(t, [t]) for t in toks], df, thr)]
        tids = np.unique([ix["vocab"][t] for t in toks if t in ix["vocab"]]).astype(np.int64)
        if not len(tids):
            return np.zeros(0, np.int64)
        lo, hi = ix["ptr"][tids], ix["ptr"][tids + 1]
        lens = hi - lo
        sel = np.repeat(lo - np.cumsum(np.r_[0, lens[:-1]]), lens) + np.arange(lens.sum())  # flat posting slices
        d, tf = ix["doc"][sel], ix["tf"][sel]
        w = np.repeat(ix["idf"][tids], lens) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * ix["dl"][d] / ix["avgdl"]))
        ud, inv = np.unique(d, return_inverse=True)
        s = np.bincount(inv, weights=w)
        top = np.argsort(-s, kind="stable")[:n]
        return ud[top]

    # ---- dense -----------------------------------------------------------------------------------------

    def _dense(self, texts: list, n: int, embed_fn: EmbedFn | None, block: int = 65536) -> list:
        idx = self._index()
        if embed_fn is None or idx["vec"] is None or not idx["has_vec"].any():
            return [np.zeros(0, np.int64) for _ in texts]
        q = np.asarray(embed_fn(texts), np.float32)
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        V, N = idx["vec"], len(idx["rowid"])
        sims = np.empty((len(texts), N), np.float32)
        for s in range(0, N, block):  # blockwise fp16 -> fp32 keeps peak memory bounded
            sims[:, s:s + block] = q @ V[s:s + block].astype(np.float32).T
        sims[:, ~idx["has_vec"]] = -np.inf
        k = min(n, int(idx["has_vec"].sum()))
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, part, 1), axis=1, kind="stable")
        top = np.take_along_axis(part, order, 1)
        return list(top)

    # ---- fusion ----------------------------------------------------------------------------------------

    def search(self, query_texts: Sequence, k: int = 50, embed_fn: EmbedFn | None = None,
               weights: dict | None = None, n_per_list: int = 100, k0: int = RRF_K0,
               union: str = "max") -> SearchResult:
        """Hybrid search over the UNION of query texts. query_texts: str or (str, weight). Each text yields a BM25
        list and a dense list (top n_per_list), fused per text by RRF: s_t = sum_r w_t * w_r / (k0 + rank_r).
        Across texts: union="sum" adds s_t (plain RRF over every list; rewards chunks hit by several texts, but
        deep co-occurrences of common terms add up too); union="max" keeps each chunk's best s_t (each text's
        top hits compete on equal terms; default: on eval/store_recall.py generic option lists such as city names
        otherwise outrank the one record the question names). weights: {"bm25": 1.0, "dense": 1.0}; weight 0 skips a retriever."""
        if union not in ("sum", "max"):
            raise ValueError("union must be 'sum' or 'max'")
        embed_fn = embed_fn or self.embed_fn
        w = {"bm25": 1.0, "dense": 1.0, **(weights or {})}
        texts, tw = _norm_queries(query_texts)
        lists = []  # (positions, weight, retriever, text index)
        if texts and w["bm25"] > 0:
            lists += [(p, tw[j] * w["bm25"], 0, j) for j, p in enumerate(self._lexical(texts, n_per_list))]
        if texts and w["dense"] > 0:
            lists += [(p, tw[j] * w["dense"], 1, j) for j, p in enumerate(self._dense(texts, n_per_list, embed_fn))]
        idx = self._index()
        lens = np.asarray([len(x[0]) for x in lists], np.int64)
        if not lens.sum():
            e = np.zeros(0)
            return SearchResult([], e, e.astype(np.int64), e.astype(np.int64), e.astype(np.int64))
        pos = np.concatenate([x[0] for x in lists]).astype(np.int64)
        rank = np.concatenate([np.arange(1, m + 1) for m in lens])
        rr = np.repeat(np.asarray([x[1] for x in lists], np.float64), lens) / (k0 + rank)
        kind = np.repeat(np.asarray([x[2] for x in lists], np.int64), lens)
        u, inv = np.unique(pos, return_inverse=True)
        if union == "sum":
            score = np.bincount(inv, weights=rr, minlength=len(u))
        else:
            tix = np.repeat(np.asarray([x[3] for x in lists], np.int64), lens)
            ut, tinv = np.unique(inv * len(texts) + tix, return_inverse=True)  # (chunk, text) pairs
            per = np.bincount(tinv, weights=rr, minlength=len(ut))
            score = np.zeros(len(u))
            np.maximum.at(score, ut // len(texts), per)
        best = np.full((2, len(u)), np.iinfo(np.int64).max, np.int64)
        np.minimum.at(best, (kind, inv), rank)
        best[best == np.iinfo(np.int64).max] = 0
        # ties broken by (ts, id) so the candidate order is deterministic
        o = np.lexsort((idx["id"][u].astype(str), idx["ts"][u], -score))[:k]
        return SearchResult(list(idx["id"][u[o]]), score[o], best[0, o], best[1, o], idx["rowid"][u[o]])

    # ---- selection -------------------------------------------------------------------------------------

    def pinned_ids(self) -> list:
        return [r[0] for r in self.db.execute(
            "SELECT id FROM chunks WHERE pinned > 0 ORDER BY pinned, ts IS NULL, ts, id")]

    def select(self, query_texts: Sequence, token_budget: int, count_tokens: CountFn | None = None,
               mmr_lambda: float = 0.7, embed_fn: EmbedFn | None = None, n_candidates: int = 400,
               **search_kw) -> Selection:
        """Pinned chunks (always, even past the budget), then greedy MMR over fused candidates:
        argmax lambda * rel - (1 - lambda) * max_cos_to_selected, rel = fused score / max fused score, skipping
        chunks that no longer fit. `count_tokens` overrides the stored n_tokens (recounted for candidates only)."""
        idx = self._index()
        pin_ids = self.pinned_ids()
        ppos = self._pos_of_ids(pin_ids)
        res = self.search(query_texts, k=n_candidates, embed_fn=embed_fn, **search_kw)
        cpos = self._pos(res.rowids)
        keep = ~np.isin(cpos, ppos)
        cpos, rel = cpos[keep], res.scores[keep]
        ntok_p, ntok_c = idx["ntok"][ppos], idx["ntok"][cpos]
        if count_tokens is not None:
            texts = self._texts(np.concatenate([ppos, cpos]))
            nt = np.asarray([count_tokens(t) for t in texts], np.int64)
            ntok_p, ntok_c = nt[:len(ppos)], nt[len(ppos):]
        used = int(ntok_p.sum())
        rel = rel / rel.max() if len(rel) and rel.max() > 0 else rel
        red = np.zeros(len(cpos), np.float64)
        V, has = idx["vec"], idx["has_vec"]
        C = None
        if V is not None and len(cpos):
            C = V[cpos].astype(np.float32) * has[cpos, None]
            P = V[ppos].astype(np.float32) * has[ppos, None]
            if len(ppos):
                red = np.maximum(red, (C @ P.T).max(1))
        alive = np.ones(len(cpos), bool)
        chosen = []
        while True:
            alive &= ntok_c <= token_budget - used
            if not alive.any():
                break
            m = np.where(alive, mmr_lambda * rel - (1 - mmr_lambda) * red, -np.inf)
            j = int(np.argmax(m))
            chosen.append(j); alive[j] = False; used += int(ntok_c[j])
            if C is not None:
                red = np.maximum(red, C @ C[j])
        ids = pin_ids + list(idx["id"][cpos[chosen]]) if chosen else list(pin_ids)
        return Selection(ids, len(pin_ids), used, len(cpos))

    # ---- assembly --------------------------------------------------------------------------------------

    def _texts(self, pos: np.ndarray) -> list:
        return [t for t, _ in self._rows(pos, "text, is_json")]

    def _rows(self, pos: np.ndarray, cols: str) -> list:
        rid = self._index()["rowid"][pos]
        got = {r[0]: r[1:] for r in self.db.execute(
            f"SELECT rowid, {cols} FROM chunks WHERE rowid IN (SELECT value FROM json_each(?))",
            (json.dumps(rid.tolist()),))}
        return [got[int(r)] for r in rid]

    def _order(self, selected_ids: Sequence[str], order: str) -> tuple:
        idx = self._index()
        pos = self._pos_of_ids(list(dict.fromkeys(selected_ids)))
        pin = idx["pinned"][pos]
        ids, ts = idx["id"][pos].astype(str), idx["ts"][pos]
        if order == "ts":
            o = np.lexsort((ids, ts, pin == 0))          # pinned block first; nan ts sorts last
        elif order == "id":
            o = np.lexsort((ids, pin == 0))
        else:
            raise ValueError("order must be 'ts' or 'id'")
        pos = pos[o]
        npin = int((pin > 0).sum())
        # pinned block in pinned order (priority, ts, id)
        pp = pos[:npin]
        pp = pp[np.lexsort((idx["id"][pp].astype(str), idx["ts"][pp], idx["pinned"][pp]))]
        return np.concatenate([pp, pos[npin:]]), npin

    def assemble(self, selected_ids: Sequence[str], order: str = "ts") -> tuple:
        """-> (texts, n_pinned). Pinned chunks first (pinned order), then the rest by a STABLE key: (ts, id) for
        order="ts" (missing ts last), id for order="id". Never by retrieval score: the same set of ids always
        assembles identically whatever query produced it."""
        pos, npin = self._order(selected_ids, order)
        return self._texts(pos), npin

    def as_state(self, selected_ids: Sequence[str], order: str = "ts", fmt: str = "dict") -> Any:
        """JSON-serialisable state. fmt="dict": {"pinned": [...], "records": [...]}; fmt="list": one flat list
        (pinned first). Chunks added as JSON (add_records, or is_json=True) are decoded back to objects, so
        formatting.state_text re-serialises them byte-identically; others stay strings."""
        pos, npin = self._order(selected_ids, order)
        items = [json.loads(t) if j else t for t, j in self._rows(pos, "text, is_json")]
        if fmt == "list":
            return items
        if fmt != "dict":
            raise ValueError("fmt must be 'dict' or 'list'")
        return {"pinned": items[:npin], "records": items[npin:]}

    def build_state(self, query_texts: Sequence, token_budget: int, fmt: str = "dict", order: str = "ts",
                    **select_kw) -> tuple:
        """select + as_state. -> (state, Selection)."""
        sel = self.select(query_texts, token_budget, **select_kw)
        return self.as_state(sel.ids, order=order, fmt=fmt), sel

    def close(self):
        self.db.close()
