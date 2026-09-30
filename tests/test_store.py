"""Tier 2 store: lexical (FTS5 and numpy fallback), dense, RRF, MMR, budget, pinning, stable assembly."""
import json

import numpy as np
import pytest

from opendecider.formatting import state_text
from opendecider.store import StateStore, fts5_available, fts_query, hash_embedder, record_text

EMB = hash_embedder(dim=512)
MODES = [pytest.param(True, id="fts5", marks=pytest.mark.skipif(not fts5_available(), reason="no FTS5")),
         pytest.param(False, id="numpy_bm25")]

RECS = [
    {"id": "ORD-1001", "ts": 5, "customer": "Nadia", "status": "shipped", "note": "left at the front door"},
    {"id": "ORD-1002", "ts": 1, "customer": "Mateo", "status": "pending", "note": "awaiting payment"},
    {"id": "ORD-1003", "ts": 3, "customer": "Wen", "status": "cancelled", "note": "customer requested a refund"},
    {"id": "ORD-1004", "ts": 2, "customer": "Hugo", "status": "delivered", "note": "signed by recipient"},
    {"id": "ORD-1005", "ts": 4, "customer": "Ines", "status": "returned", "note": "damaged packaging"},
]


def store(use_fts, emb=EMB, path=":memory:"):
    s = StateStore(path, embed_fn=emb, use_fts=use_fts)
    s.add_records(RECS, prefix="orders")
    return s


def test_fts5_detection_and_forced_fallback():
    s = StateStore(use_fts=False)
    assert not s.use_fts
    assert s.db.execute("SELECT 1 FROM sqlite_master WHERE name='chunks_fts'").fetchone() is None
    if fts5_available():
        assert StateStore().use_fts


@pytest.mark.parametrize("use_fts", MODES)
def test_lexical_exact_id_and_key_path(use_fts):
    s = store(use_fts, emb=None)
    assert s.search(["ORD-1004"], k=1, weights={"dense": 0}).ids == ["ORD-1004"]
    assert s.search(["orders[2]"], k=1, weights={"dense": 0}).ids == ["ORD-1003"]


@pytest.mark.parametrize("use_fts", MODES)
@pytest.mark.parametrize("q", ['say "hi', 'status: shipped', '(Nadia', 'ORD-1001)', 'front-door', 'ship*', '"',
                               'AND OR NOT NEAR', '::()**--', '', 'x" OR "y', "col:a*b^c+d{e}"])
def test_fts_query_sanitisation(use_fts, q):
    s = store(use_fts, emb=None)
    r = s.search([q], k=3, weights={"dense": 0})  # must not raise
    assert len(r.ids) <= 3
    fq = fts_query(q)
    assert fq is None or all(t.startswith('"') and t.endswith('"') for t in fq.split(" OR "))


def test_fts_query_phrase_for_ids():
    assert '"ord 1001"' in fts_query("find ORD-1001 now").split(" OR ")


@pytest.mark.parametrize("use_fts", MODES)
def test_dense_paraphrase(use_fts):
    s = store(use_fts)
    # "refunding"/"customers"/"requests" share only stems with ORD-1003; no exact token overlap on "refund"
    r = s.search(["refunding requests"], k=1, weights={"bm25": 0})
    assert r.ids == ["ORD-1003"] and r.dense_rank[0] == 1 and r.bm25_rank[0] == 0
    lex = s.search(["refunding requests"], k=5, weights={"dense": 0})
    assert "ORD-1003" not in lex.ids


def test_rrf_fusion_order():
    s = StateStore(use_fts=True if fts5_available() else False)
    s.add([{"id": x, "text": x} for x in "abcd"])
    L = [s._pos_of_ids(list("abc")), s._pos_of_ids(list("cbd"))]
    s._lexical = lambda texts, n: [L[0]]
    s._dense = lambda texts, n, e: [L[1]]
    r = s.search(["q"], k=4)
    # a: 1/61; b: 1/62+1/62; c: 1/63+1/61; d: 1/63
    exp = {"a": 1 / 61, "b": 2 / 62, "c": 1 / 63 + 1 / 61, "d": 1 / 63}
    assert r.ids == sorted(exp, key=lambda x: -exp[x]) == ["c", "b", "a", "d"]
    np.testing.assert_allclose(r.scores, sorted(exp.values(), reverse=True))
    assert list(r.bm25_rank) == [3, 2, 1, 0] and list(r.dense_rank) == [1, 2, 0, 3]
    # retriever weight scales its lists
    r2 = s.search(["q"], k=4, weights={"dense": 0.0})
    assert r2.ids == ["a", "b", "c"]


def test_union_sum_vs_max():
    s = StateStore(use_fts=True if fts5_available() else False)
    s.add([{"id": x, "text": x} for x in "abc"])
    P = {"q1": s._pos_of_ids(["a", "b"]), "q2": s._pos_of_ids(["c", "b"])}
    s._lexical = lambda texts, n: [P[t] for t in texts]
    s._dense = lambda texts, n, e: [np.zeros(0, np.int64) for _ in texts]
    # b is 2nd in both lists: sum -> 2/62 beats a, c (1/61); max -> 1/62 loses to both top hits
    assert s.search(["q1", "q2"], k=3, union="sum").ids[0] == "b"
    r = s.search(["q1", "q2"], k=3, union="max")
    assert r.ids[2] == "b" and set(r.ids[:2]) == {"a", "c"}
    np.testing.assert_allclose(r.scores, [1 / 61, 1 / 61, 1 / 62])
    # per-text weights scale that text's lists
    assert s.search([("q1", 0.5), "q2"], k=3, union="max").ids[0] == "c"


def test_union_over_query_texts():
    s = store(True if fts5_available() else False)
    r = s.search(["ORD-1002", "ORD-1005"], k=2, weights={"dense": 0})
    assert set(r.ids) == {"ORD-1002", "ORD-1005"}


def test_mmr_avoids_near_duplicates():
    s = StateStore(embed_fn=EMB)
    dup = "invoice 77 refund approved for customer Nadia"
    s.add([{"id": "d1", "text": dup}, {"id": "d2", "text": dup + " ok"}, {"id": "d3", "text": dup + " yes"},
           {"id": "o1", "text": "invoice 77 shipping address changed to Lyon"}])
    q = ["invoice 77 refund Nadia"]
    greedy = s.select(q, token_budget=24, mmr_lambda=1.0).ids
    diverse = s.select(q, token_budget=24, mmr_lambda=0.3).ids
    assert sum(i.startswith("d") for i in greedy) == 2
    assert "o1" in diverse and sum(i.startswith("d") for i in diverse) == 1


@pytest.mark.parametrize("use_fts", MODES)
def test_budget_and_pinned(use_fts):
    s = store(use_fts)
    s.add([{"id": "schema", "text": "Orders have id, customer, status, note.", "pinned": 1, "ts": 99},
           {"id": "policy", "text": "Refunds within 30 days.", "pinned": 2}])
    for budget in (0, 5, 30, 60, 10_000):
        sel = s.select(["refund status of ORD-1003"], token_budget=budget)
        assert sel.ids[:2] == ["schema", "policy"] and sel.n_pinned == 2
        ntok = dict(s.db.execute("SELECT id, n_tokens FROM chunks"))
        pin_tok = ntok["schema"] + ntok["policy"]
        assert sel.tokens == sum(ntok[i] for i in sel.ids)
        assert sel.tokens <= max(budget, pin_tok)
    assert len(s.select(["refund"], token_budget=10_000).ids) == len(RECS) + 2
    # custom token counter overrides the stored counts
    one = s.select(["refund"], token_budget=3, count_tokens=lambda t: 1)
    assert len(one.ids) == 3 and one.tokens == 3


@pytest.mark.parametrize("use_fts", MODES)
def test_assemble_stable_and_score_independent(use_fts):
    s = store(use_fts)
    s.add([{"id": "schema", "text": "SCHEMA", "pinned": 1}, {"id": "zz-no-ts", "text": "no timestamp"}])
    a = s.select(["ORD-1001 shipped"], 10_000).ids
    b = s.select(["damaged packaging returned"], 10_000).ids
    assert a != b and set(a) == set(b)
    ta, npa = s.assemble(a)
    tb, npb = s.assemble(list(reversed(b)))
    assert ta == tb and npa == npb == 1 and ta[0] == "SCHEMA"
    by_ts = sorted(RECS, key=lambda r: r["ts"])
    assert ta[1:] == [record_text(r) for r in by_ts] + ["no timestamp"]  # ts ascending, missing ts last
    ti, _ = s.assemble(a, order="id")
    assert ti[1:] == [record_text(r) for r in RECS] + ["no timestamp"]


def test_as_state_roundtrip_matches_state_text():
    s = store(True if fts5_available() else False)
    s.add([{"id": "schema", "text": "SCHEMA", "pinned": 1}])
    ids = ["ORD-1005", "schema", "ORD-1002"]
    st = s.as_state(ids)
    assert st == {"pinned": ["SCHEMA"], "records": [RECS[1], RECS[4]]}
    json.dumps(st)
    assert s.as_state(ids, fmt="list") == ["SCHEMA", RECS[1], RECS[4]]
    assert record_text(RECS[1]) in state_text(st["records"])  # same compact serialisation


@pytest.mark.parametrize("use_fts", MODES)
def test_upsert_replaces(use_fts, tmp_path):
    p = str(tmp_path / "s.db")
    s = store(use_fts, path=p)
    n = len(s)
    assert s.search(["awaiting payment"], k=1).ids == ["ORD-1002"]
    s.add_records([{**RECS[1], "note": "teleported to Mars"}], prefix="orders", start=1)
    assert len(s) == n
    assert s.search(["teleported Mars"], k=1).ids == ["ORD-1002"]
    lex = s.search(["awaiting"], k=5, weights={"dense": 0})
    assert "ORD-1002" not in lex.ids
    assert "teleported" in s.assemble(["ORD-1002"])[0][0]
    s.close()
    s2 = StateStore(p, embed_fn=EMB, use_fts=use_fts)  # reopen: index and vectors persist
    assert s2.search(["teleported Mars"], k=1).ids == ["ORD-1002"] and s2.dim == 512


def test_add_without_embedder_drops_stale_vector():
    s = store(True if fts5_available() else False)
    s.embed_fn = None
    s.add([{"id": "ORD-1001", "text": "replaced text"}])
    assert not s._index()["has_vec"][s._pos_of_ids(["ORD-1001"])[0]]
    assert s._index()["has_vec"].sum() == len(RECS) - 1


def test_fallback_matches_fts_top1():
    if not fts5_available():
        pytest.skip("no FTS5")
    a, b = store(True, emb=None), store(False, emb=None)
    for q in ["ORD-1004", "Wen cancelled", "damaged", "orders[0]", "signed recipient"]:
        assert a.search([q], k=1, weights={"dense": 0}).ids == b.search([q], k=1, weights={"dense": 0}).ids


def test_bad_inputs():
    s = StateStore(embed_fn=EMB)
    with pytest.raises(ValueError):
        s.add([{"text": "no id"}])
    s.add([{"id": "a", "text": "x"}])
    with pytest.raises(ValueError):
        s.add([{"id": "b", "text": "y"}], embed_fn=hash_embedder(dim=8))
    with pytest.raises(KeyError):
        s.assemble(["missing"])
    assert s.search([], k=3).ids == [] and s.select([], 100).ids == []


@pytest.mark.parametrize("use_fts", MODES)
def test_common_term_pruning(use_fts):
    s = StateStore(use_fts=use_fts)
    s.add([{"id": f"r{i}", "text": f"order status {'shipped' if i % 3 else 'pending'} note {i}"} for i in range(60)]
          + [{"id": "rare", "text": "order status shipped note chargeback"}])
    s.prune_min_n = 10
    q = ["order status chargeback"]
    assert s.search(q, k=1, weights={"dense": 0}).ids == ["rare"]
    # all terms common: rarest ('pending', df 20) is kept, so the list is not empty and all hits are 'pending'
    r = s.search(["order status pending"], k=100, weights={"dense": 0})
    assert len(r.ids) == 20
    s.max_df = None
    assert len(s.search(["order status pending"], k=100, weights={"dense": 0}).ids) == 61
