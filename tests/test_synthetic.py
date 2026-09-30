"""Tests for data/synthetic/generate.py (CPU only, fast).

Label correctness is re-verified by small independent solvers that parse the rendered *state* text
(not the generator's hidden world), for several families.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import itertools
import json
import re
from collections import Counter, deque
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("synthetic_generate", ROOT / "data/synthetic/generate.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

N_PER_FAMILY = 60


@pytest.fixture(scope="module")
def train_recs():
    fams = gen.TRAIN_FAMILIES
    return gen.generate_split("train", N_PER_FAMILY * len(fams), 123, fams)


@pytest.fixture(scope="module")
def ood_recs():
    fams = gen.OOD_FAMILIES
    return gen.generate_split("test_ood", N_PER_FAMILY * len(fams), 123, fams)


@pytest.fixture(scope="module")
def all_recs(train_recs, ood_recs):
    return train_recs + ood_recs


def by_family(recs, fam):
    out = [r for r in recs if r["family"] == fam]
    assert out, f"no records for {fam}"
    return out


def qs(recs, fam, kind=None):
    for r in by_family(recs, fam):
        for q in r["questions"]:
            if kind is None or q["kind"] == kind:
                yield r, q


def answer(q):
    return q["options"][q["label"]]


def yes(q):
    return q["label"] == 0


# ------------------------------------------------------------------------------------------------
# Structural properties
# ------------------------------------------------------------------------------------------------
def test_family_split():
    assert len(gen.FAMILY_SPLIT) >= 14
    assert set(gen.FAMILY_SPLIT.values()) == {"train", "ood"}
    n_ood = sum(v == "ood" for v in gen.FAMILY_SPLIT.values())
    assert n_ood / len(gen.FAMILY_SPLIT) >= 0.30
    assert set(gen.FAMILIES) == set(gen.FAMILY_SPLIT)


def test_determinism():
    fams = gen.TRAIN_FAMILIES
    a = gen.generate_split("val", 2 * len(fams), 7, fams)
    b = gen.generate_split("val", 2 * len(fams), 7, fams)
    c = gen.generate_split("val", 2 * len(fams), 8, fams)
    assert json.dumps(a) == json.dumps(b)
    assert json.dumps(a) != json.dumps(c)


def test_schema_and_labels(all_recs):
    ids = set()
    for r in all_recs:
        assert set(r) == {"id", "family", "state", "questions"}
        assert r["id"] not in ids
        ids.add(r["id"])
        json.dumps(r["state"])  # serialisable
        assert 1 <= len(r["questions"]) <= gen.MAX_Q_PER_STATE
        for q in r["questions"]:
            assert q["type"] in {"choice", "noul", "score"}
            assert isinstance(q["prompt"], str) and q["prompt"].strip()
            opts = q["options"]
            assert all(isinstance(o, str) and o.strip() for o in opts)
            assert len(set(opts)) == len(opts), opts
            assert 2 <= len(opts) <= 255
            assert isinstance(q["label"], int) and 0 <= q["label"] < len(opts)
            if q["type"] == "noul":
                assert opts == ["yes", "no"]
            if q["type"] == "score":
                assert opts in (["1", "2", "3", "4", "5"], ["critical", "low", "ok", "overstocked"],
                                ["low", "medium", "high", "critical"], ["light", "moderate", "heavy", "overloaded"])


def test_every_family_present_and_split_respected(train_recs, ood_recs):
    assert {r["family"] for r in train_recs} == set(gen.TRAIN_FAMILIES)
    assert {r["family"] for r in ood_recs} == set(gen.OOD_FAMILIES)
    assert not any(gen.FAMILY_SPLIT[r["family"]] == "ood" for r in train_recs)


def test_option_counts_and_big_fraction(train_recs):
    ks = [len(q["options"]) for r in train_recs for q in r["questions"] if q["type"] == "choice"]
    nq = sum(len(r["questions"]) for r in train_recs)
    assert min(ks) == 2 and max(ks) >= 60
    assert max(k for k in ks if k < 60) <= gen.MAX_SMALL_K
    big = sum(k >= 60 for k in ks) / nq
    assert 0.01 <= big <= 0.06, big


def test_label_positions_spread(all_recs):
    choice = [q for r in all_recs for q in r["questions"] if q["type"] == "choice" and len(q["options"]) >= 3]
    rel = [q["label"] / (len(q["options"]) - 1) for q in choice]
    assert abs(sum(rel) / len(rel) - 0.5) < 0.05
    first = sum(q["label"] == 0 for q in choice) / len(choice)
    expect = sum(1 / len(q["options"]) for q in choice) / len(choice)
    assert abs(first - expect) < 0.04
    # binned uniformity of relative position
    bins = Counter(min(4, int(x * 5)) for x in rel)
    for b in range(5):
        assert 0.12 < bins[b] / len(rel) < 0.28, bins


def test_noul_balance(all_recs):
    ys = [yes(q) for r in all_recs for q in r["questions"] if q["type"] == "noul"]
    assert 0.35 < sum(ys) / len(ys) < 0.65


def test_splits_disjoint_states():
    fams = gen.TRAIN_FAMILIES
    seen: set = set()
    a = gen.generate_split("train", 3 * len(fams), 0, fams, seen)
    b = gen.generate_split("val", 3 * len(fams), 0, fams, seen)
    ka = {gen.state_key(r["state"]) for r in a}
    kb = {gen.state_key(r["state"]) for r in b}
    assert not ka & kb


def test_cli_writes_files(tmp_path):
    gen.main(["--out", str(tmp_path), "--n-train", "40", "--n-val", "13", "--n-test", "14", "--n-calib", "13",
              "--seed", "1"])
    for split, n in [("train", 40), ("val", 13), ("calib", 13), ("test_id", 14), ("test_ood", 14)]:
        lines = (tmp_path / f"{split}.jsonl").read_text().splitlines()
        assert len(lines) == n
        fams = {json.loads(l)["family"] for l in lines}
        want = "ood" if split == "test_ood" else "train"
        assert all(gen.FAMILY_SPLIT[f] == want for f in fams)
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["splits"]["train"]["records"] == 40
    assert "choice_option_count_hist" in stats["splits"]["train"]


def test_state_lengths(all_recs):
    lens = sorted(gen.state_chars(r["state"]) for r in all_recs)
    assert lens[len(lens) // 2] >= 200
    assert lens[-1] <= 6000


# ------------------------------------------------------------------------------------------------
# Independent label verification
# ------------------------------------------------------------------------------------------------
def parse_table(state):
    lines = [l for l in state.splitlines() if l.strip()]
    for i, l in enumerate(lines):
        for delim in ("|", ";", ","):
            cells = [c.strip() for c in l.strip().strip("|").split(delim)]
            if cells[0] in ("Region", "Store", "Team") and len(cells) > 1:
                cols = cells[1:]
                rows = {}
                for m in lines[i + 1:]:
                    if set(m.strip()) <= set("|-"):
                        continue
                    cs = [c.strip() for c in m.strip().strip("|").split(delim)]
                    if len(cs) != len(cells) or not all(c.isdigit() for c in cs[1:]):
                        break
                    rows[cs[0]] = dict(zip(cols, map(int, cs[1:])))
                return cols, rows
    raise AssertionError("no table")


def test_verify_table_arith(train_recs):
    n = 0
    for r, q in qs(train_recs, "table_arith"):
        cols, rows = parse_table(r["state"])
        a, ans = q["args"], answer(q)
        op = a.get("op")
        if op == "col_sum":
            assert int(ans) == sum(v[a["col"]] for v in rows.values())
        elif op in ("argmax", "argmin"):
            f = max if op == "argmax" else min
            assert ans == f(rows, key=lambda k: rows[k][a["col"]])
        elif op == "row_total":
            assert int(ans) == sum(rows[a["row"]].values())
        elif op == "diff":
            assert int(ans) == rows[a["a"]][a["col"]] - rows[a["b"]][a["col"]]
        elif op == "gt":
            assert yes(q) == (rows[a["row"]][a["col"]] > a["threshold"])
        elif op == "row_total_argmax":
            assert ans == max(rows, key=lambda k: sum(rows[k].values()))
        n += 1
    assert n > 50


TTT_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
A1 = ["A1", "B1", "C1", "A2", "B2", "C2", "A3", "B3", "C3"]
NAMED = ["top-left", "top-middle", "top-right", "middle-left", "center", "middle-right", "bottom-left",
         "bottom-middle", "bottom-right"]


def parse_ttt(state):
    rows = []
    for l in state.splitlines():
        m = re.match(r"^\s*(?:\d\s+)?([XO.])\s*\|?\s*([XO.])\s*\|?\s*([XO.])\s*$", l)
        if m:
            rows += list(m.groups())
    assert len(rows) == 9
    return rows


def ttt_win(b):
    for l in TTT_LINES:
        if b[l[0]] != "." and b[l[0]] == b[l[1]] == b[l[2]]:
            return b[l[0]]
    return None


def test_verify_tictactoe(train_recs):
    n = 0
    for r, q in qs(train_recs, "tictactoe"):
        b = parse_ttt(r["state"])
        names = A1 if "A-C" in r["state"] else NAMED
        ans = answer(q)
        w = ttt_win(b)
        if q["kind"] == "winner":
            assert (w is None) == ans.startswith("nobody")
            if w:
                assert ans.startswith(w) or f"player {w}" in ans
        elif q["kind"] == "legal_move":
            assert b[names.index(ans)] == "."
            assert all(b[names.index(o)] != "." for o in q["options"] if o != ans)
        elif q["kind"] == "winning_move":
            p = "X" if b.count("X") == b.count("O") else "O"
            i = names.index(ans)
            assert ttt_win(b[:i] + [p] + b[i + 1:]) == p
        elif q["kind"] == "cell_empty":
            assert yes(q) == (b[names.index(q["args"]["cell"])] == ".")
        elif q["kind"] == "game_over":
            assert yes(q) == (w is not None or "." not in b)
        elif q["kind"] == "turn":
            assert ans == ("X" if b.count("X") == b.count("O") else "O")
        n += 1
    assert n > 50


def parse_graph(state):
    adj = {}
    for l in state.splitlines():
        m = re.match(r"^(\w+) -> (.*)$", l) or re.match(r"^(\w+): ([\w ]+|-)$", l)
        if m:
            rest = m.group(2).strip()
            adj[m.group(1)] = [] if rest in ("(none)", "-") else re.split(r"[,\s]+", rest)
            continue
        m = re.match(r"^From (\w+) you can go to (.*)\.$", l)
        if m:
            adj[m.group(1)] = [x for x in re.split(r"[,\s]+", m.group(2)) if x]
            continue
        m = re.match(r"^(\w+) has no outgoing links\.$", l)
        if m:
            adj[m.group(1)] = []
    return adj


def bfs(adj, s):
    d = {s: 0}
    dq = deque([s])
    while dq:
        u = dq.popleft()
        for v in adj.get(u, []):
            if v not in d:
                d[v] = d[u] + 1
                dq.append(v)
    return d


def test_verify_graph(ood_recs):
    n = 0
    for r, q in qs(ood_recs, "graph_reachability"):
        adj = parse_graph(r["state"])
        assert len(adj) >= 5
        a = q["args"]
        if q["kind"] == "reachable":
            assert yes(q) == (a["t"] in bfs(adj, a["s"]))
        elif q["kind"] == "hops":
            assert int(answer(q)) == bfs(adj, a["s"])[a["t"]]
        elif q["kind"] == "which_reachable":
            d = bfs(adj, a["s"])
            assert answer(q) in d
            assert all(o not in d for o in q["options"] if o != answer(q))
        elif q["kind"] == "max_outdeg":
            assert answer(q) == max(adj, key=lambda u: len(adj[u]))
        n += 1
    assert n > 50


def test_verify_grid(ood_recs):
    moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    n = 0
    for r, q in qs(ood_recs, "grid_navigation"):
        g = [l for l in r["state"].splitlines() if l and set(l) <= set(".#SG")]
        H, W = len(g), len(g[0])
        find = lambda ch: next((i, j) for i in range(H) for j in range(W) if g[i][j] == ch)
        s, t = find("S"), find("G")
        d = {t: 0}
        dq = deque([t])
        while dq:
            u = dq.popleft()
            for dr, dc in moves.values():
                v = (u[0] + dr, u[1] + dc)
                if 0 <= v[0] < H and 0 <= v[1] < W and g[v[0]][v[1]] != "#" and v not in d:
                    d[v] = d[u] + 1
                    dq.append(v)
        if q["kind"] == "reachable":
            assert yes(q) == (s in d)
        elif q["kind"] == "path_len":
            assert int(answer(q)) == d[s]
        elif q["kind"] == "first_move":
            key = next(k for k in moves if answer(q) == k or f"move {k}" in answer(q))
            dr, dc = moves[key]
            assert d.get((s[0] + dr, s[1] + dc)) == d[s] - 1
        elif q["kind"] == "wall_adjacent":
            dr, dc = moves[q["args"]["dir"]]
            assert yes(q) == (g[s[0] + dr][s[1] + dc] == "#")
        n += 1
    assert n > 50


def test_verify_dates(train_recs):
    wd = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    n = 0
    for r, q in qs(train_recs, "date_reasoning"):
        a = q["args"]
        if q["kind"] == "weekday":
            d = dt.date.fromisoformat(a["date"])
            assert answer(q) == wd[d.weekday()]
            # the reference line lets the weekday be derived from the state
            m = re.search(r"For reference, (\d{4}-\d\d-\d\d) is a (\w+)\.", r["state"])
            ref = dt.date.fromisoformat(m.group(1))
            assert wd[(wd.index(m.group(2)) + (d - ref).days) % 7] == answer(q)
        elif q["kind"] == "day_gap":
            assert int(answer(q)) == abs((dt.date.fromisoformat(a["b"]) - dt.date.fromisoformat(a["a"])).days)
        elif q["kind"] == "date_after":
            assert answer(q) == (dt.date.fromisoformat(a["date"]) + dt.timedelta(days=a["days"])).isoformat()
        elif q["kind"] == "before":
            assert yes(q) == (a["a"] < a["b"])
        n += 1
    assert n > 30


def test_verify_sequence(ood_recs):
    n = 0
    for r, q in qs(ood_recs, "sequence_simulation"):
        ops = re.findall(r"^\d+\. (.+)$", r["state"], flags=re.M)
        if "stack" in r["state"].lower():
            st = []
            for o in ops:
                if o == "pop":
                    st.pop()
                else:
                    st.append(o.split(" ", 1)[1])
            if q["kind"] == "stack_top":
                assert answer(q) == st[-1]
            elif q["kind"] == "stack_size":
                assert int(answer(q)) == len(st)
            elif q["kind"] == "stack_contains":
                assert yes(q) == (q["args"]["item"] in st)
        else:
            v = int(re.search(r"(?:starts at|Initial value:|x =) (-?\d+)", r["state"]).group(1))
            hist = []
            for o in ops:
                w = o.split()
                if w[0] == "add":
                    v += int(w[1])
                elif w[0] == "subtract":
                    v -= int(w[1])
                elif w[0] == "multiply":
                    v *= int(w[2])
                elif w[0] == "set":
                    v = int(w[2])
                elif w[0] == "negate":
                    v = -v
                hist.append(v)
            if q["kind"] == "final_value":
                assert int(answer(q)) == v
            elif q["kind"] == "value_after_step":
                assert int(answer(q)) == hist[q["args"]["step"] - 1]
            elif q["kind"] == "final_even":
                assert yes(q) == (v % 2 == 0)
            elif q["kind"] == "final_gt":
                assert yes(q) == (v > q["args"]["threshold"])
        n += 1
    assert n > 50


def test_verify_logs(train_recs):
    n = 0
    for r, q in qs(train_recs, "log_counting"):
        entries = re.findall(r"^\S+ (\d\d:\d\d:\d\d) \[(\w+)\] (\w+): ", r["state"], flags=re.M)
        if q["kind"] == "count_level":
            assert int(answer(q)) == sum(e[1] == q["args"]["level"] for e in entries)
        elif q["kind"] == "count_service":
            assert int(answer(q)) == sum(e[2] == q["args"]["service"] for e in entries)
        elif q["kind"] == "any_level_service":
            a = q["args"]
            assert yes(q) == any(e[1] == a["level"] and e[2] == a["service"] for e in entries)
        elif q["kind"] == "most_level_service":
            c = Counter(e[2] for e in entries if e[1] == q["args"]["level"])
            assert answer(q) == c.most_common(1)[0][0]
        elif q["kind"] == "error_time":
            errs = [e[0] for e in entries if e[1] == "ERROR"]
            assert answer(q) == (errs[-1] if q["args"]["which"] == "last" else errs[0])
        n += 1
    assert n > 50


def test_verify_set_membership(train_recs):
    n = 0
    for r, q in qs(train_recs, "set_membership"):
        if not isinstance(r["state"], dict):
            continue
        lists = r["state"]["lists"]
        a = q["args"]
        op = a.get("op")
        if op == "in":
            assert yes(q) == (a["item"] in lists[a["list"]])
        elif op == "not_in":
            assert yes(q) == (a["item"] not in lists[a["list"]])
        elif op == "both":
            assert yes(q) == all(a["item"] in lists[l] for l in a["lists"])
        elif op == "a_not_b":
            assert yes(q) == (a["item"] in lists[a["lists"][0]] and a["item"] not in lists[a["lists"][1]])
        elif q["kind"] == "which_member":
            assert answer(q) in lists[a["list"]]
            assert all(o not in lists[a["list"]] for o in q["options"] if o != answer(q))
        n += 1
    assert n > 20


def test_verify_connect_four(ood_recs):
    n = 0
    for r, q in qs(ood_recs, "connect_four"):
        rows = [l.split() for l in r["state"].splitlines() if re.fullmatch(r"([RY.] ){6}[RY.]", l)]
        assert len(rows) == 6
        if q["kind"] == "column_full":
            assert yes(q) == (rows[0][q["args"]["col"] - 1] != ".")
        elif q["kind"] == "landing_row":
            c = q["args"]["col"] - 1
            empty = sum(1 for rr in rows if rr[c] == ".")
            assert int(answer(q)) == 6 - empty + 1
        n += 1
    assert n > 30


def test_verify_schedule(ood_recs):
    n = 0
    for r, q in qs(ood_recs, "schedule_conflict", "overlap"):
        st = r["state"]
        if isinstance(st, dict):
            ev = {e["title"]: (e["start"], e["end"]) for e in st["events"]}
        else:
            ev = {m.group(3): (m.group(1), m.group(2))
                  for m in re.finditer(r"^(\d\d:\d\d)-(\d\d:\d\d)  (.+)$", st, flags=re.M)}
        (s1, e1), (s2, e2) = ev[q["args"]["a"]], ev[q["args"]["b"]]
        assert yes(q) == (s1 < e2 and s2 < e1)
        n += 1
    assert n > 20


def test_verify_schedule_load(ood_recs):
    n = 0
    for r, q in qs(ood_recs, "schedule_conflict", "day_load"):
        st = r["state"]
        text = st["load_rule"] if isinstance(st, dict) else st
        th = [int(x) for x in re.findall(r"under (\d+) min", text)]
        if isinstance(st, dict):
            spans = [(e["start"], e["end"]) for e in st["events"]]
        else:
            spans = re.findall(r"^(\d\d:\d\d)-(\d\d:\d\d)  ", st, flags=re.M)
        mins = lambda h: int(h[:2]) * 60 + int(h[3:])
        tot = sum(mins(e) - mins(s) for s, e in spans)
        assert q["label"] == next((j for j, t in enumerate(th) if tot < t), len(th))
        n += 1
    assert n > 10


def test_verify_logic_puzzle_unique(ood_recs):
    """Re-solve each puzzle by brute force from the clue text and check the labelled answers."""
    n = 0
    for r in by_family(ood_recs, "logic_puzzle"):
        st = r["state"]
        m = re.search(r"(?:puzzle\.|sheet:) (.+?) each own exactly one different pet \((.+?)\)(?: and each live in a different house \((.+?)\))?", st)
        people = re.split(r", | and ", m.group(1))
        pets = m.group(2).split(", ")
        cols = m.group(3).split(", ") if m.group(3) else [None] * len(people)
        clues = re.findall(r"^\d+\. (.+)$", st, flags=re.M)
        sols = []
        for pp in itertools.permutations(pets):
            for cc in itertools.permutations(cols):
                ok = True
                for c in clues:
                    if mm := re.fullmatch(r"(\w+) owns the (\w+)\.", c):
                        ok &= pp[people.index(mm[1])] == mm[2]
                    elif mm := re.fullmatch(r"(\w+) does not own the (\w+)\.", c):
                        ok &= pp[people.index(mm[1])] != mm[2]
                    elif mm := re.fullmatch(r"(\w+) lives in the (\w+) house\.", c):
                        ok &= cc[people.index(mm[1])] == mm[2]
                    elif mm := re.fullmatch(r"(\w+) does not live in the (\w+) house\.", c):
                        ok &= cc[people.index(mm[1])] != mm[2]
                    elif mm := re.fullmatch(r"The (\w+) owner lives in the (\w+) house\.", c):
                        ok &= cc[pp.index(mm[1])] == mm[2]
                    else:
                        raise AssertionError(c)
                if ok:
                    sols.append((pp, cc))
        sols = list({s for s in sols})
        assert len(sols) == 1, (st, sols)
        pp, cc = sols[0]
        for q in r["questions"]:
            if q["kind"] == "who_owns":
                assert pp[people.index(answer(q))] in q["prompt"]
            elif q["kind"] == "what_pet":
                who = next(p for p in people if p in q["prompt"])
                assert answer(q) == pp[people.index(who)]
            elif q["kind"] == "owns":
                assert yes(q) == (pp[people.index(q["args"]["person"])] == q["args"]["pet"])
            n += 1
    assert n > 30
