#!/usr/bin/env python
"""Deterministic generator of programmatically verifiable decision tasks (docs/SPEC.md §5.1, §7.1).

No LLM teacher: every state is rendered from a hidden structured world and every label is computed by
code from that world, so labels are ground truth. One JSONL record per *state*:

    {"id": ..., "family": ..., "state": <str | JSON object>,
     "questions": [{"type": "choice"|"noul"|"score", "prompt": ..., "options": [...], "label": int,
                    "kind": <question sub-type>, "args": {...}}]}

* noul  -> options == ["yes", "no"], label 0 == yes.
* score -> options are ordered levels (never shuffled), label == level index.
* choice-> options shuffled; label is the index of the correct option.

`kind`/`args` are metadata for analysis and for independent re-verification in tests; the training
loader should ignore them (they are not part of the §6 API).

Splits are by *family* (FAMILY_SPLIT): train/val/test_id/calib use train families with disjoint seeds;
test_ood uses only ood families. States that collide with an earlier-generated state are regenerated.

Usage:
    python data/synthetic/generate.py --out data/synthetic --n-train 20000 --n-val 2000 \
        --n-test 2000 --n-calib 2000 --seed 0
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict, deque

# ---------------------------------------------------------------------------------------------------
# Global knobs
# ---------------------------------------------------------------------------------------------------
P_BIG = 0.20          # prob. that a big-option-capable choice question gets 60..255 options (~3% overall)
MAX_SMALL_K = 40      # normal option counts are 2..40 (log-uniform, skewed small)
MAX_Q_PER_STATE = 6

FAMILY_SPLIT = {
    # train families (13)
    "record_lookup": "train",      # JSON / directory records: field extraction, id lookup
    "table_arith": "train",        # pipe/CSV tables: sums, argmax, differences
    "compare_sort": "train",       # prose facts: largest / smallest / rank
    "log_counting": "train",       # server logs: counting, first/last, argmax service
    "ticket_routing": "train",     # explicit ordered routing rules + tickets (long/short labels)
    "date_reasoning": "train",     # event dates: order, day gaps, weekday offsets
    "unit_conversion": "train",    # mixed units: comparisons and conversions
    "set_membership": "train",     # named lists: membership / negation (noul-heavy)
    "tictactoe": "train",          # 3x3 board: winner, turn, legal / winning move
    "inventory_levels": "train",   # stock table + explicit thresholds (score levels, actions)
    "string_props": "train",       # word lists: longest, alphabetical, letters
    "severity_scoring": "train",   # rubric + incident reports (score)
    "price_calc": "train",         # shopping cart + discount/shipping rules
    # ood families (7 / 20 = 35%) -- different surface forms and reasoning types
    "grid_navigation": "ood",      # ASCII maze: shortest-path first move
    "logic_puzzle": "ood",         # constraint clues: who owns what
    "schedule_conflict": "ood",    # calendars: interval overlap
    "graph_reachability": "ood",   # directed adjacency lists: reachability, hops
    "chat_transcript": "ood",      # multi-party chat: vote tallies, who said what
    "connect_four": "ood",         # 6x7 gravity board game
    "sequence_simulation": "ood",  # step-by-step register / stack programs
}

# ---------------------------------------------------------------------------------------------------
# Word pools
# ---------------------------------------------------------------------------------------------------
FIRST = ["Alice", "Bruno", "Chen", "Dana", "Elif", "Farah", "Goran", "Hana", "Ivan", "Jonas", "Keiko",
         "Lena", "Mateo", "Nadia", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tariq", "Uma", "Victor",
         "Wen", "Ximena", "Yusuf", "Zoe", "Amara", "Bilal", "Carmen", "Dmitri", "Esme", "Felix", "Gita",
         "Hugo", "Ines", "Jamal", "Kofi", "Lucia", "Milan", "Noor"]
LAST = ["Garcia", "Smith", "Okafor", "Tanaka", "Novak", "Rossi", "Kim", "Silva", "Muller", "Haddad",
        "Petrov", "Nguyen", "Kowalski", "Dubois", "Larsen", "Mensah", "Costa", "Iyer", "Moreau", "Schmidt"]
CITIES = ["Lisbon", "Porto", "Madrid", "Lyon", "Munich", "Vienna", "Prague", "Krakow", "Oslo", "Bergen",
          "Dublin", "Leeds", "Glasgow", "Zurich", "Milan", "Naples", "Seville", "Valencia", "Rotterdam",
          "Antwerp", "Gdansk", "Tallinn", "Riga", "Helsinki", "Athens", "Sofia", "Zagreb", "Ljubljana",
          "Bordeaux", "Hamburg"]
PLANS = ["free", "basic", "pro", "team", "enterprise"]
COMPANIES = [f"{a} {b}" for a in ["Acme", "Blue", "Crest", "Delta", "Evergreen", "Falcon", "Granite",
                                   "Harbor", "Iris", "Juniper", "Keystone", "Lumen", "Meridian", "Nimbus",
                                   "Orion", "Pioneer", "Quartz", "Redwood", "Summit", "Tidal", "Unity",
                                   "Vertex", "Willow", "Zenith"]
             for b in ["Labs", "Logistics", "Foods", "Systems", "Partners", "Media", "Energy", "Health",
                       "Works", "Capital"]]
WORDS = sorted(set("""apple anchor arrow autumn badge bamboo banner basket beacon blanket blossom border
bottle bracket breeze bridge bucket butter cabin cactus camera candle canyon carpet castle cellar
channel chapter cherry circle clover cobalt comet copper cotton crayon crystal curtain dagger dancer
desert diamond dinner dolphin donkey dragon drawer eagle echo elbow ember engine falcon fabric feather
fence fiddle finger forest fossil fountain frame galaxy garden garlic giant ginger glacier globe goblet
granite guitar hammer harbor harvest helmet hermit hollow honey horizon hunter island ivory jacket
jasmine jelly jewel jungle kettle kingdom kitten ladder lagoon lantern lemon library lily linen lizard
locket magnet mango marble meadow mirror monkey mosaic mountain muffin napkin needle nest noodle
number oasis ocean olive orbit orchard otter oyster paddle palace panther parrot pebble pencil pepper
piano pillow pirate planet pocket poem pollen pony puzzle quarry quill rabbit radar raisin ribbon
river rocket saddle salmon sandal satchel scarf shadow shelter silver sketch slipper socket spider
spiral sponge stable statue stone sugar summit sunset swamp table tablet teapot temple thunder ticket
tiger timber tomato torch tower trumpet tulip tunnel turtle umbrella valley velvet violin volcano
wagon walnut water whistle window winter wizard wolf yogurt zebra zipper acorn bakery beetle cinder
dune flute gravel igloo jigsaw kayak lobster nectar pumpkin quartz reef sleigh thimble
walrus yarn sparrow""".split()))
SERVICES = ["auth", "billing", "search", "gateway", "db", "cache", "scheduler", "mailer", "storage",
            "payments", "notifier", "indexer"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


# ---------------------------------------------------------------------------------------------------
# Question builders
# ---------------------------------------------------------------------------------------------------
def sample_k(rng: random.Random, big_ok: bool) -> int:
    if big_ok and rng.random() < P_BIG:
        return rng.randint(60, 255)
    k = int(round(math.exp(rng.uniform(math.log(2), math.log(MAX_SMALL_K)))))
    return max(2, min(MAX_SMALL_K, k))


def mk_choice(rng, kind, prompt, correct, hard=(), pool=(), gen=None, big_ok=False, k=None, args=None):
    """Choice question: correct + distractors (hard first, then pool, then gen(k)), shuffled."""
    correct = str(correct)
    k = k if k is not None else sample_k(rng, big_ok)
    seen = {correct}
    ds: list[str] = []

    def add(cands):
        cands = [str(c) for c in cands]
        rng.shuffle(cands)
        for c in cands:
            if len(ds) >= k - 1:
                return
            if c not in seen and c.strip():
                seen.add(c)
                ds.append(c)

    add(list(hard))
    add(list(pool))
    if gen is not None and len(ds) < k - 1:
        add(gen(k))
    if not ds:
        return None
    opts = [correct] + ds
    rng.shuffle(opts)
    return {"type": "choice", "prompt": prompt, "options": opts, "label": opts.index(correct),
            "kind": kind, "args": args or {}}


def mk_noul(kind, prompt, truth: bool, args=None):
    return {"type": "noul", "prompt": prompt, "options": ["yes", "no"], "label": 0 if truth else 1,
            "kind": kind, "args": args or {}}


def mk_score(kind, prompt, levels, idx, args=None):
    assert 0 <= idx < len(levels)
    return {"type": "score", "prompt": prompt, "options": list(levels), "label": int(idx),
            "kind": kind, "args": args or {}}


def near_numbers(rng, correct: int, lo: int = 0, width: int = 6):
    """Distractor generator: integers near `correct` (window widens with the option count)."""
    def gen(k):
        w = max(width, k)
        a, b = max(lo, correct - w), correct + w
        vals = [v for v in range(a, b + 1) if v != correct]
        rng.shuffle(vals)
        return vals
    return gen


def T(rng, templates, **kw):
    return rng.choice(templates).format(**kw)


def rand_date(rng, y0=2019, y1=2026):
    start = dt.date(y0, 1, 1).toordinal()
    end = dt.date(y1, 12, 31).toordinal()
    return dt.date.fromordinal(rng.randint(start, end))


def fmt_date(d: dt.date, style: str) -> str:
    if style == "iso":
        return d.isoformat()
    if style == "long":
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    return f"{d.day} {MONTHS[d.month - 1][:3]} {d.year}"


def rand_ids(rng, n, prefix, lo=1000, hi=9999):
    return [f"{prefix}{v}" for v in rng.sample(range(lo, hi + 1), n)]


def money(cents: int) -> str:
    return f"${cents // 100}.{cents % 100:02d}"


# ---------------------------------------------------------------------------------------------------
# Train families
# ---------------------------------------------------------------------------------------------------
def fam_record_lookup(rng):
    n = rng.randint(4, 12) if rng.random() < 0.9 else rng.randint(13, 24)
    names = [f"{f} {rng.choice(LAST)}" for f in rng.sample(FIRST, n)]
    ids = rand_ids(rng, n, "C-")
    ages = rng.sample(range(18, 81), n)
    recs = [{"id": ids[i], "name": names[i], "city": rng.choice(CITIES), "plan": rng.choice(PLANS),
             "age": ages[i], "signup": rand_date(rng, 2018, 2025).isoformat()} for i in range(n)]
    if rng.random() < 0.6:
        state = {"source": "crm_export", "exported_at": rand_date(rng, 2025, 2026).isoformat(),
                 "records": recs}
    else:
        lines = [T(rng, ["Customer directory (export).", "CRM snapshot - active customers:",
                         "Accounts on file:"])]
        for r in recs:
            lines.append(f"- {r['id']}: {r['name']}, {r['age']} years old, based in {r['city']}, "
                         f"{r['plan']} plan, customer since {r['signup']}.")
        state = "\n".join(lines)
    other_names = lambda k: [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(3 * k)]

    def q_city():
        r = rng.choice(recs)
        p = T(rng, ["Which city is {n} based in?", "What is the city listed for {n}?",
                    "According to the records, where is {n} located?"], n=r["name"])
        return mk_choice(rng, "city_of_name", p, r["city"], hard=[x["city"] for x in recs], pool=CITIES,
                         args={"name": r["name"]})

    def q_id():
        r = rng.choice(recs)
        p = T(rng, ["What is the customer ID of {n}?", "Which ID belongs to {n}?",
                    "Find the record ID for {n}."], n=r["name"])
        return mk_choice(rng, "id_of_name", p, r["id"], hard=ids,
                         gen=lambda k: rand_ids(rng, min(3 * k, 8000), "C-"), big_ok=True,
                         args={"name": r["name"]})

    def q_name():
        r = rng.choice(recs)
        p = T(rng, ["Who is customer {i}?", "Which customer has the ID {i}?",
                    "Record {i} belongs to which person?"], i=r["id"])
        return mk_choice(rng, "name_of_id", p, r["name"], hard=names, gen=other_names, big_ok=True,
                         args={"id": r["id"]})

    def q_plan():
        r = rng.choice(recs)
        p = T(rng, ["Which plan is customer {i} on?", "What subscription plan does {i} have?",
                    "Plan tier for account {i}:"], i=r["id"])
        return mk_choice(rng, "plan_of_id", p, r["plan"], pool=PLANS, args={"id": r["id"]})

    def q_oldest():
        young = rng.random() < 0.5
        tgt = (min if young else max)(recs, key=lambda x: x["age"])
        p = T(rng, ["Who is the {w} customer?", "Which person in the list is the {w}?",
                    "Identify the {w} customer."], w="youngest" if young else "oldest")
        return mk_choice(rng, "youngest" if young else "oldest", p, tgt["name"], hard=names,
                         gen=other_names)

    def q_plan_noul():
        r = rng.choice(recs)
        plan = r["plan"] if rng.random() < 0.5 else rng.choice([x for x in PLANS if x != r["plan"]])
        p = T(rng, ["Is {n} on the {p} plan?", "Does {n} subscribe to the {p} plan?",
                    "Is the plan for {n} '{p}'?"], n=r["name"], p=plan)
        return mk_noul("plan_is", p, plan == r["plan"], {"name": r["name"], "plan": plan})

    def q_age_noul():
        r = rng.choice(recs)
        th = r["age"] + rng.choice([-7, -3, -1, 1, 2, 5, 9])
        p = T(rng, ["Is {n} older than {t}?", "Is {n}'s age above {t}?",
                    "Does {n} have an age greater than {t} years?"], n=r["name"], t=th)
        return mk_noul("age_gt", p, r["age"] > th, {"name": r["name"], "threshold": th})

    return state, [q_city, q_id, q_name, q_plan, q_oldest, q_plan_noul, q_age_noul]


TABLE_ROWS = {
    "Region": ["North", "South", "East", "West", "Central", "Coastal", "Highlands", "Metro", "Valley",
               "Islands"],
    "Store": ["Downtown", "Airport", "Mall", "Harbor", "Uptown", "Riverside", "Station", "Campus",
              "Old Town", "Lakeside"],
    "Team": ["Falcons", "Otters", "Comets", "Lynx", "Ravens", "Tigers", "Pandas", "Sharks", "Wolves",
             "Herons"],
}
TABLE_COLS = [["Q1", "Q2", "Q3", "Q4"], ["Jan", "Feb", "Mar", "Apr"], ["Mon", "Tue", "Wed", "Thu", "Fri"],
              ["2022", "2023", "2024"]]
TABLE_METRICS = ["Units sold", "Tickets closed", "Revenue in thousands of dollars", "Visitors",
                 "Orders shipped"]


def render_table(header, rows, style):
    if style == "pipe":
        out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
        out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    elif style == "csv":
        out = [",".join(header)] + [",".join(str(c) for c in r) for r in rows]
    else:  # semicolon
        out = [";".join(header)] + [";".join(str(c) for c in r) for r in rows]
    return "\n".join(out)


def fam_table_arith(rng):
    rowkey = rng.choice(list(TABLE_ROWS))
    allrows = TABLE_ROWS[rowkey]
    cols = rng.choice(TABLE_COLS)
    cols = cols[: rng.randint(2, len(cols))]
    nr = rng.randint(3, 10)
    rows = rng.sample(allrows, nr)
    vals = {c: dict(zip(rows, rng.sample(range(1, 200), nr))) for c in cols}
    metric = rng.choice(TABLE_METRICS)
    style = rng.choice(["pipe", "csv", "semicolon"])
    table = render_table([rowkey] + cols, [[r] + [vals[c][r] for c in cols] for r in rows], style)
    state = f"{metric} by {rowkey.lower()}\n\n{table}\n"
    if rng.random() < 0.5:
        state += T(rng, ["\nAll figures are final.", "\nSource: internal dashboard export.",
                         "\nNumbers are whole units; no rows are missing."])
    rtot = {r: sum(vals[c][r] for c in cols) for r in rows}

    def q_sum():
        c = rng.choice(cols)
        s = sum(vals[c].values())
        p = T(rng, ["What is the total of column {c}?", "Sum all {c} values in the table.",
                    "Across every {k}, what do the {c} figures add up to?"], c=c, k=rowkey.lower())
        hard = [sum(vals[o].values()) for o in cols if o != c] + [s - vals[c][r] for r in rows]
        return mk_choice(rng, "col_sum", p, s, hard=hard, gen=near_numbers(rng, s, 1, 15), big_ok=True,
                         args={"op": "col_sum", "col": c})

    def q_arg():
        c = rng.choice(cols)
        hi = rng.random() < 0.6
        tgt = (max if hi else min)(rows, key=lambda r: vals[c][r])
        w = "highest" if hi else "lowest"
        p = T(rng, ["Which {k} has the {w} {c} value?", "In {c}, which {k} scored {w}?",
                    "Name the {k} with the {w} figure for {c}."], k=rowkey.lower(), w=w, c=c)
        return mk_choice(rng, "col_argmax" if hi else "col_argmin", p, tgt, hard=rows, pool=allrows,
                         args={"op": "argmax" if hi else "argmin", "col": c})

    def q_rowtot():
        r = rng.choice(rows)
        p = T(rng, ["What is the combined total for {r} across all columns?",
                    "Add up every value in the {r} row.", "Total for {r} ({cs}):"],
              r=r, cs=" + ".join(cols))
        return mk_choice(rng, "row_total", p, rtot[r], hard=[rtot[o] for o in rows],
                         gen=near_numbers(rng, rtot[r], 1, 12), big_ok=True,
                         args={"op": "row_total", "row": r})

    def q_diff():
        c = rng.choice(cols)
        a, b = rng.sample(rows, 2)
        if vals[c][a] < vals[c][b]:
            a, b = b, a
        d = vals[c][a] - vals[c][b]
        p = T(rng, ["By how much does {a} exceed {b} in {c}?", "What is {a}'s {c} minus {b}'s {c}?",
                    "Difference in {c} between {a} and {b}:"], a=a, b=b, c=c)
        return mk_choice(rng, "diff", p, d, hard=[vals[c][a] + vals[c][b], d + 10, d - 10],
                         gen=near_numbers(rng, d, 0, 8), big_ok=True,
                         args={"op": "diff", "col": c, "a": a, "b": b})

    def q_top_total():
        best = max(rows, key=lambda r: rtot[r])
        if sum(1 for r in rows if rtot[r] == rtot[best]) > 1:
            return None
        p = T(rng, ["Which {k} has the largest total across all columns?",
                    "Summing each row, which {k} comes out on top?",
                    "Which {k} has the highest overall sum?"], k=rowkey.lower())
        return mk_choice(rng, "row_total_argmax", p, best, hard=rows, pool=allrows,
                         args={"op": "row_total_argmax"})

    def q_gt():
        r, c = rng.choice(rows), rng.choice(cols)
        th = max(0, vals[c][r] + rng.choice([-20, -5, -1, 0, 3, 11]))
        p = T(rng, ["Is the {c} value for {r} greater than {t}?", "Does {r} exceed {t} in {c}?",
                    "Is {r}'s {c} figure above {t}?"], c=c, r=r, t=th)
        return mk_noul("cell_gt", p, vals[c][r] > th, {"op": "gt", "row": r, "col": c, "threshold": th})

    return state, [q_sum, q_arg, q_rowtot, q_diff, q_top_total, q_gt]


CMP_DOMAINS = [
    dict(what="mountain", names=["Alder Peak", "Mount Brenna", "Cinder Ridge", "Dover Spire", "Eagle Crest",
                                 "Frost Horn", "Granite Dome", "Hollow Peak", "Iron Summit", "Juno Crag",
                                 "Kestrel Point", "Lark Mountain", "Mica Tor", "North Needle"],
         lo=900, hi=8800, fmt="{v:,} m", max="tallest", min="lowest", more="taller",
         sent=["{n} rises to {v}.", "The summit of {n} sits at {v}.", "{n} is {v} high."]),
    dict(what="city", names=CITIES, lo=20000, hi=3000000, fmt="{v:,}", max="most populous",
         min="least populous", more="more populous",
         sent=["{n} has a population of {v}.", "{n} is home to {v} residents.",
               "The census counted {v} people in {n}."]),
    dict(what="river", names=["Arlen", "Brask", "Corran", "Delve", "Eskin", "Fyle", "Gorran", "Hask",
                              "Ister", "Jorra", "Kell", "Lune", "Marrow", "Nith"],
         lo=40, hi=4000, fmt="{v:,} km", max="longest", min="shortest", more="longer",
         sent=["The {n} river runs for {v}.", "The {n} is {v} long.", "Measured end to end, the {n} covers {v}."]),
    dict(what="laptop", names=["Zephyr 14", "Aria Pro", "Nova Book", "Quill X2", "Terra 15", "Flux Air",
                               "Orbit S", "Pulse 13", "Vega Max", "Echo Lite", "Rune 16", "Sable One"],
         lo=299, hi=3999, fmt="${v:,}", max="most expensive", min="cheapest", more="more expensive",
         sent=["The {n} costs {v}.", "The {n} is priced at {v}.", "You can buy the {n} for {v}."]),
    dict(what="runner", names=FIRST, lo=600, hi=2400, fmt="{v} seconds", max="slowest", min="fastest",
         more="slower",
         sent=["{n} finished the 5k in {v}.", "{n}'s finishing time was {v}.", "{n} crossed the line after {v}."]),
]


def fam_compare_sort(rng):
    d = rng.choice(CMP_DOMAINS)
    n = rng.randint(3, 10)
    names = rng.sample(d["names"], n)
    vs = dict(zip(names, rng.sample(range(d["lo"], d["hi"]), n)))
    sents = [rng.choice(d["sent"]).format(n=nm, v=d["fmt"].format(v=vs[nm])) for nm in names]
    intro = T(rng, ["Here are some facts about {w}s.", "Reference notes ({w}s):", "Compiled {w} data:"],
              w=d["what"])
    state = intro + " " + " ".join(sents) if rng.random() < 0.5 else intro + "\n" + "\n".join(sents)
    order = sorted(names, key=lambda x: vs[x], reverse=True)

    def q_max():
        hi = rng.random() < 0.5
        tgt = order[0] if hi else order[-1]
        w = d["max"] if hi else d["min"]
        p = T(rng, ["Which {x} is the {w}?", "Of those listed, which {x} is {w}?", "Pick the {w} {x}."],
              x=d["what"], w=w)
        return mk_choice(rng, "max" if hi else "min", p, tgt, hard=names, pool=d["names"],
                         args={"which": "max" if hi else "min"})

    def q_second():
        if n < 3:
            return None
        p = T(rng, ["Which {x} ranks second when ordered from {w} down?",
                    "Which {x} is the second {w}?", "Excluding the {w}, which {x} is next?"],
              x=d["what"], w=d["max"])
        return mk_choice(rng, "second_max", p, order[1], hard=names, pool=d["names"])

    def q_pair():
        a, b = rng.sample(names, 2)
        p = T(rng, ["Which is {m}: {a} or {b}?", "Between {a} and {b}, which one is {m}?",
                    "{a} vs {b} - which is {m}?"], m=d["more"], a=a, b=b)
        return mk_choice(rng, "pair_more", p, a if vs[a] > vs[b] else b, hard=[a, b], k=2)

    def q_noul():
        a, b = rng.sample(names, 2)
        p = T(rng, ["Is {a} {m} than {b}?", "Is it true that {a} is {m} than {b}?",
                    "Would you say {a} is {m} than {b}?"], a=a, b=b, m=d["more"])
        return mk_noul("more_than", p, vs[a] > vs[b], {"a": a, "b": b})

    def q_rank():
        r = rng.randint(1, n)
        p = T(rng, ["Sorted from {w} to {v}, which {x} is in position {r}?",
                    "Which {x} is number {r} in a ranking from {w} to {v}?",
                    "Rank the {x}s from {w} to {v}. Which is at position {r}?"],
              w=d["max"], v=d["min"], x=d["what"], r=r)
        return mk_choice(rng, "rank", p, order[r - 1], hard=names, pool=d["names"], args={"rank": r})

    return state, [q_max, q_second, q_pair, q_noul, q_rank]


LOG_MSG = {
    "INFO": ["request completed", "cache warmed", "job started", "health check ok", "config reloaded",
             "user session created"],
    "DEBUG": ["retry budget 3", "payload size 2kb", "lock acquired", "queue depth 4"],
    "WARN": ["slow response", "retrying connection", "disk usage high", "deprecated endpoint used"],
    "ERROR": ["connection refused", "timeout after 30s", "null reference in handler", "write failed",
              "token expired"],
}


def fam_log_counting(rng):
    n = rng.randint(8, 30) if rng.random() < 0.9 else rng.randint(31, 55)
    svcs = rng.sample(SERVICES, rng.randint(3, 6))
    day = rand_date(rng, 2024, 2026)
    t = rng.randint(0, 20 * 3600)
    lines = []
    for _ in range(n):
        t += rng.randint(1, 400)
        lvl = rng.choices(["INFO", "DEBUG", "WARN", "ERROR"], [5, 2, 3, 3])[0]
        s = rng.choice(svcs)
        hh, mm, ss = t // 3600 % 24, t // 60 % 60, t % 60
        lines.append(dict(time=f"{hh:02d}:{mm:02d}:{ss:02d}", lvl=lvl, svc=s, msg=rng.choice(LOG_MSG[lvl])))
    state = "\n".join(f"{day.isoformat()} {l['time']} [{l['lvl']}] {l['svc']}: {l['msg']}" for l in lines)
    lvl_count = Counter(l["lvl"] for l in lines)
    svc_count = Counter(l["svc"] for l in lines)

    def q_count_lvl():
        L = rng.choice(["INFO", "WARN", "ERROR", "DEBUG"])
        c = lvl_count[L]
        p = T(rng, ["How many {L} lines are in the log?", "Count the log entries at level {L}.",
                    "Number of [{L}] entries:"], L=L)
        return mk_choice(rng, "count_level", p, c, hard=[lvl_count[x] for x in lvl_count],
                         gen=near_numbers(rng, c, 0, 5), big_ok=True, args={"level": L})

    def q_count_svc():
        s = rng.choice(svcs)
        c = svc_count[s]
        p = T(rng, ["How many lines did the {s} service write?", "Count the entries from {s}.",
                    "How often does {s} appear as the source of a log line?"], s=s)
        return mk_choice(rng, "count_service", p, c, gen=near_numbers(rng, c, 0, 5), big_ok=True,
                         args={"service": s})

    def q_most_err():
        L = rng.choice(["WARN", "ERROR"])
        cnt = Counter(l["svc"] for l in lines if l["lvl"] == L)
        if not cnt:
            return None
        top = cnt.most_common(2)
        if len(top) > 1 and top[0][1] == top[1][1]:
            return None
        p = T(rng, ["Which service logged the most {L} entries?", "Which service has the highest {L} count?",
                    "Top source of {L} lines:"], L=L)
        return mk_choice(rng, "most_level_service", p, top[0][0], hard=svcs, pool=SERVICES, args={"level": L})

    def q_first_err():
        errs = [l for l in lines if l["lvl"] == "ERROR"]
        if not errs:
            return None
        last = rng.random() < 0.5
        e = errs[-1] if last else errs[0]
        p = T(rng, ["At what time was the {w} ERROR logged?", "Timestamp of the {w} error:",
                    "When did the {w} ERROR occur?"], w="last" if last else "first")
        return mk_choice(rng, "error_time", p, e["time"], hard=[l["time"] for l in lines],
                         args={"which": "last" if last else "first"})

    def q_any():
        s = rng.choice(svcs)
        L = rng.choice(["WARN", "ERROR"])
        truth = any(l["svc"] == s and l["lvl"] == L for l in lines)
        p = T(rng, ["Did {s} log any {L} entries?", "Is there at least one {L} line from {s}?",
                    "Does the {s} service appear with level {L}?"], s=s, L=L)
        return mk_noul("any_level_service", p, truth, {"service": s, "level": L})

    return state, [q_count_lvl, q_count_svc, q_most_err, q_first_err, q_any]


QUEUES = {"billing": "Billing department", "technical": "Technical support team",
          "refund": "Issue a refund to the customer", "account": "Account management",
          "shipping": "Shipping and logistics desk", "sales": "Sales and new business team",
          "security": "Security incident response", "other": "General inquiries (other)"}
QUEUE_KW = {"billing": ["invoice", "charged twice", "payment", "receipt"],
            "technical": ["error message", "crashes", "bug", "freezes"],
            "refund": ["refund", "money back", "reimburse"],
            "account": ["password", "username", "two-factor", "profile"],
            "shipping": ["tracking number", "delivery", "courier", "package"],
            "sales": ["pricing for", "bulk order", "quote", "upgrade options"],
            "security": ["phishing", "suspicious login", "hacked", "leaked"]}
TICKET_KW_SENT = ["I have a question about the {kw}.", "Something is off with the {kw} on my side.",
                  "This is regarding the {kw}.", "Can someone look at the {kw} issue?",
                  "Mentioning {kw} since it seems relevant."]
TICKET_FILLER = ["Thanks in advance.", "I have been a customer for years.", "Please get back to me soon.",
                 "Hope you are well.", "This started last week.", "My colleague noticed it first.",
                 "Let me know what you need from me.", "I appreciate the help."]
ESCALATE_KW = ["lawyer", "cancel my account", "urgent"]


def fam_ticket_routing(rng):
    long_labels = rng.random() < 0.5
    lab = (lambda q: QUEUES[q]) if long_labels else (lambda q: q)
    rq = rng.sample([q for q in QUEUES if q != "other"], rng.randint(3, 5))
    rules = [(q, rng.sample(QUEUE_KW[q], rng.randint(2, len(QUEUE_KW[q])))) for q in rq]
    lines = [T(rng, ["Routing rules (checked in order; the first matching rule wins):",
                     "Ticket triage policy. Apply rules top to bottom and stop at the first match:",
                     "Support routing - evaluate rules in order, first match decides:"])]
    for i, (q, kws) in enumerate(rules, 1):
        lines.append(f"{i}. If the ticket mentions " + ", ".join(f"'{k}'" for k in kws[:-1])
                     + f" or '{kws[-1]}' -> {lab(q)}")
    lines.append(f"{len(rules) + 1}. Otherwise -> {lab('other')}")
    lines.append("Escalation: escalate any ticket that mentions " + ", ".join(f"'{k}'" for k in ESCALATE_KW)
                 + ", or that comes from a Platinum-tier customer.")
    lines.append("")
    tickets = []
    for j in range(rng.randint(1, 3)):
        tid = f"T-{rng.randint(10000, 99999)}"
        tier = rng.choice(["Standard", "Gold", "Platinum"])
        parts = rng.sample(TICKET_FILLER, rng.randint(1, 3))
        ncat = rng.choices([0, 1, 2], [2, 5, 3])[0]
        for q in rng.sample(list(QUEUE_KW), ncat):
            # mostly keywords that are in the rule list, occasionally an off-list synonym
            rk = dict(rules).get(q)
            kw = rng.choice(rk) if rk and rng.random() < 0.85 else rng.choice(QUEUE_KW[q])
            parts.insert(rng.randint(0, len(parts)), rng.choice(TICKET_KW_SENT).format(kw=kw))
        if rng.random() < 0.2:
            parts.insert(rng.randint(0, len(parts)), f"This is {rng.choice(ESCALATE_KW)}."
                         if rng.random() < 0.5 else "Honestly I might talk to a lawyer.")
        body = " ".join(parts)
        tickets.append(dict(id=tid, tier=tier, body=body))
        lines.append(f"Ticket {tid} (customer tier: {tier}): \"{body}\"")
    state = "\n".join(lines)

    def mentions(text, kw):
        return re.search(r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])", text.lower()) is not None

    def route(t):
        for i, (q, kws) in enumerate(rules, 1):
            if any(mentions(t["body"], k) for k in kws):
                return q, i
        return "other", len(rules) + 1

    def q_route():
        t = rng.choice(tickets)
        q, _ = route(t)
        extra = [lab(x) for x in QUEUES if x not in rq and x != "other"]
        p = T(rng, ["Where should ticket {i} be routed?", "Which queue does ticket {i} go to?",
                    "Apply the routing rules to {i}. Destination?"], i=t["id"])
        return mk_choice(rng, "route", p, lab(q), hard=[lab(x) for x in rq] + [lab("other")],
                         pool=extra, k=rng.randint(len(rq) + 1, len(rq) + 1 + len(extra)),
                         args={"ticket": t["id"], "queue": q})

    def q_rule_no():
        t = rng.choice(tickets)
        _, i = route(t)
        p = T(rng, ["Which rule number decides ticket {t}?", "Which rule fires first for ticket {t}?",
                    "Ticket {t} is matched by rule:"], t=t["id"])
        return mk_choice(rng, "rule_number", p, f"rule {i}",
                         hard=[f"rule {j}" for j in range(1, len(rules) + 2)], k=len(rules) + 1)

    def q_match():
        t = rng.choice(tickets)
        hit = [j for j in range(1, len(rules) + 1) if any(mentions(t["body"], k) for k in rules[j - 1][1])]
        i = rng.choice(hit) if hit and rng.random() < 0.7 else rng.randint(1, len(rules))
        truth = any(mentions(t["body"], k) for k in rules[i - 1][1])
        p = T(rng, ["Does ticket {t} satisfy the condition of rule {i} (ignoring rule order)?",
                    "Taken on its own, does rule {i}'s condition match ticket {t}?",
                    "Would rule {i} match ticket {t} if it were checked alone?"], t=t["id"], i=i)
        return mk_noul("rule_matches", p, truth, {"ticket": t["id"], "rule": i})

    def q_escalate():
        t = rng.choice(tickets)
        truth = t["tier"] == "Platinum" or any(mentions(t["body"], k) for k in ESCALATE_KW)
        p = T(rng, ["Should ticket {t} be escalated?", "Does ticket {t} meet the escalation policy?",
                    "Escalate {t}?"], t=t["id"])
        return mk_noul("escalate", p, truth, {"ticket": t["id"]})

    return state, [q_route, q_route, q_rule_no, q_match, q_escalate]


VENUES = ["Room 4B", "main auditorium", "remote", "HQ floor 2", "partner office", "Lab 1", "cafeteria"]
EVENTS = ["Project kickoff", "Design review", "Budget approval", "Beta launch", "Security audit",
          "Team offsite", "Board meeting", "Code freeze", "Customer workshop", "Contract renewal",
          "Hiring fair", "Product launch", "Quarterly review", "Data migration"]


def fam_date_reasoning(rng):
    n = rng.randint(3, 10)
    evs = rng.sample(EVENTS, n)
    base = rand_date(rng, 2020, 2027)
    offs = rng.sample(range(-200, 400), n)
    dates = {e: base + dt.timedelta(days=o) for e, o in zip(evs, offs)}
    ref = base + dt.timedelta(days=rng.randint(-100, 300))
    lines = [T(rng, ["Upcoming calendar:", "Key dates for the programme:", "Milestone schedule:"])]
    for e in evs:
        where = f" ({rng.choice(VENUES)})" if rng.random() < 0.5 else ""
        lines.append(f"- {e}: {fmt_date(dates[e], rng.choice(['iso', 'long', 'short']))}{where}")
    lines.append(f"For reference, {ref.isoformat()} is a {WEEKDAYS[ref.weekday()]}.")
    state = "\n".join(lines)
    order = sorted(evs, key=lambda e: dates[e])

    def q_first():
        last = rng.random() < 0.5
        p = T(rng, ["Which event happens {w}?", "What is the {w} milestone on the calendar?",
                    "Chronologically, which event comes {w2}?"],
              w="last" if last else "first", w2="last" if last else "first")
        return mk_choice(rng, "last_event" if last else "first_event", p, order[-1] if last else order[0],
                         hard=evs, pool=EVENTS)

    def q_gap():
        a, b = rng.sample(evs, 2)
        g = abs((dates[b] - dates[a]).days)
        p = T(rng, ["How many days are there between the {a} and the {b}?",
                    "Number of days separating {a} and {b}:", "How far apart, in days, are {a} and {b}?"],
              a=a, b=b)
        return mk_choice(rng, "day_gap", p, g, hard=[g + 1, g - 1, g + 7, g - 7, g + 30],
                         gen=near_numbers(rng, g, 0, 10), big_ok=True,
                         args={"a": dates[a].isoformat(), "b": dates[b].isoformat()})

    def q_before():
        a, b = rng.sample(evs, 2)
        p = T(rng, ["Does the {a} happen before the {b}?", "Is the {a} scheduled earlier than the {b}?",
                    "Will the {a} take place before the {b}?"], a=a, b=b)
        return mk_noul("before", p, dates[a] < dates[b], {"a": dates[a].isoformat(), "b": dates[b].isoformat()})

    def q_weekday():
        e = rng.choice(evs)
        d = dates[e]
        p = T(rng, ["On what day of the week does the {e} fall?", "Which weekday is the {e}?",
                    "The {e} is on a:"], e=e)
        return mk_choice(rng, "weekday", p, WEEKDAYS[d.weekday()], pool=WEEKDAYS,
                         k=rng.randint(2, 7), args={"date": d.isoformat()})

    def q_after():
        e = rng.choice(evs)
        k = rng.randint(2, 45)
        d = dates[e] + dt.timedelta(days=k)
        p = T(rng, ["What date is {k} days after the {e}? (YYYY-MM-DD)",
                    "Add {k} days to the date of the {e}. Result?",
                    "Which date falls exactly {k} days after the {e}?"], k=k, e=e)
        hard = [(d + dt.timedelta(days=x)).isoformat() for x in (-1, 1, -7, 7, -k, 30)]
        return mk_choice(rng, "date_after", p, d.isoformat(), hard=hard,
                         args={"date": dates[e].isoformat(), "days": k})

    return state, [q_first, q_gap, q_before, q_weekday, q_after]


UNIT_CATS = {
    "mass": dict(units={"kg": 1000.0, "g": 1.0, "lb": 453.59237, "oz": 28.349523125}, lo=50, hi=30000,
                 item="Parcel", verb="weighs", max="heaviest", min="lightest", more="heavier"),
    "length": dict(units={"km": 1000.0, "m": 1.0, "mi": 1609.344, "ft": 0.3048}, lo=200, hi=20000,
                   item="Trail", verb="is", max="longest", min="shortest", more="longer"),
    "time": dict(units={"h": 3600.0, "min": 60.0, "s": 1.0}, lo=90, hi=30000, item="Task", verb="takes",
                 max="longest", min="shortest", more="longer"),
    "volume": dict(units={"L": 1000.0, "mL": 1.0, "gal": 3785.411784, "cup": 236.5882365}, lo=100,
                   hi=40000, item="Tank", verb="holds", max="largest", min="smallest", more="larger"),
}


def sig(x, n=3):
    if x == 0:
        return "0"
    d = n - int(math.floor(math.log10(abs(x)))) - 1
    v = round(x, d)
    s = f"{v:.{max(d, 0)}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def fam_unit_conversion(rng):
    cname = rng.choice(list(UNIT_CATS))
    c = UNIT_CATS[cname]
    units = list(c["units"])
    n = rng.randint(3, 7)
    for _ in range(50):
        items = []
        for i in range(n):
            base = math.exp(rng.uniform(math.log(c["lo"]), math.log(c["hi"])))
            u = rng.choice(units)
            shown = sig(base / c["units"][u])
            items.append(dict(name=f"{c['item']} {chr(65 + i)}", unit=u, shown=shown,
                              base=float(shown) * c["units"][u]))
        bs = sorted(x["base"] for x in items)
        if all(b2 / b1 > 1.05 for b1, b2 in zip(bs, bs[1:])):
            break
    else:
        return None
    lines = [T(rng, ["Measurements recorded today:", "Item sheet:", "Logged quantities:"])]
    for x in items:
        lines.append(f"{x['name']} {c['verb']} {x['shown']} {x['unit']}.")
    lines.append(T(rng, ["Conversions: 1 kg = 1000 g, 1 lb = 453.59237 g, 1 oz = 28.349523125 g; "
                         "1 km = 1000 m, 1 mi = 1609.344 m, 1 ft = 0.3048 m; 1 h = 60 min = 3600 s; "
                         "1 L = 1000 mL, 1 gal = 3.785411784 L, 1 cup = 236.5882365 mL.",
                         "Use standard unit conversions."]))
    state = "\n".join(lines)
    names = [x["name"] for x in items]
    by = {x["name"]: x for x in items}

    def q_max():
        hi = rng.random() < 0.5
        tgt = (max if hi else min)(items, key=lambda x: x["base"])["name"]
        w = c["max"] if hi else c["min"]
        p = T(rng, ["After converting to a common unit, which item is the {w}?",
                    "Which {it} is the {w}?", "Accounting for units, the {w} one is:"], w=w,
              it=c["item"].lower())
        return mk_choice(rng, "max" if hi else "min", p, tgt, hard=names)

    def q_cmp():
        a, b = rng.sample(items, 2)
        p = T(rng, ["Is {a} {m} than {b}?", "Taking units into account, is {a} {m} than {b}?",
                    "Compare {a} and {b}: is {a} {m}?"], a=a["name"], b=b["name"], m=c["more"])
        return mk_noul("more_than", p, a["base"] > b["base"], {"a": a["name"], "b": b["name"]})

    def q_conv():
        x = rng.choice(items)
        tu = rng.choice([u for u in units if u != x["unit"]])
        val = x["base"] / c["units"][tu]
        if not (0.05 <= val < 1e7):
            return None
        f = lambda v: f"{v:.2f} {tu}"
        wrong = [val * 10, val / 10, val * 1000, val / 1000, float(x["shown"]),
                 float(x["shown"]) * c["units"][tu] / c["units"][x["unit"]] if c["units"][x["unit"]] else val,
                 val * 1.1, val * 0.9, val * 2.2, val / 2.2, val * 60, val / 60]
        p = T(rng, ["Convert {n} to {u} (2 decimal places).", "What is {n} expressed in {u}? Round to 2 decimals.",
                    "{n} in {u}, rounded to two decimal places:"], n=x["name"], u=tu)
        return mk_choice(rng, "convert", p, f(val), hard=[f(w) for w in wrong if w > 0],
                         gen=lambda k: [f(val * rng.uniform(0.5, 1.5)) for _ in range(3 * k)],
                         args={"item": x["name"], "unit": tu})

    return state, [q_max, q_cmp, q_conv, q_conv]


LIST_NAMES = ["Approved vendors", "Blocked accounts", "Beta testers", "Priority clients", "Audit watchlist",
              "Newsletter opt-outs"]


def fam_set_membership(rng):
    lnames = rng.sample(LIST_NAMES, rng.randint(2, 3))
    universe = rng.sample(COMPANIES, rng.randint(8, 18))
    lists = {ln: sorted(rng.sample(universe, rng.randint(3, min(9, len(universe))))) for ln in lnames}
    for v in lists.values():
        rng.shuffle(v)
    mode = rng.random()
    if mode < 0.35:
        state = {"lists": lists, "note": "Membership is exact; names not listed are not members."}
    elif mode < 0.7:
        state = "\n".join(f"{ln}: " + ", ".join(v) + "." for ln, v in lists.items())
    else:
        state = "\n\n".join(f"{ln}:\n" + "\n".join(f"  - {x}" for x in v) for ln, v in lists.items())
    others = [x for x in universe if not any(x in v for v in lists.values())]

    def pick(ln, member):
        pool = lists[ln] if member else [x for x in universe + rng.sample(COMPANIES, 5) if x not in lists[ln]]
        return rng.choice(pool)

    def q_in():
        ln = rng.choice(lnames)
        x = pick(ln, rng.random() < 0.5)
        neg = rng.random() < 0.35
        if neg:
            p = T(rng, ["Is {x} absent from the {l} list?", "Is it true that {x} is NOT on {l}?",
                        "Is {x} missing from {l}?"], x=x, l=ln)
            return mk_noul("not_in", p, x not in lists[ln], {"item": x, "list": ln, "op": "not_in"})
        p = T(rng, ["Is {x} on the {l} list?", "Does {l} include {x}?", "Is {x} one of the {l}?"], x=x, l=ln)
        return mk_noul("in", p, x in lists[ln], {"item": x, "list": ln, "op": "in"})

    def q_both():
        if len(lnames) < 2:
            return None
        a, b = rng.sample(lnames, 2)
        both = rng.random() < 0.5
        good = [x for x in lists[a] if (x in lists[b]) == both]
        x = rng.choice(good) if good and rng.random() < 0.5 else rng.choice(universe)
        if both:
            p = T(rng, ["Is {x} on both {a} and {b}?", "Does {x} appear in {a} as well as in {b}?",
                        "Is {x} a member of both lists, {a} and {b}?"], x=x, a=a, b=b)
            return mk_noul("both", p, x in lists[a] and x in lists[b], {"item": x, "lists": [a, b], "op": "both"})
        p = T(rng, ["Is {x} on {a} but not on {b}?", "Is {x} listed in {a} while missing from {b}?",
                    "Does {x} appear in {a} and not in {b}?"], x=x, a=a, b=b)
        return mk_noul("a_not_b", p, x in lists[a] and x not in lists[b],
                       {"item": x, "lists": [a, b], "op": "a_not_b"})

    def q_which_member():
        ln = rng.choice(lnames)
        x = rng.choice(lists[ln])
        non = [y for y in universe if y not in lists[ln]]
        p = T(rng, ["Which of these is on the {l} list?", "Which option appears in {l}?",
                    "Pick the name that belongs to {l}."], l=ln)
        return mk_choice(rng, "which_member", p, x, hard=non,
                         gen=lambda k: [y for y in rng.sample(COMPANIES, len(COMPANIES)) if y not in lists[ln]],
                         big_ok=True, args={"list": ln})

    def q_which_list():
        x = rng.choice(universe)
        where = [ln for ln in lnames if x in lists[ln]]
        if len(where) > 1:
            return None
        ans = where[0] if where else "none of the lists"
        p = T(rng, ["Which list contains {x}?", "Where is {x} listed?", "{x} appears on which list?"], x=x)
        return mk_choice(rng, "which_list", p, ans, hard=lnames + ["none of the lists"],
                         k=len(lnames) + 1, args={"item": x})

    return state, [q_in, q_in, q_both, q_which_member, q_which_list]


TTT_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
TTT_NAMED = ["top-left", "top-middle", "top-right", "middle-left", "center", "middle-right", "bottom-left",
             "bottom-middle", "bottom-right"]
TTT_A1 = ["A1", "B1", "C1", "A2", "B2", "C2", "A3", "B3", "C3"]


def ttt_winner(b):
    for l in TTT_LINES:
        if b[l[0]] != "." and b[l[0]] == b[l[1]] == b[l[2]]:
            return b[l[0]]
    return None


def fam_tictactoe(rng):
    b = ["."] * 9
    moves = rng.randint(1, 9)
    player = "X"
    for _ in range(moves):
        empt = [i for i in range(9) if b[i] == "."]
        b[rng.choice(empt)] = player
        player = "O" if player == "X" else "X"
        if ttt_winner(b):
            break
    win = ttt_winner(b)
    full = "." not in b
    to_move = "X" if b.count("X") == b.count("O") else "O"
    a1 = rng.random() < 0.5
    cellname = TTT_A1 if a1 else TTT_NAMED
    if a1:
        grid = "   A B C\n" + "\n".join(f"{r + 1}  " + " ".join(b[3 * r:3 * r + 3]) for r in range(3))
        legend = "Columns A-C run left to right, rows 1-3 top to bottom. '.' is an empty cell."
    else:
        grid = "\n---------\n".join(" | ".join(b[3 * r:3 * r + 3]) for r in range(3))
        legend = ("Cells are named top-left, top-middle, top-right, middle-left, center, middle-right, "
                  "bottom-left, bottom-middle, bottom-right. '.' is an empty cell.")
    head = T(rng, ["Tic-tac-toe position (X always moves first).", "Current noughts-and-crosses board; X opened the game.",
                   "Game #{g} of tic-tac-toe. X started."], g=rng.randint(1, 999))
    state = f"{head}\n{legend}\n\n{grid}\n"
    long_lab = rng.random() < 0.5
    wl = {"X": "player X (crosses) has won" if long_lab else "X",
          "O": "player O (noughts) has won" if long_lab else "O",
          None: "nobody has won" if long_lab else "nobody"}

    def q_winner():
        p = T(rng, ["Who has won this game?", "Has anyone completed three in a row? If so, who?",
                    "Winner of the position shown:"])
        return mk_choice(rng, "winner", p, wl[win], hard=list(wl.values()), k=3, args={"op": "winner"})

    def q_turn():
        if win or full:
            return None
        p = T(rng, ["Whose turn is it?", "Which player moves next?", "Who is to move?"])
        return mk_choice(rng, "turn", p, to_move, hard=["X", "O"], k=2, args={"op": "turn"})

    def q_legal():
        emp = [i for i in range(9) if b[i] == "."]
        occ = [i for i in range(9) if b[i] != "."]
        if not emp or not occ or win:
            return None
        c = rng.choice(emp)
        p = T(rng, ["Which of these cells is a legal move?", "Which listed square is still empty?",
                    "Where can the next piece legally be placed?"])
        return mk_choice(rng, "legal_move", p, cellname[c], hard=[cellname[i] for i in occ],
                         args={"op": "legal", "names": "A1" if a1 else "named"})

    def q_winmove():
        if win or full:
            return None
        emp = [i for i in range(9) if b[i] == "."]
        wins = [i for i in emp if ttt_winner(b[:i] + [to_move] + b[i + 1:]) == to_move]
        if len(wins) != 1:
            return None
        p = T(rng, ["Which move wins immediately for {p}?", "{p} to play: which square completes three in a row?",
                    "Where should {p} move to win right now?"], p=to_move)
        return mk_choice(rng, "winning_move", p, cellname[wins[0]], hard=[cellname[i] for i in emp],
                         args={"op": "winmove", "player": to_move, "names": "A1" if a1 else "named"})

    def q_empty():
        c = rng.randrange(9)
        p = T(rng, ["Is {c} empty?", "Is the {c} cell unoccupied?", "Is there no piece on {c}?"], c=cellname[c])
        return mk_noul("cell_empty", p, b[c] == ".", {"cell": cellname[c]})

    def q_over():
        p = T(rng, ["Is the game over?", "Has the game ended (win or full board)?", "Is play finished?"])
        return mk_noul("game_over", p, bool(win) or full, {"op": "over"})

    return state, [q_winner, q_turn, q_legal, q_winmove, q_empty, q_over]


PARTS = ["hex bolts", "gaskets", "valves", "bearings", "hinges", "fuses", "washers", "filters", "springs",
         "cable ties", "brackets", "pulleys", "o-rings", "rivets", "clamps", "sensors", "relays", "nozzles"]
ACTIONS = {"reorder": "Place a reorder with the supplier", "hold": "Hold current stock (no action)",
           "transfer": "Transfer surplus to another warehouse"}


def fam_inventory_levels(rng):
    n = rng.randint(3, 9)
    items = rng.sample(PARTS, n)
    usage = {i: rng.randint(1, 40) for i in items}
    cover_target = {i: rng.choice([0.5, 1, 2, 4, 6, 10, 15, 25, 40, 70, 100]) * rng.uniform(0.7, 1.3) for i in items}
    onhand = {i: max(0, int(usage[i] * cover_target[i])) for i in items}
    rop = {i: usage[i] * rng.randint(3, 10) for i in items}
    five = rng.random() < 0.4
    if five:
        th = sorted(rng.sample([2, 3, 5, 7, 10, 14, 21, 30], 4))
        levels = ["1", "2", "3", "4", "5"]
        rule = (f"Stock level score (days of cover = on hand / daily usage): 1 if cover < {th[0]}, 2 if < {th[1]}, "
                f"3 if < {th[2]}, 4 if < {th[3]}, otherwise 5.")
    else:
        th = sorted(rng.sample([2, 3, 5, 7, 10, 14, 21, 30, 45], 3))
        levels = ["critical", "low", "ok", "overstocked"]
        rule = (f"Status (days of cover = on hand / daily usage): critical if cover < {th[0]}, low if < {th[1]}, "
                f"ok if < {th[2]}, otherwise overstocked.")
    long_lab = rng.random() < 0.5
    alab = (lambda a: ACTIONS[a]) if long_lab else (lambda a: a)
    rule2 = (f"Action: {alab('reorder')} if on hand is at or below the reorder point; "
             f"{alab('transfer')} if cover exceeds 60 days; otherwise {alab('hold')}.")

    def level(i):
        cov = onhand[i] / usage[i]  # exact comparison below uses integers
        for j, t in enumerate(th):
            if onhand[i] < t * usage[i]:
                return j
        return len(th)

    def action(i):
        if onhand[i] <= rop[i]:
            return "reorder"
        if onhand[i] > 60 * usage[i]:
            return "transfer"
        return "hold"

    rows = [[i, onhand[i], usage[i], rop[i]] for i in items]
    if rng.random() < 0.5:
        state = {"warehouse": f"WH-{rng.randint(1, 40)}", "rules": [rule, rule2],
                 "items": [{"item": i, "on_hand": onhand[i], "daily_usage": usage[i], "reorder_point": rop[i]}
                           for i in items]}
    else:
        state = (f"Warehouse WH-{rng.randint(1, 40)} stock report\n\n"
                 + render_table(["item", "on_hand", "daily_usage", "reorder_point"], rows, rng.choice(["pipe", "csv"]))
                 + "\n\n" + rule + "\n" + rule2)

    def q_level():
        i = rng.choice(items)
        p = T(rng, ["What is the stock level of {i}?", "Classify the stock position of {i}.",
                    "Using the rule, how would you rate {i}?"], i=i)
        return mk_score("stock_level", p, levels, level(i), {"item": i})

    def q_fewest():
        cov = {i: onhand[i] / usage[i] for i in items}
        best = min(items, key=cov.get)
        if sum(1 for i in items if cov[i] == cov[best]) > 1:
            return None
        p = T(rng, ["Which item has the fewest days of cover?", "Which part will run out first at current usage?",
                    "Lowest days-of-cover item:"])
        return mk_choice(rng, "fewest_cover", p, best, hard=items, pool=PARTS)

    def q_reorder():
        i = rng.choice(items)
        p = T(rng, ["Is {i} at or below its reorder point?", "Does {i} need to be reordered?",
                    "Has {i} hit the reorder point?"], i=i)
        return mk_noul("needs_reorder", p, onhand[i] <= rop[i], {"item": i})

    def q_action():
        i = rng.choice(items)
        p = T(rng, ["What action applies to {i}?", "Per the action rule, what should happen with {i}?",
                    "Recommended action for {i}:"], i=i)
        return mk_choice(rng, "action", p, alab(action(i)), hard=[alab(a) for a in ACTIONS], k=3, args={"item": i})

    return state, [q_level, q_level, q_fewest, q_reorder, q_action]


def fam_string_props(rng):
    n = rng.randint(6, 22)
    words = rng.sample(WORDS, n)
    mode = rng.random()
    if mode < 0.4:
        state = "Word list: " + ", ".join(words) + "."
    elif mode < 0.7:
        state = "Vocabulary cards drawn this round:\n" + "\n".join(f"{i + 1}. {w}" for i, w in enumerate(words))
    else:
        state = {"game": "word round", "round": rng.randint(1, 20), "words": words}
    notw = [w for w in WORDS if w not in words]

    def q_len():
        long = rng.random() < 0.5
        tgt = (max if long else min)(words, key=len)
        if sum(1 for w in words if len(w) == len(tgt)) > 1:
            return None
        w = "longest" if long else "shortest"
        p = T(rng, ["Which word in the list is the {w}?", "Pick the {w} word (by number of letters).",
                    "Among the listed words, which has the {w} spelling?"], w=w)
        return mk_choice(rng, w, p, tgt, hard=words, args={"which": w})

    def q_alpha():
        first = rng.random() < 0.5
        tgt = sorted(words)[0 if first else -1]
        p = T(rng, ["Which listed word comes {w} alphabetically?", "Sorted A-Z, which word is {w}?",
                    "Alphabetically {w} word in the list:"], w="first" if first else "last")
        return mk_choice(rng, "alpha_first" if first else "alpha_last", p, tgt, hard=words)

    def q_starts():
        by = defaultdict(list)
        for w in words:
            by[w[0]].append(w)
        uniq = [l for l, ws in by.items() if len(ws) == 1]
        if not uniq:
            return None
        l = rng.choice(uniq)
        p = T(rng, ["Which word in the list starts with '{l}'?", "Find the listed word beginning with the letter {l}.",
                    "The only word in the list that begins with '{l}' is:"], l=l)
        return mk_choice(rng, "starts_with", p, by[l][0], hard=[w for w in words if w[0] != l],
                         pool=[w for w in notw if w[0] != l], args={"letter": l})

    def q_count_letter():
        l = rng.choice("aeiorstnl")
        c = sum(1 for w in words if l in w)
        p = T(rng, ["How many words in the list contain the letter '{l}'?", "Count the listed words with an '{l}' in them.",
                    "Number of words containing '{l}':"], l=l)
        return mk_choice(rng, "count_letter", p, c, gen=near_numbers(rng, c, 0, 4), args={"letter": l})

    def q_in():
        w = rng.choice(words) if rng.random() < 0.5 else rng.choice(notw)
        p = T(rng, ["Is '{w}' in the list?", "Does the list include the word {w}?", "Was '{w}' one of the words?"], w=w)
        return mk_noul("in_list", p, w in words, {"word": w})

    def q_which_in():
        w = rng.choice(words)
        p = T(rng, ["Which of these words appears in the list?", "Which option was one of the listed words?",
                    "Select the word that is in the list."])
        return mk_choice(rng, "which_in", p, w, pool=notw, big_ok=True)

    return state, [q_len, q_alpha, q_starts, q_count_letter, q_in, q_which_in]


SEV_FLAGS = {
    "outage": (["Customers could not reach the service at all.", "The public site was fully down for users."],
               ["Customers were not affected by any downtime.", "The customer-facing service stayed up throughout."]),
    "data_loss": (["Some records were permanently lost.", "Data written during the window could not be recovered."],
                  ["No data was lost.", "All data was recovered intact."]),
    "security": (["The cause was an unauthorised access attempt that succeeded.", "Credentials were exposed publicly."],
                 ["There is no security component to this incident.", "Security review found no breach."]),
    "workaround": (["A documented workaround is available.", "Users can switch to the backup flow as a workaround."],
                   ["No workaround exists.", "There is currently no workaround."]),
}


def fam_severity_scoring(rng):
    five = rng.random() < 0.5
    levels = ["1", "2", "3", "4", "5"] if five else ["low", "medium", "high", "critical"]
    th_users = rng.choice([100, 500, 1000, 5000])
    rubric = ("Severity rubric: start at 1 point. Add 2 if there is a customer-facing outage. "
              f"Add 1 if more than {th_users:,} users are affected. Add 1 if any data was lost. "
              "Add 2 if there is a security breach. Subtract 1 if a workaround exists. "
              "Clamp the result to the range 1-5.")
    if not five:
        rubric += " Map points to labels: 1 = low, 2 = medium, 3 = high, 4 or 5 = critical."
    rubric += " Page the on-call manager when the points total is 4 or more."
    incs = []
    for _ in range(rng.randint(1, 3)):
        f = {k: rng.random() < pr for k, pr in [("outage", .5), ("data_loss", .3), ("security", .2), ("workaround", .35)]}
        users = int(th_users * math.exp(rng.uniform(-2.5, 2.5)))
        if users == th_users:
            users += 1
        pts = 1 + 2 * f["outage"] + (users > th_users) + f["data_loss"] + 2 * f["security"] - f["workaround"]
        pts = max(1, min(5, pts))
        sents = [rng.choice(SEV_FLAGS[k][0 if v else 1]) for k, v in f.items()]
        sents.append(f"Approximately {users:,} users were affected." if rng.random() < 0.5
                     else f"Affected users: {users:,}.")
        rng.shuffle(sents)
        incs.append(dict(id=f"INC-{rng.randint(100, 999)}", pts=pts, text=" ".join(sents),
                         svc=rng.choice(SERVICES)))
    if len({i["id"] for i in incs}) < len(incs):
        incs = incs[:1]
    state = rubric + "\n\n" + "\n".join(f"{i['id']} ({i['svc']}): {i['text']}" for i in incs)
    idx = lambda pts: pts - 1 if five else min(pts, 4) - 1

    def q_sev():
        i = rng.choice(incs)
        p = T(rng, ["What severity does {i} get under the rubric?", "Score incident {i}.",
                    "Apply the rubric: severity of {i}?"], i=i["id"])
        return mk_score("severity", p, levels, idx(i["pts"]), {"incident": i["id"], "points": i["pts"]})

    def q_page():
        i = rng.choice(incs)
        p = T(rng, ["Should the on-call manager be paged for {i}?", "Does {i} require paging the manager?",
                    "Page the on-call manager for {i}?"], i=i["id"])
        return mk_noul("page_manager", p, i["pts"] >= 4, {"incident": i["id"]})

    def q_worst():
        if len(incs) < 2:
            return None
        best = max(incs, key=lambda i: i["pts"])
        if sum(1 for i in incs if i["pts"] == best["pts"]) > 1:
            return None
        p = T(rng, ["Which incident scores the most rubric points?", "Which incident is most severe per the rubric?",
                    "Highest-severity incident:"])
        return mk_choice(rng, "worst_incident", p, best["id"], hard=[i["id"] for i in incs])

    return state, [q_sev, q_sev, q_page, q_worst]


GROCERIES = ["olive oil", "coffee beans", "rice", "oat milk", "dark chocolate", "cheddar", "pasta", "honey",
             "green tea", "almonds", "tomato sauce", "granola", "salmon fillet", "sourdough loaf", "avocados"]


def fam_price_calc(rng):
    n = rng.randint(2, 6)
    items = rng.sample(GROCERIES, n)
    price = {i: 20 * rng.randint(10, 200) for i in items}   # multiples of 20 cents -> exact discounts
    qty = {i: rng.randint(1, 5) for i in items}
    pct = rng.choice([10, 20, 25, 50])
    sub = sum(price[i] * qty[i] for i in items)
    disc_th = 500 * max(1, round(sub * rng.uniform(0.6, 1.4) / 500))      # multiple of $5 near subtotal
    ship = 100 * rng.choice([3, 5, 7]) + rng.choice([0, 50, 99])
    disc = sub * pct // 100 if sub > disc_th else 0
    after = sub - disc
    free_th = 500 * max(1, round(after * rng.uniform(0.6, 1.4) / 500))
    ship_cost = 0 if after >= free_th else ship
    total = after + ship_cost
    lines = [T(rng, ["Order summary:", "Shopping cart:", "Basket contents:"])]
    for i in items:
        lines.append(f"- {i}: {qty[i]} x {money(price[i])}")
    lines.append(f"Promotion: {pct}% off the subtotal when the subtotal is over {money(disc_th)}.")
    lines.append(f"Shipping costs {money(ship)}, free when the discounted subtotal is at least {money(free_th)}.")
    state = "\n".join(lines)
    mgen = lambda c: (lambda k: [money(max(1, c + 20 * rng.randint(-6 * k, 6 * k))) for _ in range(4 * k)])

    def q_sub():
        p = T(rng, ["What is the subtotal before any discount?", "Sum of all line items (before promotions):",
                    "Cart subtotal?"])
        return mk_choice(rng, "subtotal", p, money(sub), hard=[money(after), money(total), money(sub + ship)],
                         gen=mgen(sub), big_ok=True, args={"cents": sub})

    def q_total():
        p = T(rng, ["What is the final amount to pay, including shipping?", "Order total after discount and shipping:",
                    "How much will the customer be charged in total?"])
        return mk_choice(rng, "total", p, money(total),
                         hard=[money(sub), money(after), money(sub + ship), money(after + ship), money(sub - disc + 0)],
                         gen=mgen(total), big_ok=True, args={"cents": total})

    def q_biggest():
        lt = {i: price[i] * qty[i] for i in items}
        best = max(items, key=lt.get)
        if n < 2 or sum(1 for i in items if lt[i] == lt[best]) > 1:
            return None
        p = T(rng, ["Which item contributes the most to the subtotal?", "Largest line item (quantity x price):",
                    "Which product costs the most in total on this order?"])
        return mk_choice(rng, "biggest_line", p, best, hard=items, pool=GROCERIES)

    def q_free():
        p = T(rng, ["Does this order get free shipping?", "Is shipping free for this basket?",
                    "Will shipping be waived?"])
        return mk_noul("free_shipping", p, ship_cost == 0)

    def q_disc():
        p = T(rng, ["Does the promotion apply to this order?", "Is the order eligible for the percentage discount?",
                    "Will the {p}% discount be applied?"], p=pct)
        return mk_noul("discount_applies", p, disc > 0)

    return state, [q_sub, q_total, q_biggest, q_free, q_disc]


# ---------------------------------------------------------------------------------------------------
# OOD families
# ---------------------------------------------------------------------------------------------------
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
DIR_LONG = {"up": "move up (north)", "down": "move down (south)", "left": "move left (west)",
            "right": "move right (east)"}


def grid_bfs(grid, src):
    H, W = len(grid), len(grid[0])
    dist = {src: 0}
    dq = deque([src])
    while dq:
        r, c = dq.popleft()
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != "#" and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                dq.append((nr, nc))
    return dist


def fam_grid_navigation(rng):
    want_reach = rng.random() < 0.55
    for _ in range(30):
        H, W = rng.randint(4, 9), rng.randint(4, 10)
        dens = rng.uniform(0.1, 0.45)
        g = [["#" if rng.random() < dens else "." for _ in range(W)] for _ in range(H)]
        cells = [(r, c) for r in range(H) for c in range(W)]
        s, t = rng.sample(cells, 2)
        g[s[0]][s[1]], g[t[0]][t[1]] = "S", "G"
        grid = ["".join(r) for r in g]
        dist = grid_bfs(grid, t)
        reach = s in dist
        if reach == want_reach:
            break
    long_lab = rng.random() < 0.5
    lab = (lambda d: DIR_LONG[d]) if long_lab else (lambda d: d)
    state = (T(rng, ["Maze map (row 1 is the top).", "Warehouse floor plan for the robot.", "Dungeon level layout."])
             + " Legend: S = start, G = goal, # = wall, . = open floor. Moves are one cell up, down, left or right;"
             " walls and the map edge cannot be crossed.\n\n" + "\n".join(grid) + "\n")

    def q_first():
        if not reach or dist[s] == 0:
            return None
        good = [d for d, (dr, dc) in DIRS.items() if dist.get((s[0] + dr, s[1] + dc), 1e9) == dist[s] - 1]
        if len(good) != 1:
            return None
        p = T(rng, ["Which first move from S lies on a shortest path to G?", "The agent is at S. Which way should it step first to reach G as fast as possible?",
                    "First step from S along the shortest route to G:"])
        return mk_choice(rng, "first_move", p, lab(good[0]), hard=[lab(d) for d in DIRS], k=rng.randint(2, 4),
                         args={"op": "first_move"})

    def q_len():
        if not reach:
            return None
        L = dist[s]
        p = T(rng, ["How many moves does the shortest path from S to G take?", "Minimum number of steps from S to G:",
                    "What is the length of the shortest route from S to G?"])
        return mk_choice(rng, "path_len", p, L, hard=[L + 1, L + 2, L - 1, L + 4], gen=near_numbers(rng, L, 1, 5),
                         args={"op": "path_len"})

    def q_reach():
        p = T(rng, ["Can G be reached from S?", "Is there any path from S to G?", "Is the goal reachable?"])
        return mk_noul("reachable", p, reach, {"op": "reachable"})

    def q_wall():
        opts = [d for d, (dr, dc) in DIRS.items() if 0 <= s[0] + dr < H and 0 <= s[1] + dc < W]
        if not opts:
            return None
        walls = [d for d in opts if grid[s[0] + DIRS[d][0]][s[1] + DIRS[d][1]] == "#"]
        d = rng.choice(walls) if walls and rng.random() < 0.5 else rng.choice(opts)
        dr, dc = DIRS[d]
        rel = {"up": "directly above", "down": "directly below", "left": "immediately left of",
               "right": "immediately right of"}[d]
        p = T(rng, ["Is the cell {r} S a wall?", "Is there a wall {r} the start?", "Is the square {r} S blocked by a wall?"], r=rel)
        return mk_noul("wall_adjacent", p, grid[s[0] + dr][s[1] + dc] == "#", {"op": "wall", "dir": d})

    return state, [q_first, q_first, q_len, q_reach, q_wall]


PETS = ["cat", "dog", "fish", "parrot", "rabbit", "hamster", "turtle"]
COLORS = ["red", "blue", "green", "yellow", "white"]


def fam_logic_puzzle(rng):
    n = rng.randint(3, 4)
    people = rng.sample(FIRST, n)
    pets = rng.sample(PETS, n)
    cols = rng.sample(COLORS, n) if rng.random() < 0.6 else None
    sol_p = rng.sample(pets, n)
    sol_c = rng.sample(cols, n) if cols else None
    cands = [(pp, cc) for pp in itertools.permutations(pets) for cc in (itertools.permutations(cols) if cols else [None])]
    clues = []

    def make_clue():
        i = rng.randrange(n)
        P = people[i]
        kind = rng.choice(["pos", "neg", "neg", "neg"] + (["link", "clneg", "cpos"] if cols else []))
        if kind == "pos":
            return f"{P} owns the {sol_p[i]}.", lambda s, i=i, v=sol_p[i]: s[0][i] == v
        if kind == "neg":
            v = rng.choice([x for x in pets if x != sol_p[i]])
            return f"{P} does not own the {v}.", lambda s, i=i, v=v: s[0][i] != v
        if kind == "cpos":
            return f"{P} lives in the {sol_c[i]} house.", lambda s, i=i, v=sol_c[i]: s[1][i] == v
        if kind == "clneg":
            v = rng.choice([x for x in cols if x != sol_c[i]])
            return f"{P} does not live in the {v} house.", lambda s, i=i, v=v: s[1][i] != v
        # link: owner of pet lives in colour house
        return (f"The {sol_p[i]} owner lives in the {sol_c[i]} house.",
                lambda s, a=sol_p[i], b=sol_c[i]: s[1][s[0].index(a)] == b)

    for _ in range(40):
        if len(cands) == 1:
            break
        text, f = make_clue()
        new = [s for s in cands if f(s)]
        if len(new) < len(cands) and text not in clues:
            clues.append(text)
            cands = new
    if len(cands) != 1:
        return None
    rng.shuffle(clues)
    state = (T(rng, ["Logic puzzle.", "A small deduction puzzle.", "Clue sheet:"])
             + f" {', '.join(people[:-1])} and {people[-1]} each own exactly one different pet"
             + f" ({', '.join(pets)})"
             + (f" and each live in a different house ({', '.join(cols)})." if cols else ".")
             + "\nClues:\n" + "\n".join(f"{k + 1}. {c}" for k, c in enumerate(clues)))

    def q_who():
        i = rng.randrange(n)
        p = T(rng, ["Who owns the {x}?", "Which person has the {x}?", "The {x} belongs to:"], x=sol_p[i])
        return mk_choice(rng, "who_owns", p, people[i], hard=people, pool=FIRST, k=rng.randint(n, n + 3))

    def q_what():
        i = rng.randrange(n)
        p = T(rng, ["Which pet does {P} own?", "What is {P}'s pet?", "{P} owns the:"], P=people[i])
        return mk_choice(rng, "what_pet", p, sol_p[i], hard=pets, pool=PETS, k=rng.randint(n, len(PETS)))

    def q_house():
        if not cols:
            return None
        i = rng.randrange(n)
        p = T(rng, ["What colour is {P}'s house?", "Which house does {P} live in?", "{P} lives in the ___ house."], P=people[i])
        return mk_choice(rng, "house_of", p, sol_c[i], hard=cols, pool=COLORS)

    def q_noul():
        i = rng.randrange(n)
        v = sol_p[i] if rng.random() < 0.5 else rng.choice(pets)
        p = T(rng, ["Does {P} own the {v}?", "Is the {v} {P}'s pet?", "Is {P} the owner of the {v}?"], P=people[i], v=v)
        return mk_noul("owns", p, v == sol_p[i], {"person": people[i], "pet": v})

    return state, [q_who, q_what, q_house, q_noul]


MEETINGS = ["Standup", "1:1 with manager", "Design sync", "Vendor call", "Sprint planning", "Lunch with client",
            "Interview", "Budget review", "All-hands", "Code review", "Customer demo", "Retro", "Training"]


def hm(m):
    return f"{m // 60:02d}:{m % 60:02d}"


def fam_schedule_conflict(rng):
    n = rng.randint(3, 8)
    names = rng.sample(MEETINGS, n)
    mt = []
    for nm in names:
        st = rng.randrange(8 * 60, 17 * 60, 15)
        mt.append(dict(name=nm, s=st, e=st + rng.choice([15, 30, 45, 60, 90])))
    mt.sort(key=lambda m: m["s"])
    person = rng.choice(FIRST)
    lth = sorted(rng.sample([60, 90, 120, 150, 180, 240, 300, 360], 3))
    load_rule = (f"Day load (sum of all meeting durations): light if under {lth[0]} min, moderate if under "
                 f"{lth[1]} min, heavy if under {lth[2]} min, otherwise overloaded.")
    if rng.random() < 0.5:
        state = (f"{person}'s calendar for {rand_date(rng, 2025, 2026).isoformat()}:\n"
                 + "\n".join(f"{hm(m['s'])}-{hm(m['e'])}  {m['name']}" for m in mt)
                 + "\nNote: meetings that merely touch (one ends when the next starts) do not conflict.\n"
                 + load_rule)
    else:
        state = {"owner": person, "timezone": "local",
                 "policy": "Back-to-back meetings (end == start) are not a conflict.", "load_rule": load_rule,
                 "events": [{"title": m["name"], "start": hm(m["s"]), "end": hm(m["e"])} for m in mt]}
    ov = lambda a, b: a["s"] < b["e"] and b["s"] < a["e"]

    def q_pair():
        ovp = [(x, y) for x, y in itertools.combinations(mt, 2) if ov(x, y)]
        a, b = rng.choice(ovp) if ovp and rng.random() < 0.5 else rng.sample(mt, 2)
        if rng.random() < 0.5:
            a, b = b, a
        p = T(rng, ["Do '{a}' and '{b}' overlap?", "Is there a conflict between {a} and {b}?",
                    "Would {person} be double-booked by {a} and {b}?"], a=a["name"], b=b["name"], person=person)
        return mk_noul("overlap", p, ov(a, b), {"a": a["name"], "b": b["name"]})

    def q_free():
        st = rng.randrange(8 * 60, 18 * 60, 15)
        dur = rng.choice([15, 30, 60])
        slot = dict(s=st, e=st + dur)
        p = T(rng, ["Is {P} free from {s} to {e}?", "Can a meeting be booked {s}-{e} without a conflict?",
                    "Is the slot {s}-{e} open?"], P=person, s=hm(slot["s"]), e=hm(slot["e"]))
        return mk_noul("free_slot", p, not any(ov(slot, m) for m in mt), {"s": slot["s"], "e": slot["e"]})

    def q_which():
        a = rng.choice(mt)
        hits = [m["name"] for m in mt if m is not a and ov(a, m)]
        if len(hits) != 1:
            return None
        p = T(rng, ["Which meeting conflicts with '{a}'?", "'{a}' overlaps with which other meeting?",
                    "Find the meeting that clashes with {a}."], a=a["name"])
        return mk_choice(rng, "which_conflict", p, hits[0], hard=[m["name"] for m in mt if m is not a])

    def q_count():
        c = sum(1 for a, b in itertools.combinations(mt, 2) if ov(a, b))
        p = T(rng, ["How many pairs of meetings overlap?", "Count the conflicting meeting pairs.",
                    "Number of double-booked pairs:"])
        return mk_choice(rng, "conflict_pairs", p, c, gen=near_numbers(rng, c, 0, 4))

    def q_last():
        e = max(m["e"] for m in mt)
        ends = [m for m in mt if m["e"] == e]
        if len(ends) > 1:
            return None
        p = T(rng, ["Which meeting ends last?", "What is the final meeting to finish?", "Latest-ending event:"])
        return mk_choice(rng, "ends_last", p, ends[0]["name"], hard=names, pool=MEETINGS)

    def q_load():
        tot = sum(m["e"] - m["s"] for m in mt)
        lvl = next((j for j, t in enumerate(lth) if tot < t), len(lth))
        p = T(rng, ["How loaded is {P}'s day?", "Rate the day's meeting load using the rule.",
                    "Day load classification for {P}:"], P=person)
        return mk_score("day_load", p, ["light", "moderate", "heavy", "overloaded"], lvl, {"minutes": tot})

    return state, [q_pair, q_pair, q_free, q_which, q_count, q_last, q_load]


GRAPH_NAMES = ["Ash", "Birch", "Cedar", "Dale", "Elm", "Fern", "Glen", "Heath", "Ivy", "Juniper", "Kiln",
               "Larch", "Moss", "Nettle"]


def graph_bfs(adj, s):
    dist = {s: 0}
    dq = deque([s])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


def fam_graph_reachability(rng):
    n = rng.randint(5, 12)
    nodes = rng.sample([chr(65 + i) for i in range(16)] if rng.random() < 0.5 else GRAPH_NAMES, n)
    m = rng.randint(n - 1, int(n * 1.8))
    edges = set()
    while len(edges) < m:
        a, b = rng.sample(nodes, 2)
        edges.add((a, b))
    adj = {u: sorted(v for (x, v) in edges if x == u) for u in nodes}
    style = rng.choice(["arrow", "colon", "prose"])
    lines = [T(rng, ["One-way links between stations:", "Directed dependency graph (edges are one-way):",
                     "Network routes - each link can only be travelled in the direction given:"])]
    for u in nodes:
        if style == "arrow":
            lines.append(f"{u} -> " + (", ".join(adj[u]) if adj[u] else "(none)"))
        elif style == "colon":
            lines.append(f"{u}: " + (" ".join(adj[u]) if adj[u] else "-"))
        else:
            lines.append(f"From {u} you can go to " + ", ".join(adj[u]) + "." if adj[u] else f"{u} has no outgoing links.")
    state = "\n".join(lines)

    def q_reach():
        s, t = rng.sample(nodes, 2)
        p = T(rng, ["Can {t} be reached from {s}?", "Is there a directed path from {s} to {t}?",
                    "Starting at {s}, is it possible to get to {t}?"], s=s, t=t)
        return mk_noul("reachable", p, t in graph_bfs(adj, s), {"s": s, "t": t})

    def q_which():
        s = rng.choice(nodes)
        d = graph_bfs(adj, s)
        yes = [x for x in d if x != s]
        no = [x for x in nodes if x not in d]
        if not yes or not no:
            return None
        p = T(rng, ["Which of these can be reached from {s}?", "Starting from {s}, which node is reachable?",
                    "Pick the node reachable from {s}."], s=s)
        return mk_choice(rng, "which_reachable", p, rng.choice(yes), hard=no, args={"s": s})

    def q_hops():
        s = rng.choice(nodes)
        d = graph_bfs(adj, s)
        ts = [x for x in d if x != s]
        if not ts:
            return None
        t = rng.choice(ts)
        p = T(rng, ["What is the fewest number of links needed to go from {s} to {t}?",
                    "Shortest path length (in edges) from {s} to {t}:", "How many hops from {s} to {t} at minimum?"], s=s, t=t)
        return mk_choice(rng, "hops", p, d[t], hard=[d[t] + 1, d[t] + 2, d[t] - 1], gen=near_numbers(rng, d[t], 1, 4),
                         args={"s": s, "t": t})

    def q_outdeg():
        best = max(nodes, key=lambda u: len(adj[u]))
        if sum(1 for u in nodes if len(adj[u]) == len(adj[best])) > 1:
            return None
        p = T(rng, ["Which node has the most outgoing links?", "Which node has the highest out-degree?",
                    "From which node do the most edges leave?"])
        return mk_choice(rng, "max_outdeg", p, best, hard=nodes)

    return state, [q_reach, q_reach, q_which, q_hops, q_outdeg]


CHAT_TOPICS = [("offsite venue", ["Harbor Hall", "Pine Lodge", "City Loft", "Lake House", "Old Mill"]),
               ("lunch spot", ["the ramen bar", "the taco stand", "the salad place", "the pizzeria", "the dim sum hall"]),
               ("release day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])]
CHAT_VOTE = ["I vote for {o}.", "My pick is {o}.", "Put me down for {o}.", "{o} works best for me.",
             "Actually, change my vote to {o}.", "I'd go with {o}."]
CHAT_FILLER = ["Sounds good to me.", "Can we decide by noon?", "Is everyone here?", "Sorry, was in a call.",
               "Reminder: one vote each.", "lol", "Let's keep this quick.", "Thanks all!"]


def fam_chat_transcript(rng):
    topic, allopts = rng.choice(CHAT_TOPICS)
    opts = rng.sample(allopts, rng.randint(2, 4))
    people = rng.sample(FIRST, rng.randint(3, 5))
    t = rng.randint(8 * 60, 17 * 60)
    msgs = []
    for _ in range(rng.randint(6, 16)):
        t += rng.randint(0, 4)
        who = rng.choice(people)
        if rng.random() < 0.6:
            o = rng.choice(opts)
            text = rng.choice(CHAT_VOTE).format(o=o)
            msgs.append(dict(t=hm(t), who=who, text=text[0].upper() + text[1:], vote=o))
        else:
            msgs.append(dict(t=hm(t), who=who, text=rng.choice(CHAT_FILLER), vote=None))
    last = {}
    for m in msgs:
        if m["vote"]:
            last[m["who"]] = m["vote"]
    tally = Counter(last.values())
    state = (f"#team-chat - choosing the {topic}. Rule: each person's most recent vote is the one that counts; "
             f"options are {', '.join(opts)}.\n\n" + "\n".join(f"[{m['t']}] {m['who']}: {m['text']}" for m in msgs))

    def q_winner():
        if not tally:
            return None
        top = tally.most_common(2)
        if len(top) > 1 and top[0][1] == top[1][1]:
            return None
        p = T(rng, ["Which {t} wins the vote?", "Counting only each person's latest vote, which {t} has the most votes?",
                    "Result of the {t} vote:"], t=topic)
        return mk_choice(rng, "vote_winner", p, top[0][0], hard=opts, pool=allopts)

    def q_last_vote():
        if not last:
            return None
        who = rng.choice(list(last))
        p = T(rng, ["What is {w}'s final vote?", "Which option did {w} vote for most recently?",
                    "{w}'s vote that counts:"], w=who)
        return mk_choice(rng, "last_vote", p, last[who], hard=opts, pool=allopts, args={"who": who})

    def q_first():
        p = T(rng, ["Who sent the first message?", "Who spoke first in the chat?", "The conversation was started by:"])
        return mk_choice(rng, "first_speaker", p, msgs[0]["who"], hard=people, pool=FIRST)

    def q_count():
        who = rng.choice(people)
        c = sum(1 for m in msgs if m["who"] == who)
        p = T(rng, ["How many messages did {w} send?", "Count {w}'s messages.", "Number of lines written by {w}:"], w=who)
        return mk_choice(rng, "msg_count", p, c, gen=near_numbers(rng, c, 0, 4), args={"who": who})

    def q_ever():
        who = rng.choice(people)
        o = rng.choice(opts)
        p = T(rng, ["Did {w} ever vote for {o}?", "At any point, did {w} back {o}?", "Has {w} voted {o} at least once?"],
              w=who, o=o)
        return mk_noul("ever_voted", p, any(m["who"] == who and m["vote"] == o for m in msgs), {"who": who, "opt": o})

    return state, [q_winner, q_winner, q_last_vote, q_first, q_count, q_ever]


def c4_winner(b):
    R, C = 6, 7
    for r in range(R):
        for c in range(C):
            p = b[r][c]
            if p == ".":
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if all(0 <= r + k * dr < R and 0 <= c + k * dc < C and b[r + k * dr][c + k * dc] == p for k in range(4)):
                    return p
    return None


def c4_drop(b, c, p):
    for r in range(5, -1, -1):
        if b[r][c] == ".":
            nb = [row[:] for row in b]
            nb[r][c] = p
            return nb, 6 - r  # row counted from the bottom, 1..6
    return None, None


def fam_connect_four(rng):
    b = [["."] * 7 for _ in range(6)]
    player = "R"
    for _ in range(rng.randint(2, 36)):
        cols = [c for c in range(7) if b[0][c] == "."]
        if not cols:
            break
        b, _ = c4_drop(b, rng.choice(cols), player)
        player = "Y" if player == "R" else "R"
        if c4_winner(b):
            break
    win = c4_winner(b)
    nR = sum(row.count("R") for row in b)
    nY = sum(row.count("Y") for row in b)
    to_move = "R" if nR == nY else "Y"
    pname = {"R": "Red", "Y": "Yellow"}
    state = (T(rng, ["Connect Four board.", "Four-in-a-row game in progress.", "Board state from the tournament app."])
             + " Red (R) moved first; Yellow (Y) second. Pieces fall to the lowest empty cell of a column. "
             "Columns are numbered 1-7 left to right; the top line is the top of the board.\n\n"
             + " ".join(str(i) for i in range(1, 8)) + "\n" + "\n".join(" ".join(r) for r in b) + "\n")

    def q_winner():
        lab = {"R": "Red", "Y": "Yellow", None: "no one"}
        p = T(rng, ["Has anyone connected four? If so, who?", "Who has won?", "Winner so far:"])
        return mk_choice(rng, "winner", p, lab[win], hard=list(lab.values()), k=3)

    def q_land():
        if win:
            return None
        cols = [c for c in range(7) if b[0][c] == "."]
        c = rng.choice(cols)
        _, row = c4_drop(b, c, to_move)
        p = T(rng, ["If a piece is dropped in column {c}, in which row (counted from the bottom, 1-6) does it land?",
                    "Dropping into column {c}: landing row, counting from the bottom?",
                    "Which row from the bottom would a disc in column {c} occupy?"], c=c + 1)
        return mk_choice(rng, "landing_row", p, row, hard=[str(i) for i in range(1, 7)], k=rng.randint(2, 6),
                         args={"col": c + 1})

    def q_full():
        fullc = [c for c in range(7) if b[0][c] != "."]
        if not fullc and rng.random() < 0.6:
            return None
        c = rng.choice(fullc) if fullc and rng.random() < 0.5 else rng.randrange(7)
        p = T(rng, ["Is column {c} full?", "Can no more pieces go into column {c}?", "Is column {c} completely filled?"], c=c + 1)
        return mk_noul("column_full", p, b[0][c] != ".", {"col": c + 1})

    def q_winmove():
        if win:
            return None
        cols = [c for c in range(7) if b[0][c] == "."]
        wins = [c for c in cols if c4_winner(c4_drop(b, c, to_move)[0]) == to_move]
        if len(wins) != 1:
            return None
        p = T(rng, ["{p} to move. Which column wins immediately?", "Which column gives {p} four in a row right now?",
                    "Winning drop for {p}:"], p=pname[to_move])
        return mk_choice(rng, "winning_column", p, f"column {wins[0] + 1}", hard=[f"column {c + 1}" for c in cols],
                         args={"player": to_move})

    def q_turn():
        if win:
            return None
        p = T(rng, ["Whose turn is it?", "Which colour moves next?", "Next to play:"])
        return mk_choice(rng, "turn", p, pname[to_move], hard=["Red", "Yellow"], k=2)

    return state, [q_winner, q_land, q_land, q_full, q_winmove, q_winmove, q_turn]


def fam_sequence_simulation(rng):
    if rng.random() < 0.55:
        v = rng.randint(0, 20)
        start = v
        ops, hist = [], []
        for _ in range(rng.randint(4, 14)):
            k = rng.choice(["add", "sub", "mul", "set"]) if rng.random() < 0.95 else "neg"
            if k == "add":
                a = rng.randint(1, 15); v += a; ops.append(f"add {a}")
            elif k == "sub":
                a = rng.randint(1, 15); v -= a; ops.append(f"subtract {a}")
            elif k == "mul":
                a = rng.choice([2, 3]) if abs(v) < 200 else 1; v *= a; ops.append(f"multiply by {a}")
            elif k == "set":
                if rng.random() < 0.7:
                    a = rng.randint(1, 15); v += a; ops.append(f"add {a}")
                else:
                    v = rng.randint(0, 30); ops.append(f"set to {v}")
            else:
                v = -v; ops.append("negate")
            hist.append(v)
        state = (T(rng, ["A register starts at {s}. Apply these operations in order:",
                         "Counter program. Initial value: {s}. Steps:", "Start with x = {s}, then run:"], s=start)
                 + "\n" + "\n".join(f"{i + 1}. {o}" for i, o in enumerate(ops)))

        def q_final():
            p = T(rng, ["What is the final value?", "Value after the last step:", "Where does the counter end up?"])
            return mk_choice(rng, "final_value", p, v, hard=[hist[-2] if len(hist) > 1 else start, v + 1, v - 1, -v],
                             gen=near_numbers(rng, v, -10 ** 6, 6), big_ok=True, args={"op": "final"})

        def q_step():
            i = rng.randrange(len(ops))
            p = T(rng, ["What is the value right after step {i}?", "Value after executing step {i}:",
                        "After step {i}, the register holds:"], i=i + 1)
            return mk_choice(rng, "value_after_step", p, hist[i], hard=[h for h in hist if h != hist[i]],
                             gen=near_numbers(rng, hist[i], -10 ** 6, 6), args={"op": "step", "step": i + 1})

        def q_even():
            if rng.random() < 0.5:
                p = T(rng, ["Is the final value even?", "Does the program end on an even number?", "Final value even?"])
                return mk_noul("final_even", p, v % 2 == 0, {"op": "even"})
            th = v + rng.choice([-9, -2, -1, 1, 4, 10])
            p = T(rng, ["Is the final value greater than {t}?", "Does the counter finish above {t}?",
                        "Final value > {t}?"], t=th)
            return mk_noul("final_gt", p, v > th, {"op": "gt", "threshold": th})

        return state, [q_final, q_final, q_step, q_even]

    # stack variant
    stack, ops, pushed = [], [], []
    for _ in range(rng.randint(5, 16)):
        if stack and rng.random() < 0.35:
            stack.pop(); ops.append("pop")
        else:
            x = rng.choice([w for w in WORDS[:120] if w not in pushed] or WORDS)
            pushed.append(x); stack.append(x); ops.append(f"push {x}")
    state = (T(rng, ["A stack starts empty. Operations (pop removes the most recently pushed item):",
                     "LIFO stack trace, starting from an empty stack:", "Stack program:"])
             + "\n" + "\n".join(f"{i + 1}. {o}" for i, o in enumerate(ops)))

    def q_top():
        if not stack:
            return None
        p = T(rng, ["Which item is on top of the stack at the end?", "What would the next pop return?",
                    "Top of the stack after all operations:"])
        return mk_choice(rng, "stack_top", p, stack[-1], hard=pushed, pool=WORDS, big_ok=True, args={"op": "top"})

    def q_size():
        c = len(stack)
        p = T(rng, ["How many items are on the stack at the end?", "Final stack size:", "How many elements remain?"])
        return mk_choice(rng, "stack_size", p, c, hard=[len(pushed)], gen=near_numbers(rng, c, 0, 4), args={"op": "size"})

    def q_in():
        x = rng.choice(pushed)
        p = T(rng, ["Is '{x}' still on the stack at the end?", "Does the final stack contain {x}?",
                    "After all operations, is {x} in the stack?"], x=x)
        return mk_noul("stack_contains", p, x in stack, {"op": "contains", "item": x})

    return state, [q_top, q_size, q_in]


FAMILIES = {
    "record_lookup": fam_record_lookup, "table_arith": fam_table_arith, "compare_sort": fam_compare_sort,
    "log_counting": fam_log_counting, "ticket_routing": fam_ticket_routing, "date_reasoning": fam_date_reasoning,
    "unit_conversion": fam_unit_conversion, "set_membership": fam_set_membership, "tictactoe": fam_tictactoe,
    "inventory_levels": fam_inventory_levels, "string_props": fam_string_props,
    "severity_scoring": fam_severity_scoring, "price_calc": fam_price_calc,
    "grid_navigation": fam_grid_navigation, "logic_puzzle": fam_logic_puzzle,
    "schedule_conflict": fam_schedule_conflict, "graph_reachability": fam_graph_reachability,
    "chat_transcript": fam_chat_transcript, "connect_four": fam_connect_four,
    "sequence_simulation": fam_sequence_simulation,
}
assert set(FAMILIES) == set(FAMILY_SPLIT)
TRAIN_FAMILIES = [f for f in FAMILIES if FAMILY_SPLIT[f] == "train"]
OOD_FAMILIES = [f for f in FAMILIES if FAMILY_SPLIT[f] == "ood"]


# ---------------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------------
def state_key(state) -> str:
    s = state if isinstance(state, str) else json.dumps(state, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()


CTX_SYSTEMS = ["ops portal", "data warehouse", "support desk", "planning tool", "field app", "audit service"]
CTX_HEADERS = ["Source: {sys} export, retrieved {d}.", "Document {id}. Generated {d} by the {sys}.",
               "Context supplied by {name} on {d}.", "Internal snapshot from the {sys} taken {d}."]
CTX_NOTES = ["This snapshot was generated automatically and may be reviewed later.",
             "Please keep this information internal.", "Formatting may differ slightly from the source system.",
             "Questions about this data can go to the operations team.", "No manual edits were made after export.",
             "Earlier versions of this document are archived separately.", "Values were checked by the nightly job."]


def decorate(rng, state):
    """Irrelevant context (header / footer notes) so states vary in length and need locating the facts."""
    ctx = dict(sys=rng.choice(CTX_SYSTEMS), d=rand_date(rng, 2024, 2026).isoformat(),
               id=f"DOC-{rng.randint(1000, 99999)}", name=rng.choice(FIRST))
    if isinstance(state, str):
        if rng.random() < 0.6:
            state = rng.choice(CTX_HEADERS).format(**ctx) + "\n\n" + state
        if rng.random() < 0.4:
            state = state.rstrip("\n") + "\n\n" + " ".join(rng.sample(CTX_NOTES, rng.randint(1, 3)))
    elif isinstance(state, dict) and rng.random() < 0.5:
        state = {"meta": {"source": ctx["sys"], "retrieved": ctx["d"], "doc_id": ctx["id"]}, **state}
    return state


def generate_record(family: str, rng: random.Random, rid: str):
    out = FAMILIES[family](rng)
    if out is None:
        return None
    state, makers = out
    state = decorate(rng, state)
    nq = rng.randint(1, MAX_Q_PER_STATE)
    qs, keys = [], set()
    for _ in range(nq * 8):
        if len(qs) >= nq:
            break
        q = rng.choice(makers)()
        if q is None:
            continue
        # drop exact and semantic repeats (same sub-type, same arguments, same answer)
        key = (q["kind"], json.dumps(q["args"], sort_keys=True), q["options"][q["label"]])
        if q["prompt"] in keys or key in keys:
            continue
        keys.update([q["prompt"], key])
        qs.append(q)
    if not qs:
        return None
    return {"id": rid, "family": family, "state": state, "questions": qs}


def generate_split(split: str, n: int, seed: int, families, seen: set | None = None):
    """n records, round-robin over `families`; per-record RNG seeded by (seed, split, family, i, attempt)."""
    seen = set() if seen is None else seen
    recs = []
    for i in range(n):
        fam = families[i % len(families)]
        rec = None
        for attempt in range(50):
            rng = random.Random(f"{seed}|{split}|{fam}|{i}|{attempt}")
            rec = generate_record(fam, rng, f"{split}-{fam}-{i:06d}")
            if rec is not None and state_key(rec["state"]) not in seen:
                break
        if rec is None:
            continue
        seen.add(state_key(rec["state"]))
        recs.append(rec)
    random.Random(f"{seed}|{split}|order").shuffle(recs)
    return recs


def state_chars(state) -> int:
    return len(state) if isinstance(state, str) else len(json.dumps(state))


K_BINS = [(2, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 59), (60, 127), (128, 255)]


def split_stats(recs):
    fam = Counter(r["family"] for r in recs)
    qtype = Counter(q["type"] for r in recs for q in r["questions"])
    ks = [len(q["options"]) for r in recs for q in r["questions"] if q["type"] == "choice"]
    hist = {f"{a}-{b}": sum(1 for k in ks if a <= k <= b) for a, b in K_BINS}
    lens = sorted(state_chars(r["state"]) for r in recs)
    pct = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))] if lens else 0
    nq = sum(len(r["questions"]) for r in recs)
    noul_yes = [q["label"] == 0 for r in recs for q in r["questions"] if q["type"] == "noul"]
    return {
        "records": len(recs), "questions": nq, "records_per_family": dict(sorted(fam.items())),
        "questions_per_type": dict(qtype), "choice_option_count_hist": hist,
        "choice_big_option_frac_of_all_questions": round(sum(k >= 60 for k in ks) / max(1, nq), 4),
        "noul_yes_frac": round(sum(noul_yes) / max(1, len(noul_yes)), 3),
        "state_chars": {"min": lens[0] if lens else 0, "p05": pct(.05), "p50": pct(.5), "p95": pct(.95),
                        "max": lens[-1] if lens else 0},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000, help="size of test_id and of test_ood")
    ap.add_argument("--n-calib", "--calib", dest="n_calib", type=int, default=None,
                    help="size of calib split (default: same as --n-val)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    n_calib = a.n_val if a.n_calib is None else a.n_calib
    os.makedirs(a.out, exist_ok=True)
    seen: set = set()
    plan = [("train", a.n_train, TRAIN_FAMILIES), ("val", a.n_val, TRAIN_FAMILIES),
            ("calib", n_calib, TRAIN_FAMILIES), ("test_id", a.n_test, TRAIN_FAMILIES),
            ("test_ood", a.n_test, OOD_FAMILIES)]
    stats = {"seed": a.seed, "family_split": FAMILY_SPLIT, "splits": {}}
    for split, n, fams in plan:
        recs = generate_split(split, n, a.seed, fams, seen)
        path = os.path.join(a.out, f"{split}.jsonl")
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats["splits"][split] = split_stats(recs)
        print(f"{split:9s} {len(recs):6d} records -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)", file=sys.stderr)
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
