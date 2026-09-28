#!/usr/bin/env python3
"""3-way arm comparison on the 181-task pallets slice: nograph vs graph vs oracle."""
import json, statistics as st
from math import comb

def load(f):
    try:
        return {json.loads(l)["instance_id"]: json.loads(l)
                for l in open(f"data/eval_run/{f}.jsonl")}
    except FileNotFoundError:
        return {}

n = load("qwen_nograph_full")
g = {**load("qwen_graph_full"), **load("qwen_graph_A"),
     **load("qwen_graph_B"), **load("qwen_graph_rest")}
o = load("qwen_oracle_181")

def mcnemar(a, b, key):
    d1 = sum(1 for i in a if key(a[i]) and i in b and not key(b[i]))
    d2 = sum(1 for i in a if i in b and not key(a[i]) and key(b[i]))
    m = d1 + d2
    p = (min(1.0, 2 * sum(comb(m, i) for i in range(min(d1, d2) + 1)) / 2 ** m)
         if m else 1.0)
    return d1, d2, p

passed = lambda r: str(r.get("passed")) == "True"
print(f"{'arm':10s} {'n':>4s} {'patch':>6s} {'pass':>5s} {'steps':>6s}")
for name, d in [("nograph", n), ("graph", g), ("oracle", o)]:
    if not d: continue
    hp = sum(1 for r in d.values() if r.get("has_patch"))
    ps = sum(1 for r in d.values() if passed(r))
    print(f"{name:10s} {len(d):4d} {hp:4d} {hp/len(d):5.1%} "
          f"{ps:3d} {ps/len(d):5.1%} pass | steps {st.mean(r['steps'] for r in d.values()):5.1f}")
print()
common = set(n) & set(g) & set(o) if o else set(n) & set(g)
if o:
    d1, d2, p = mcnemar({i: n[i] for i in common}, {i: o[i] for i in common},
                        lambda r: r.get("has_patch"))
    print(f"patch McNemar nograph-vs-oracle: {d1}/{d2} p={p:.3f}")
    d1, d2, p = mcnemar({i: n[i] for i in common}, {i: o[i] for i in common},
                        passed)
    print(f"pass  McNemar nograph-vs-oracle: {d1}/{d2} p={p:.3f}")
    print("oracle-only solves:",
          [i for i in common if passed(o[i]) and not passed(n[i])])
