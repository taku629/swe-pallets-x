#!/usr/bin/env python3
"""Fix-localization eval on SWE-Pallets-X graph assets.

Per test instance: BM25-rank graph nodes against the problem statement,
map nodes->files, and measure recall@k of the patched files.

flat : file score = max node score
graph: file score = max(node score + 0.5 * max neighbor node score)
"""
import json
import math
import os
import re
import sys
import tarfile
from collections import Counter, defaultdict

TOK = re.compile(r"[A-Za-z_]{2,}")


def tokens(s):
    return TOK.findall(s.lower())


def node_file(node_id, fileset):
    parts = node_id.split(".")
    for k in range(min(6, len(parts) - 1), 0, -1):
        for cand in ("/".join(parts[:k]) + ".py",
                     "/".join(parts[:k]) + "/__init__.py"):
            if cand in fileset:
                return cand
    return None


def patched_files(patch):
    return set(re.findall(r"--- a/(\S+)", patch))


def main(tasks_dir, test_path, ks=(1, 3, 5, 10)):
    tasks = {json.loads(l)["instance_id"]: json.loads(l)
             for l in open(os.path.join(tasks_dir, "tasks.jsonl"))}
    recs = [json.loads(l) for l in open(test_path)]
    hits = {"flat": Counter(), "graph": Counter()}
    n = 0
    for r in recs:
        t = tasks[r["instance_id"]]
        gpath = os.path.join(tasks_dir, "graphs", r["instance_id"] + ".json")
        spath = os.path.join(tasks_dir, "snapshots",
                             r["instance_id"] + ".tgz")
        if not (os.path.exists(gpath) and os.path.exists(spath)):
            continue
        g = json.load(open(gpath))
        with tarfile.open(spath) as tf:
            fileset = {m.name.lstrip("./") for m in tf.getmembers()}
        targets = patched_files(t["patch"]) & fileset
        if not targets:
            continue
        n += 1
        docs = [tokens(nd["id"]) + tokens(nd.get("text") or "")[:400]
                for nd in g["nodes"]]
        df = Counter()
        for d in docs:
            df.update(set(d))
        N = len(docs)
        avg = sum(map(len, docs)) / max(N, 1)
        q = tokens(r["user"].split("## Files")[0])
        scores = []
        for d in docs:
            c = Counter(d)
            s = 0.0
            for w in set(q):
                if w not in c:
                    continue
                idf = math.log(1 + (N - df[w] + .5) / (df[w] + .5))
                tf = c[w] * 2.2 / (c[w] + 1.2 * (.25 + .75 * len(d) / avg))
                s += idf * tf
            scores.append(s)
        id2i = {nd["id"]: i for i, nd in enumerate(g["nodes"])}
        adj = defaultdict(list)
        for e in g.get("edges", []):
            a, b = id2i.get(e.get("source")), id2i.get(e.get("target"))
            if a is not None and b is not None:
                adj[a].append(b)
                adj[b].append(a)
        fscore = defaultdict(float)
        gscore = defaultdict(float)
        for i, nd in enumerate(g["nodes"]):
            f = node_file(nd["id"], fileset)
            if not f:
                continue
            fscore[f] = max(fscore[f], scores[i])
            nb = max((scores[j] for j in adj.get(i, []) if j < N),
                     default=0.0)
            gscore[f] = max(gscore[f], scores[i] + .5 * nb)
        for k in ks:
            top_f = {f for f, _ in sorted(fscore.items(),
                                          key=lambda x: -x[1])[:k]}
            top_g = {f for f, _ in sorted(gscore.items(),
                                          key=lambda x: -x[1])[:k]}
            hits["flat"][k] += bool(top_f & targets)
            hits["graph"][k] += bool(top_g & targets)
    for k in ks:
        print(f"recall@{k}: flat {hits['flat'][k]}/{n} "
              f"({hits['flat'][k]/max(n,1):.1%}) | "
              f"graph {hits['graph'][k]}/{n} "
              f"({hits['graph'][k]/max(n,1):.1%})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
