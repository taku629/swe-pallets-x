#!/usr/bin/env python3
"""Per-instance dump of fix-localization eval (same scoring as eval_localize).

Writes one JSON row per instance: {instance_id, repo, flat_hits, graph_hits}
where *_hits = smallest k in which a patched file appears (0 = miss at k=10).
"""
import json
import math
import os
import sys
import tarfile
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_localize import tokens, node_file, patched_files  # noqa: E402


def main(tasks_dir, test_path, out_path, ks=(1, 3, 5, 10)):
    tasks = {json.loads(l)["instance_id"]: json.loads(l)
             for l in open(os.path.join(tasks_dir, "tasks.jsonl"))}
    recs = [json.loads(l) for l in open(test_path)]
    fout = open(out_path, "w")
    for r in recs:
        t = tasks[r["instance_id"]]
        gpath = os.path.join(tasks_dir, "graphs",
                             r["instance_id"] + ".json")
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
        row = {"instance_id": r["instance_id"], "repo": t["repo"],
               "n_targets": len(targets)}
        for name, sc in (("flat", fscore), ("graph", gscore)):
            ranked = [f for f, _ in sorted(sc.items(), key=lambda x: -x[1])]
            first = min((ranked.index(tg) + 1
                         for tg in targets if tg in sc), default=0)
            row[name + "_rank"] = first
        fout.write(json.dumps(row) + "\n")
        fout.flush()
    fout.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
