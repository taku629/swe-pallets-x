#!/usr/bin/env python3
"""Embedding-quality probe: do 256-dim node embeddings cluster by file?

For each sampled instance: for each node, take top-8 cosine neighbors and
measure the fraction living in the same source file, vs the chance level
(share of nodes in that file). Reports mean lift over instances.
"""
import json
import os
import re
import sys
import tarfile
from collections import Counter

import numpy as np


def node_file(node_id, fileset):
    parts = node_id.split(".")
    for k in range(min(6, len(parts) - 1), 0, -1):
        for cand in ("/".join(parts[:k]) + ".py",
                     "/".join(parts[:k]) + "/__init__.py"):
            if cand in fileset:
                return cand
    return None


def main(tasks_dir, sample=120, knn=8, max_nodes=3000, seed=0):
    rng = np.random.RandomState(seed)
    rows = [json.loads(l) for l in
            open(os.path.join(tasks_dir, "tasks.jsonl"))]
    rng.shuffle(rows)
    rows = rows[:sample]
    lifts, n_nodes = [], 0
    done = skipped = 0
    for r in rows:
        ep = os.path.join(tasks_dir, "embeddings", r["instance_id"] + ".npz")
        sp = os.path.join(tasks_dir, "snapshots", r["instance_id"] + ".tgz")
        if not (os.path.exists(ep) and os.path.exists(sp)):
            skipped += 1
            continue
        z = np.load(ep)
        ids = list(z.keys())
        if len(ids) < 30:
            skipped += 1
            continue
        with tarfile.open(sp) as tf:
            fileset = {m.name.lstrip("./") for m in tf.getmembers()}
        nf = [node_file(i, fileset) for i in ids]
        keep = [i for i, f in enumerate(nf) if f]
        if len(keep) < 30:
            skipped += 1
            continue
        if len(keep) > max_nodes:
            keep = list(rng.choice(keep, max_nodes, replace=False))
        E = np.stack([z[ids[i]] for i in keep]).astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        sim = E @ E.T
        np.fill_diagonal(sim, -2)
        top = np.argpartition(-sim, knn, axis=1)[:, :knn]
        files = [nf[i] for i in keep]
        share = Counter(files)
        N = len(keep)
        hits = chance = 0.0
        for i in range(N):
            same = sum(files[j] == files[i] for j in top[i]) / knn
            hits += same
            chance += (share[files[i]] - 1) / (N - 1)
        lifts.append(hits / N / (chance / N + 1e-9))
        n_nodes += N
        done += 1
        if done % 20 == 0:
            print(f"[{done}] lift so far {np.mean(lifts):.2f}x "
                  f"(n={done} inst, {n_nodes} nodes)", file=sys.stderr)
    lifts = np.array(lifts)
    print(f"instances={done} nodes={n_nodes} knn={knn}")
    print(f"same-file-NN share: mean lift vs chance "
          f"{lifts.mean():.2f}x (median {np.median(lifts):.2f}x, "
          f"p10 {np.percentile(lifts,10):.2f}x, "
          f"p90 {np.percentile(lifts,90):.2f}x)")
    print(f"instances with lift>1: {(lifts>1).mean():.0%}")


if __name__ == "__main__":
    main(sys.argv[1])
