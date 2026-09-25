#!/usr/bin/env python3
"""Dataset integrity check for SWE-Pallets-X.

Verifies every task in tasks.jsonl has a snapshot, graph, and embedding;
checks schema fields, duplicate ids, patch/test_patch non-emptiness, and
snapshot unpackability. Exits non-zero on any failure.
"""
import json
import os
import sys
import tarfile

REQUIRED = ["instance_id", "repo", "base_commit", "fix_commit",
            "problem_statement", "patch", "test_patch", "created_at"]


def main(task_dir):
    recs = [json.loads(l) for l in open(os.path.join(task_dir, "tasks.jsonl"))]
    errs, warns = [], []
    ids = [r["instance_id"] for r in recs]
    if len(ids) != len(set(ids)):
        from collections import Counter
        errs.append(f"duplicate ids: {[k for k, v in Counter(ids).items() if v > 1]}")

    repos = {}
    for r in recs:
        iid = r["instance_id"]
        repos[r["repo"]] = repos.get(r["repo"], 0) + 1
        for f in REQUIRED:
            if not r.get(f):
                errs.append(f"{iid}: missing/empty field {f}")
        snap = os.path.join(task_dir, "snapshots", iid + ".tgz")
        gj = os.path.join(task_dir, "graphs", iid + ".json")
        ej = os.path.join(task_dir, "embeddings", iid + ".npz")
        if not os.path.exists(snap):
            errs.append(f"{iid}: no snapshot")
        else:
            try:
                with tarfile.open(snap) as t:
                    names = t.getnames()
                if not any(n.endswith(".py") for n in names):
                    warns.append(f"{iid}: snapshot has no .py files")
            except Exception as e:
                errs.append(f"{iid}: snapshot unreadable: {e}")
        if not os.path.exists(gj):
            errs.append(f"{iid}: no graph")
        else:
            g = json.load(open(gj))
            if not g.get("nodes") or not g.get("edges"):
                warns.append(f"{iid}: graph empty nodes/edges")
        if not os.path.exists(ej):
            errs.append(f"{iid}: no embedding")
        else:
            import numpy as np
            z = np.load(ej)
            keys = list(z.keys())
            bad = [k for k in keys[:50] if z[k].shape != (256,)]
            if bad:
                warns.append(f"{iid}: {len(bad)} emb vectors with wrong shape")
            # cross-check embedding ids vs graph node ids
            if os.path.exists(gj):
                gids = {n["id"] for n in json.load(open(gj))["nodes"]}
                if set(keys) != gids:
                    warns.append(f"{iid}: emb/graph id mismatch "
                                 f"({len(keys)} emb vs {len(gids)} graph)")

    print(f"{len(recs)} tasks | repos: {repos}")
    print(f"files: {len(os.listdir(os.path.join(task_dir,'snapshots')))} snaps, "
          f"{len(os.listdir(os.path.join(task_dir,'graphs')))} graphs, "
          f"{len(os.listdir(os.path.join(task_dir,'embeddings')))} embs")
    for w in warns:
        print("WARN:", w)
    for e in errs:
        print("ERR:", e)
    print(f"RESULT: {len(errs)} errors, {len(warns)} warnings")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/tasks")
