#!/usr/bin/env python3
"""Dataset integrity & difficulty audit for SWE-Pallets-X.

Computes, for the packaged task set:
  * uniqueness of instance_id / fix_commit (no duplicate tasks)
  * era (fix_date) coverage per repo
  * patch-size / file-count difficulty distribution
  * asset completeness (snapshot, graph, embeddings on disk)
  * discriminative-power check: does the existing agent baseline
    resolve easier tasks more often? (task metadata vs outcome)

Usage: python pipeline/audit.py <tasks_dir> [--eval results.jsonl]
"""
import json
import os
import sys
import tarfile
from collections import Counter, defaultdict


def main(tasks_dir, eval_path=None):
    tj = os.path.join(tasks_dir, "tasks.jsonl")
    tasks = [json.loads(l) for l in open(tj)]
    print(f"== AUDIT {tasks_dir}: {len(tasks)} tasks ==\n")

    # --- uniqueness ---
    ids = Counter(t["instance_id"] for t in tasks)
    fcs = Counter((t["repo"], t["fix_commit"]) for t in tasks)
    dup_ids = {k: v for k, v in ids.items() if v > 1}
    dup_fc = {k: v for k, v in fcs.items() if v > 1}
    print(f"unique instance_id: {len(ids)}/{len(tasks)} "
          f"(dups: {len(dup_ids)})")
    print(f"unique (repo,fix_commit): {len(fcs)}/{len(tasks)} "
          f"(dups: {len(dup_fc)})")

    # --- era coverage (created_at is ISO) ---
    eras = defaultdict(list)
    for t in tasks:
        eras[t["repo"]].append(t["created_at"][:10])
    print("\nfix_date range per repo:")
    for r in sorted(eras):
        ds = sorted(eras[r])
        print(f"  {r:26s} {ds[0]} .. {ds[-1]}  n={len(ds)}")

    # --- difficulty distribution ---
    pl = sorted(t.get("n_patch_lines", 0) for t in tasks)
    sf = sorted(t.get("n_src_files", 0) for t in tasks)
    tf = sorted(t.get("n_test_files", 0) for t in tasks)

    def pct(a, p):
        return a[min(len(a) - 1, int(len(a) * p))]

    print(f"\npatch lines   p10/p50/p90/max: "
          f"{pct(pl,.1)}/{pct(pl,.5)}/{pct(pl,.9)}/{pl[-1]}")
    print(f"source files  p10/p50/p90/max: "
          f"{pct(sf,.1)}/{pct(sf,.5)}/{pct(sf,.9)}/{sf[-1]}")
    print(f"test files    p10/p50/p90/max: "
          f"{pct(tf,.1)}/{pct(tf,.5)}/{pct(tf,.9)}/{tf[-1]}")

    # --- asset completeness ---
    ok_s = ok_g = ok_e = 0
    snap_idx = os.path.join(tasks_dir, "snapshots")
    snap_tar = snap_idx + ".tar.gz"
    snap_names = None
    if os.path.isdir(snap_idx):
        snap_names = {os.path.splitext(f)[0] for f in
                      os.listdir(snap_idx)
                      if f.endswith((".tar.gz", ".tgz"))}
    elif os.path.exists(snap_tar):
        snap_names = set()
        with tarfile.open(snap_tar) as tar:
            for m in tar.getnames():
                b = os.path.basename(m)
                if b.endswith(".tar.gz"):
                    snap_names.add(b[:-7])
    gdir = os.path.join(tasks_dir, "graphs")
    edir = os.path.join(tasks_dir, "embeddings")
    gset = {f[:-5] for f in os.listdir(gdir)} if os.path.isdir(gdir) \
        else {f[:-5] for f in os.listdir(gdir + ".tar")} \
        if os.path.exists(gdir + ".tar") else set()
    eset = set()
    if os.path.isdir(edir):
        eset = {f[:-4] for f in os.listdir(edir) if f.endswith(".npz")}
    for t in tasks:
        i = t["instance_id"]
        ok_s += snap_names is not None and i in snap_names
        ok_g += i in gset or i + ".json" in gset
        ok_e += i in eset
    print(f"\nassets: snapshot {ok_s}/{len(tasks)} | "
          f"graph {ok_g}/{len(tasks)} | emb {ok_e}/{len(tasks)}")

    # --- discriminative power (needs eval results) ---
    if eval_path and os.path.exists(eval_path):
        res = {json.loads(l)["instance_id"]: json.loads(l)
               for l in open(eval_path)}
        meta = {t["instance_id"]: t for t in tasks}
        joined = [(meta[i], r) for i, r in res.items() if i in meta]
        print(f"\n== outcome join: {len(joined)} evaluated tasks ==")
        # bucket by patch size
        buckets = {"small(<30)": [], "mid(30-120)": [], "big(>120)": []}
        for t, r in joined:
            n = t.get("n_patch_lines", 0)
            b = ("small(<30)" if n < 30 else
                 "mid(30-120)" if n <= 120 else "big(>120)")
            buckets[b].append(str(r.get("passed")) == "True")
        for b, v in buckets.items():
            if v:
                print(f"  patch {b:14s} n={len(v):3d} "
                      f"resolved={sum(v)} ({sum(v)/len(v)*100:.1f}%)")


if __name__ == "__main__":
    td = sys.argv[1]
    ep = None
    if "--eval" in sys.argv:
        ep = sys.argv[sys.argv.index("--eval") + 1]
    main(td, ep)
