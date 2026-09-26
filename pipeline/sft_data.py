#!/usr/bin/env python3
"""Build SFT records from tasks.jsonl — oracle-context patch generation.

Each record: user = problem_statement + contents of the files the patch
touches (oracle retrieval setting, à la SWE-bench 'oracle'), assistant =
the unified-diff patch. Splits stratified by repo into train/test.
"""
import json
import os
import random
import re
import sys
import tarfile


def patched_files(patch):
    return re.findall(r"--- a/(\S+)", patch)


def extract_files(snapshot_tgz, paths, cap=6000):
    """Pull current contents of each patched file — stream only the
    members we need instead of unpacking the whole snapshot."""
    out = {}
    want = set(paths) | {"./" + p for p in paths}
    with tarfile.open(snapshot_tgz) as t:
        for m in t:
            if m.name in want or m.name.lstrip("./") in want:
                f = t.extractfile(m)
                if f:
                    out[m.name.lstrip("./")] = \
                        f.read().decode("utf-8", "replace")[:cap]
    return out


def main(tasks_dir, out_dir, max_lines=80, max_files=3, frac_test=0.2, seed=0):
    random.seed(seed)
    rows = [json.loads(l) for l in open(os.path.join(tasks_dir, "tasks.jsonl"))]
    by_repo = {}
    for r in rows:
        by_repo.setdefault(r["repo"], []).append(r)

    train, test = [], []
    for repo, recs in by_repo.items():
        random.shuffle(recs)
        n_test = max(1, int(round(len(recs) * frac_test)))
        test += recs[:n_test]
        train += recs[n_test:]

    def emit(recs, name):
        n_skip = 0
        with open(os.path.join(out_dir, f"sft_{name}.jsonl"), "w") as fo:
            for r in recs:
                if (r["n_patch_lines"] > max_lines
                        or r["n_src_files"] > max_files):
                    n_skip += 1
                    continue
                paths = patched_files(r["patch"])
                snap = os.path.join(tasks_dir, "snapshots",
                                    r["instance_id"] + ".tgz")
                if not paths or not os.path.exists(snap):
                    n_skip += 1
                    continue
                files = extract_files(snap, paths)
                ctx = "\n\n".join(
                    f"### {p}\n```python\n{txt}\n```" for p, txt in files.items())
                user = (
                    "Fix the following issue in this repository. Output a "
                    "unified diff patch.\n\n## Issue\n" + r["problem_statement"]
                    + "\n\n## Files\n" + ctx)
                rec = {"instance_id": r["instance_id"], "repo": r["repo"],
                       "user": user, "assistant": r["patch"],
                       "test_files": r["test_files"],
                       "test_patch": r["test_patch"]}
                fo.write(json.dumps(rec) + "\n")
        print(f"{name}: {sum(1 for _ in open(out_dir + '/sft_' + name + '.jsonl'))} written, {n_skip} skipped")

    os.makedirs(out_dir, exist_ok=True)
    emit(train, "train")
    emit(test, "test")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
