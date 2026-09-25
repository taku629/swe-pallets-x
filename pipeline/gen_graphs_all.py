#!/usr/bin/env python3
"""For every task instance: extract snapshot, build AST graph + embeddings."""
import json
import os
import subprocess
import sys
import tarfile
import tempfile


def main():
    tasks_dir = sys.argv[1]
    pipeline = os.path.dirname(os.path.abspath(__file__))
    tj = [json.loads(l) for l in open(os.path.join(tasks_dir, "tasks.jsonl"))]
    for i, r in enumerate(tj):
        inst = r["instance_id"]
        gj = os.path.join(tasks_dir, "graphs", inst + ".json")
        ej = os.path.join(tasks_dir, "embeddings", inst + ".npz")
        if os.path.exists(gj) and os.path.exists(ej):
            continue
        snap = os.path.join(tasks_dir, "snapshots", inst + ".tgz")
        work = tempfile.mkdtemp(prefix="g_")
        try:
            with tarfile.open(snap) as t:
                t.extractall(work)
            repo = inst.rsplit("_", 1)[0]
            if os.system(
                f"{sys.executable} {pipeline}/graphs.py {work} {repo} {gj}"
            ) != 0:
                print(f"{inst}: graph failed", file=sys.stderr)
                continue
            if os.system(
                f"{sys.executable} {pipeline}/embeddings.py {gj} {ej}"
            ) != 0:
                print(f"{inst}: embed failed", file=sys.stderr)
                continue
            if i % 10 == 0:
                print(f"{i}/{len(tj)}", file=sys.stderr)
        finally:
            import shutil; shutil.rmtree(work, ignore_errors=True)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
