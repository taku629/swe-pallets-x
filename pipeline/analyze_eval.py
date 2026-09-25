#!/usr/bin/env python3
"""Aggregate agent-eval results: resolve rates, per-repo splits,
step/runtime stats, tool-use histogram, failure taxonomy."""
import json
import re
import sys
from collections import Counter
from pathlib import Path


def classify(r):
    """Failure taxonomy from result row."""
    if r.get("passed"):
        return "resolved"
    if "error" in r:
        return "harness_error"
    th = r.get("tool_hist") or {}
    if th.get("<llm_err>", 0) >= 2:
        return "llm_timeout"
    if not r.get("has_patch"):
        if th.get("submit_patch") and not any(
                th.get(t) for t in ("edit_file", "write_file")):
            return "empty_submit_no_edit"
        if any(th.get(t) for t in ("edit_file", "write_file")):
            return "edits_failed_to_apply"
        return "no_edit_attempted"
    return "patch_failed_tests"


def main(*files):
    for f in files:
        rows = [json.loads(l) for l in open(f)]
        # dedupe: keep last result per instance
        rows = list({r["instance_id"]: r for r in rows}.values())
        n = len(rows)
        npass = sum(r["passed"] for r in rows)
        print(f"\n=== {f} ===")
        print(f"resolved: {npass}/{n} ({npass/n*100:.1f}%)")
        per = Counter()
        ok = Counter()
        for r in rows:
            repo = r["instance_id"].rsplit("_", 1)[0]
            per[repo] += 1
            ok[repo] += r["passed"]
        print("per-repo:", {k: f"{ok[k]}/{v}" for k, v in per.items()})
        cats = Counter(classify(r) for r in rows)
        print("outcome taxonomy:", dict(cats.most_common()))
        steps = [r.get("steps", 0) for r in rows]
        secs = [r.get("seconds", 0) for r in rows]
        print(f"steps: mean={sum(steps)/n:.1f} max={max(steps)} | "
              f"seconds: mean={sum(secs)/n:.0f} max={max(secs)}")
        tools = Counter()
        for r in rows:
            for k, v in (r.get("tool_hist") or {}).items():
                tools[k] += v
        print("tool calls:", dict(tools.most_common()))
        solved = [r["instance_id"] for r in rows if r["passed"]]
        print("solved:", solved)


if __name__ == "__main__":
    main(*sys.argv[1:])
