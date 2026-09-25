#!/usr/bin/env python3
"""Enrich candidate commits with GitHub PR/issue data via `gh api`.

For each candidate fix commit, find the associated PR, take its title/body as
the problem statement source, and resolve linked issues (fixes/closes #N) to
prefer the original issue report (closer to SWE-bench's problem_statement).
"""
import json
import re
import subprocess
import sys
import time

ISSUE_REF_RE = re.compile(
    r"(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+#(\d+)", re.I
)


def gh_api(path: str):
    r = subprocess.run(
        ["gh", "api", path], capture_output=True, text=True
    )
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def get_issue(full_repo: str, num: int):
    d = gh_api(f"repos/{full_repo}/issues/{num}")
    if not d or "pull_request" in d:
        return None
    return d


def main():
    in_path, full_repo = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3]
    done = set()
    try:
        with open(out_path) as f:
            for line in f:
                done.add(json.loads(line)["fix_commit"])
    except FileNotFoundError:
        pass

    fout = open(out_path, "a")
    n_ok = n_skip = 0
    for line in open(in_path):
        c = json.loads(line)
        if c["fix_commit"] in done:
            continue
        pulls = gh_api(f"repos/{full_repo}/commits/{c['fix_commit']}/pulls")
        rec = dict(c)
        if pulls:
            pr = pulls[0]
            rec["pr_number"] = pr["number"]
            rec["pr_title"] = pr["title"]
            rec["pr_body"] = (pr.get("body") or "")[:8000]
            rec["created_at"] = pr["created_at"]
            # prefer the original linked issue as problem statement
            m = ISSUE_REF_RE.search(rec["pr_body"] or "")
            if not m:
                m = ISSUE_REF_RE.search(c["subject"] or "")
            if m:
                iss = get_issue(full_repo, int(m.group(1)))
                if iss:
                    rec["issue_number"] = iss["number"]
                    rec["issue_title"] = iss["title"]
                    rec["issue_body"] = (iss.get("body") or "")[:8000]
                    rec["created_at"] = iss["created_at"]
        else:
            rec["created_at"] = c["date"]
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        n_ok += 1
        if n_ok % 25 == 0:
            print(f"  ...{n_ok} enriched", file=sys.stderr)
        time.sleep(0.05)
    print(f"done: {n_ok} enriched, {n_skip} skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
