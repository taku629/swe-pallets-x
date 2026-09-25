#!/usr/bin/env python3
"""Mine git history for candidate SWE-bench-style instances.

Finds commits that modify both non-test .py files and test files together,
following the competition's documented curation approach:
- commit modifies core functional logic (.py) AND unit tests (test_*.py / *_test.py)
- the non-test diff becomes `patch`, the test diff becomes `test_patch`
- base_commit is the parent of the fix commit

Usage: python mine.py <repo_path> [--since 2021-01-01] [--out candidates.jsonl]
"""
import json
import re
import subprocess
import sys
from pathlib import Path

TEST_RE = re.compile(r"(^|/)(tests?|testing)/|test_.*\.py$|.*_test\.py$", re.I)
PY_RE = re.compile(r"\.py$", re.I)


def is_test_file(path: str) -> bool:
    return bool(PY_RE.search(path)) and bool(TEST_RE.search(path))


def is_source_py(path: str) -> bool:
    return bool(PY_RE.search(path)) and not is_test_file(path)


def git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, check=True
    ).stdout


def iter_candidate_commits(repo: str, since: str):
    # commits touching at least one .py file, with parent (skip merges: %P has 2+)
    fmt = "%H%x1f%P%x1f%ad%x1f%s"
    out = git(repo, "log", f"--since={since}", f"--format={fmt}", "--name-only")
    cur = None
    for line in out.splitlines():
        if "\x1f" in line:
            if cur:
                yield cur
            sha, parents, date, subj = line.split("\x1f")
            cur = {
                "sha": sha,
                "parents": parents.split(),
                "date": date,
                "subject": subj,
                "files": [],
            }
        elif line.strip() and cur is not None:
            cur["files"].append(line.strip())
    if cur:
        yield cur


def main():
    repo, since = sys.argv[1], "2021-01-01"
    out_path = None
    args = sys.argv[2:]
    for i, a in enumerate(args):
        if a == "--since":
            since = args[i + 1]
        if a == "--out":
            out_path = args[i + 1]

    repo_name = Path(repo).name
    n = 0
    fout = open(out_path, "w") if out_path else sys.stdout
    for c in iter_candidate_commits(repo, since):
        if len(c["parents"]) != 1:
            continue  # skip merges & root
        src = [f for f in c["files"] if is_source_py(f)]
        tst = [f for f in c["files"] if is_test_file(f)]
        if not src or not tst:
            continue
        rec = {
            "instance_id": f"{repo_name}_{c['sha'][:7]}",
            "repo_short": repo_name,
            "fix_commit": c["sha"],
            "base_commit": c["parents"][0],
            "date": c["date"],
            "subject": c["subject"],
            "src_files": src,
            "test_files": tst,
        }
        fout.write(json.dumps(rec) + "\n")
        n += 1
    print(f"[{repo_name}] {n} candidate commits", file=sys.stderr)


if __name__ == "__main__":
    main()
