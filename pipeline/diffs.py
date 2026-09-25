#!/usr/bin/env python3
"""Extract patch (src diff) and test_patch (test diff) for each candidate.

patch: base_commit..fix_commit restricted to non-test .py files
test_patch: same range restricted to test files
"""
import json
import subprocess
import sys


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, check=True
    ).stdout


def diff(repo, base, fix, files):
    if not files:
        return ""
    return git(repo, "diff", f"{base}..{fix}", "--", *files)


def main():
    in_path, repo, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    fout = open(out_path, "w")
    n = 0
    for line in open(in_path):
        c = json.loads(line)
        c["patch"] = diff(repo, c["base_commit"], c["fix_commit"], c["src_files"])
        c["test_patch"] = diff(repo, c["base_commit"], c["fix_commit"], c["test_files"])
        if not c["patch"].strip() or not c["test_patch"].strip():
            continue
        fout.write(json.dumps(c) + "\n")
        n += 1
    print(f"{n} with non-empty patch+test_patch", file=sys.stderr)


if __name__ == "__main__":
    main()
