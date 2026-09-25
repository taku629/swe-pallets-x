#!/usr/bin/env python3
"""Package verified instances into the competition dataset layout:

  tasks.jsonl | snapshots/<id>.tgz | graphs/<id>.json | embeddings/<id>.npz

Snapshots are fresh single-commit git repos at base_commit (no future history).
Filters mirror the competition's de-noising step (patch size caps).
"""
import json
import os
import subprocess
import sys
import tarfile
import tempfile

MAX_PATCH_LINES = 200
MAX_SRC_FILES = 6


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def make_snapshot(repo, base, out_tgz):
    work = tempfile.mkdtemp(prefix="snap_")
    r = run(["git", "-C", repo, "worktree", "add", "--detach", work, base])
    if r.returncode != 0:
        return False, r.stderr[:200]
    try:
        # freeze: single squashed commit, no forward history
        gitdir = os.path.join(work, ".git")
        if os.path.isdir(gitdir):
            import shutil; shutil.rmtree(gitdir, ignore_errors=True)
        else:
            os.unlink(gitdir)
        run(["git", "init", "-q"], cwd=work)
        run(["git", "add", "-A"], cwd=work)
        run(["git", "-c", "user.email=b@b", "-c", "user.name=b",
             "commit", "-qm", "baseline"], cwd=work)
        with tarfile.open(out_tgz, "w:gz") as t:
            t.add(work, arcname=".")
        return True, ""
    finally:
        run(["git", "-C", repo, "worktree", "remove", "--force", work])
        import shutil; shutil.rmtree(work, ignore_errors=True)


def problem_statement(rec):
    if rec.get("issue_body"):
        title = rec.get("issue_title", "")
        return (title + "\n\n" + rec["issue_body"]).strip()[:12000]
    if rec.get("pr_body"):
        title = rec.get("pr_title", "")
        return (title + "\n\n" + rec["pr_body"]).strip()[:12000]
    return rec["subject"].strip()


def main():
    out_dir = sys.argv[1]
    inputs = sys.argv[2:]
    for d in ("snapshots", "graphs", "embeddings"):
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    tj = open(os.path.join(out_dir, "tasks.jsonl"), "w")
    kept = skipped = 0
    for path in inputs:
        repo = os.path.join(os.path.dirname(path), "..", "repos")
        for line in open(path):
            r = json.loads(line)
            if not r.get("verified"):
                skipped += 1
                continue
            n_lines = sum(1 for l in r["patch"].splitlines()
                          if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))
            if n_lines > MAX_PATCH_LINES or len(r["src_files"]) > MAX_SRC_FILES:
                skipped += 1
                continue
            inst = r["instance_id"]
            repo_dir = os.path.abspath(
                os.path.join("data", "repos", r["repo_short"]))
            ok, err = make_snapshot(repo_dir, r["base_commit"],
                                    os.path.join(out_dir, "snapshots", inst + ".tgz"))
            if not ok:
                print(f"{inst}: snapshot failed {err}", file=sys.stderr)
                skipped += 1
                continue
            rec = {
                "instance_id": inst,
                "repo": r.get("full_repo", r["repo_short"]),
                "base_commit": r["base_commit"],
                "problem_statement": problem_statement(r),
                "hints_text": "",
                "patch": r["patch"],
                "test_patch": r["test_patch"],
                "created_at": r.get("created_at", r["date"]),
            }
            tj.write(json.dumps(rec) + "\n")
            kept += 1
            if kept % 10 == 0:
                print(f"  {kept} packed", file=sys.stderr)
    tj.close()
    print(f"kept={kept} skipped={skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
