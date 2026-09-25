#!/usr/bin/env python3
"""Two-phase verification (mirrors the competition's harness):

Phase 1 (Fail-to-Pass): checkout base_commit, apply test_patch only,
    run the touched test files -> MUST FAIL (non-zero exit).
Phase 2 (Pass-to-Pass): additionally apply patch -> tests MUST PASS.

Environment: one shared venv per (repo, python version); package installed
editable per-instance via `pip install --no-deps -e .` then deps best-effort.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed


def run(cmd, cwd=None, env=None, timeout=600):
    try:
        r = subprocess.run(
            cmd, cwd=cwd, env=env, capture_output=True, text=True,
            timeout=timeout, shell=isinstance(cmd, str),
        )
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def make_venv(py, path):
    code, out = run([py, "-m", "venv", path])
    if code != 0:
        return False, out
    pip = os.path.join(path, "bin", "pip")
    run([pip, "install", "-q", "--upgrade", "pip", "setuptools", "wheel"])
    return True, ""


def venv_bin(venv, name):
    return os.path.join(venv, "bin", name)


def get_dep_names(repo, base):
    """Extract dependency names declared at base_commit."""
    deps = set()
    code, out = run(["git", "-C", repo, "show", f"{base}:pyproject.toml"])
    if code == 0:
        try:
            import tomllib
            data = tomllib.loads(out)
            proj = data.get("project", {})
            reqs = list(proj.get("dependencies") or [])
            for extra in (proj.get("optional-dependencies") or {}).values():
                reqs += extra
            for gname, grp in (data.get("dependency-groups") or {}).items():
                if gname.lower() in ("test", "tests", "testing"):
                    reqs += [g for g in grp if isinstance(g, str)]
            for r in reqs:
                name = r.split(";")[0].strip()
                name = re.split(r"[<>=!~\[ ]", name)[0].strip()
                if name:
                    deps.add(name)
        except Exception:
            pass
    for fname in ("requirements.txt", "requirements/main.txt", "requirements/base.txt"):
        code, out = run(["git", "-C", repo, "show", f"{base}:{fname}"])
        if code == 0:
            for ln in out.splitlines():
                ln = ln.strip()
                if ln and not ln.startswith(("#", "-")):
                    name = re.split(r"[<>=!~\[ ;]", ln)[0].strip()
                    if name and re.match(r"^[A-Za-z0-9_.-]+$", name):
                        deps.add(name)
    # setup.cfg [options] install_requires block
    code, out = run(["git", "-C", repo, "show", f"{base}:setup.cfg"])
    if code == 0:
        in_ir = False
        for ln in out.splitlines():
            if re.match(r"^\s*install_requires\s*=", ln):
                in_ir = True
                continue
            if in_ir:
                if re.match(r"^\s", ln) and ln.strip():
                    name = re.split(r"[<>=!~\[ ;]", ln.strip())[0].strip()
                    if name and re.match(r"^[A-Za-z0-9_.-]+$", name):
                        deps.add(name)
                else:
                    in_ir = False
    # setup.py requirement entries like 'Werkzeug >= 2.2.2'
    code, out = run(["git", "-C", repo, "show", f"{base}:setup.py"])
    if code == 0:
        for m in re.finditer(r"""['"]([A-Za-z0-9_.-]+)\s*[<>=!~]""", out):
            deps.add(m.group(1))
    junk = {"python", "name", "version", "description", "author", "url", "license"}
    own = os.path.basename(os.path.abspath(repo)).lower().replace("-", "_")
    return sorted(
        d for d in deps
        if d.lower() not in junk and len(d) > 1
        and d.lower().replace("-", "_") != own
    )


def era_constraints(repo, base, date, work, extra=None):
    """Pin deps to latest release before the commit date; returns file path."""
    deps = get_dep_names(repo, base) + list(extra or [])
    if not deps:
        return None
    pipeline = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    code, out = run(
        [sys.executable, os.path.join(pipeline, "era_pin.py"), date, *deps],
        timeout=120,
    )
    if code != 0 or not out.strip():
        return None
    cf = os.path.join(work, "_constraints.txt")
    with open(cf, "w") as f:
        f.write(out)
    return cf


def pick_python(date_str):
    """Era-appropriate interpreter: newer pythons break old test suites."""
    try:
        year = int(date_str[:4])
    except Exception:
        year = 2023
    if year >= 2025:
        return "3.13"
    if year >= 2024:
        return "3.12"
    if year >= 2022:
        return "3.11"
    return "3.10"


def uv_pip(venv, args, cwd=None, timeout=600):
    return run(["uv", "pip", "install", "--python", venv_bin(venv, "python"), *args],
               cwd=cwd, timeout=timeout)


def prep_instance(rec, repo, py, work):
    """Return (f2p_fail, p2p_pass, log)."""
    base, fix = rec["base_commit"], rec["fix_commit"]
    tp = pp = None
    # fresh worktree at base commit
    code, out = run(["git", "-C", repo, "worktree", "add", "--detach", work, base])
    if code != 0:
        return None, None, f"worktree: {out[:300]}"
    try:
        # dedicated venv per instance (parallel-safe), era-matched python
        venv = os.path.join(work, ".venv")
        pyver = pick_python(rec.get("created_at") or rec["date"])
        code, out = run(["uv", "venv", "--python", pyver, venv], timeout=120)
        if code != 0:
            return None, None, f"venv({pyver}): {out[:300]}"
        # era-consistent deps: pin to releases before the commit date.
        # pytest must be era-pinned too (old conftests break on new pytest).
        cf = era_constraints(repo, base, rec.get("created_at") or rec["date"], work,
                             extra=["pytest"])
        dep_args = ["--constraints", cf] if cf else []
        extra_deps = [d for d in get_dep_names(repo, base)
                      if d.lower() not in ("flask",)]
        code, out = uv_pip(
            venv, ["-e", ".", "pytest", *extra_deps, *dep_args],
            cwd=work, timeout=600)
        if code != 0:
            code, out = uv_pip(
                venv, [".", "pytest", *extra_deps, *dep_args],
                cwd=work, timeout=600)
            if code != 0:
                return None, None, f"install: {out[-500:]}"

        tests = " ".join(rec["test_files"])
        pytest = venv_bin(venv, "python")
        cmd = f"{pytest} -m pytest -x -q {tests}"
        # Phase 1: test_patch only -> expect fail
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as tf:
            tf.write(rec["test_patch"])
            tp = tf.name
        code, out = run(["git", "-C", work, "apply", "--whitespace=fix", tp], cwd=work)
        if code != 0:
            return None, None, f"test_patch apply: {out[:300]}"
        code1, out1 = run(cmd, cwd=work, timeout=600)
        f2p_fail = code1 != 0
        # Phase 2: apply src patch -> expect pass
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as pf:
            pf.write(rec["patch"])
            pp = pf.name
        code, out = run(["git", "-C", work, "apply", "--whitespace=fix", pp], cwd=work)
        if code != 0:
            return f2p_fail, None, f"patch apply: {out[:300]}"
        code2, out2 = run(cmd, cwd=work, timeout=600)
        p2p_pass = code2 == 0
        return f2p_fail, p2p_pass, f"f2p={code1} p2p={code2}\n{out1[-200:]}\n{out2[-200:]}"
    finally:
        run(["git", "-C", repo, "worktree", "remove", "--force", work])
        for p in (tp, pp):
            if p:
                try: os.unlink(p)
                except Exception: pass


def main():
    in_path, repo, py, out_path = sys.argv[1:5]
    repo = os.path.abspath(repo)
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    done = set()
    try:
        for line in open(out_path):
            done.add(json.loads(line)["fix_commit"])
    except FileNotFoundError:
        pass
    recs = [json.loads(l) for l in open(in_path)]
    todo = [r for r in recs if r["fix_commit"] not in done]
    print(f"{len(todo)} to verify ({len(done)} done)", file=sys.stderr)

    fout = open(out_path, "a")
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for r in todo:
            work = tempfile.mkdtemp(prefix="wt_")
            futs[ex.submit(prep_instance, r, repo, py, work)] = (r, work)
        for fut in as_completed(futs):
            r, work = futs[fut]
            try:
                f2p, p2p, log = fut.result()
            except Exception as e:
                f2p, p2p, log = None, None, f"exc: {e}"
            shutil.rmtree(work, ignore_errors=True)
            r["verify"] = {"f2p_fail": f2p, "p2p_pass": p2p, "log": log[-500:]}
            r["verified"] = bool(f2p and p2p)
            if r["verified"]:
                ok += 1
            fout.write(json.dumps(r) + "\n")
            fout.flush()
            print(f"{r['instance_id']}: f2p={f2p} p2p={p2p}", file=sys.stderr)
    print(f"VERIFIED: {ok}", file=sys.stderr)


if __name__ == "__main__":
    main()
