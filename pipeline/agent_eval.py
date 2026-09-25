#!/usr/bin/env python3
"""Minimal SWE-agent harness for baseline evaluation.

A small local model (Ollama) plays the role of the competition agent:
it receives the problem_statement and a toolset similar to the
competition harness (run_command / read_file / edit_file / write_file /
submit_patch, plus optional graph tools), acts for up to N steps inside
the frozen snapshot, and its final `git diff HEAD` is scored by running
the task's test_patch (PASS/FAIL) — mirroring the competition metric.

Usage:
  python agent_eval.py <tasks_dir> <model> [--graph] [--limit K]
                       [--max-steps 24] [--out results.jsonl]
"""
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify import era_constraints, pick_python  # noqa: E402

OLLAMA = "http://localhost:11434"

SYSTEM_BASE = """You are an autonomous software-engineering agent. A repository is mounted at /workspace (a git checkout). Resolve the reported issue by editing files.

Reply with EXACTLY ONE JSON object per turn: {"tool": "<name>", "args": {...}} or {"tool": "submit_patch"} when done.
Tools:
- run_command(command): bash in /workspace (pytest, grep, sed -n, git diff allowed). Max ~120s.
- read_file(filepath, start_line?, end_line?): 1-indexed inclusive slice.
- edit_file(filepath, old_string, new_string): exact-replace in file.
- write_file(filepath, content): create/overwrite.
Paths are relative to /workspace. Explore first with run_command (e.g. 'find . -name "*.py" | head -50' or 'grep -rn symbol .')."""
GRAPH_TOOLS = """- get_code_neighbors(node): neighbors of a symbol in the repo call graph.
- search_similar_code(query, k): top-k similar symbols by embedding.
- get_code_subgraph(node, depth): call-graph neighborhood around a symbol."""
SYSTEM_TAIL = """- submit_patch(): finish; your edits (git diff HEAD) are evaluated.
Be efficient: inspect, reproduce with pytest, make a minimal fix, submit."""


def system_prompt(use_graph):
    parts = [SYSTEM_BASE]
    if use_graph:
        parts.append(GRAPH_TOOLS)
    parts.append(SYSTEM_TAIL)
    return "\n".join(parts)


def chat(model, messages, timeout=300):
    body = json.dumps({"model": model, "messages": messages,
                       "stream": False,
                       "options": {"temperature": 0.2, "num_predict": 600,
                                   "num_ctx": 16384}}).encode()
    req = urllib.request.Request(OLLAMA + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["message"]["content"]


def parse_action(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        if isinstance(d, dict) and "tool" in d:
            return d
    except Exception:
        pass
    return None


class Tools:
    def __init__(self, work, graph=None, emb=None, pybin=None):
        self.work = work
        self.graph = graph
        self.emb = emb
        self.pybin = pybin
        self.done = False

    def _path(self, filepath):
        # model is told the repo lives at /workspace; map any spelling
        # ("/workspace/x", "workspace/x", "/x", "./x") into the workdir
        p = str(filepath).lstrip("/")
        if p.startswith("workspace/"):
            p = p[len("workspace/"):]
        p = os.path.normpath(p)
        if p.startswith("..") or os.path.isabs(p):
            raise ValueError("path outside workspace")
        return os.path.join(self.work, p)

    def run(self, a):
        t, args = a.get("tool"), a.get("args") or {}
        try:
            fn = getattr(self, "t_" + t, None)
            if fn is None:
                return f"unknown tool {t}"
            return fn(**args) if isinstance(args, dict) else fn()
        except Exception as e:
            return f"error: {e}"

    def t_run_command(self, command):
        if ".." in command:
            return "path traversal not allowed"
        env = dict(os.environ)
        if self.pybin:
            env["PATH"] = self.pybin + os.pathsep + env["PATH"]
            env["VIRTUAL_ENV"] = os.path.dirname(self.pybin)
        try:
            r = subprocess.run(command, shell=True, cwd=self.work,
                               capture_output=True, text=True, timeout=120,
                               executable="/bin/bash", env=env)
            out = (r.stdout + r.stderr)[-6000:]
            return f"exit={r.returncode}\n{out}"
        except subprocess.TimeoutExpired:
            return "TIMEOUT (120s)"

    def t_read_file(self, filepath, start_line=None, end_line=None):
        p = self._path(filepath)
        if not os.path.exists(p):
            return (f"error: no such file: {filepath}; locate it with "
                    "run_command 'find . -name \"*.py\" | grep <name>'")
        lines = open(p, errors="replace").read().splitlines()
        s = (start_line or 1) - 1
        e = end_line or len(lines)
        return "\n".join(f"{i+1}\t{l}" for i, l in enumerate(lines[s:e], s))[:6000]

    def t_edit_file(self, filepath, old_string, new_string, **_):
        p = self._path(filepath)
        if not os.path.exists(p):
            return (f"error: no such file: {filepath}; locate it with "
                    "run_command 'find . -name \"*.py\" | grep <name>'")
        src = open(p, errors="replace").read()
        if old_string not in src:
            return ("old_string not found; re-read the file to get exact "
                    "current content")
        open(p, "w").write(src.replace(old_string, new_string, 1))
        return "edited"

    def t_write_file(self, filepath, content):
        p = self._path(filepath)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(content)
        return "written"

    def t_submit_patch(self):
        self.done = True
        return "submitted"

    def t_get_code_neighbors(self, node, edge_type=None, max_neighbors=50):
        if not self.graph:
            return "graph tools disabled"
        out = []
        for e in self.graph["edges"]:
            if edge_type and e["type"] != edge_type:
                continue
            if e["source"] == node:
                out.append(f"-> {e['target']} ({e['type']})")
            elif e["target"] == node:
                out.append(f"<- {e['source']} ({e['type']})")
            if len(out) >= max_neighbors:
                break
        return "\n".join(out) or "no neighbors"

    def t_get_code_subgraph(self, node, depth=2, max_nodes=40):
        if not self.graph:
            return "graph tools disabled"
        adj = {}
        for e in self.graph["edges"]:
            adj.setdefault(e["source"], []).append((e["target"], e["type"]))
            adj.setdefault(e["target"], []).append((e["source"], e["type"]))
        seen, frontier = {node}, [node]
        out = []
        for _ in range(int(depth)):
            nxt = []
            for n in frontier:
                for m, t in adj.get(n, []):
                    if m not in seen:
                        seen.add(m)
                        out.append(f"{n} -[{t}]-> {m}")
                        nxt.append(m)
                        if len(out) >= max_nodes:
                            return "\n".join(out)
            frontier = nxt
        return "\n".join(out) or "no subgraph"

    def t_search_similar_code(self, query, k=10):
        if self.emb is None:
            return "graph tools disabled"
        vec, svd, ids, E, texts = self.emb
        import numpy as np
        q = svd.transform(vec.transform([query]))
        if q.shape[1] < E.shape[1]:
            q = np.pad(q, ((0, 0), (0, E.shape[1] - q.shape[1])))
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = E @ q[0]
        top = np.argsort(-sims)[:k]
        return "\n".join(f"{ids[i]} ({sims[i]:.3f})" for i in top)


def setup_env(work, rec, envdir):
    pyver = pick_python(rec.get("created_at", "2023"))
    venv = os.path.join(envdir, "venv")
    subprocess.run(["uv", "venv", "--python", pyver, venv],
                   capture_output=True, timeout=120)
    # deps from the extracted snapshot itself (era-pinned incl. pytest)
    deps = _deps_from_tree(work)
    cutoff = rec.get("created_at") or "2023"
    cons_out = subprocess.run(
        [sys.executable,
         os.path.join(os.path.dirname(os.path.abspath(__file__)), "era_pin.py"),
         cutoff, "pytest", *deps],
        capture_output=True, text=True, timeout=120).stdout
    cf = os.path.join(envdir, "_cons.txt")
    open(cf, "w").write(cons_out)
    args = ["--constraints", cf] if cons_out.strip() else []
    pip = ["uv", "pip", "install", "--python", os.path.join(venv, "bin", "python")]
    r = subprocess.run(pip + ["-e", ".", "pytest", *deps, *args], cwd=work,
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        subprocess.run(pip + [".", "pytest", *deps, *args], cwd=work,
                       capture_output=True, timeout=600)


def _deps_from_tree(root):
    import tomllib
    deps = set()
    pp = os.path.join(root, "pyproject.toml")
    if os.path.exists(pp):
        try:
            d = tomllib.loads(open(pp).read())
            proj = d.get("project", {})
            reqs = list(proj.get("dependencies") or [])
            for grp in (d.get("dependency-groups") or {}).values():
                reqs += [g for g in grp if isinstance(g, str)]
            for x in (proj.get("optional-dependencies") or {}).values():
                reqs += x
            for r in reqs:
                n = re.split(r"[<>=!~\[ ;]", r)[0].strip()
                if n:
                    deps.add(n)
        except Exception:
            pass
    sp = os.path.join(root, "setup.py")
    if os.path.exists(sp):
        for m in re.finditer(r"""['"]([A-Za-z0-9_.-]+)\s*[<>=!~]""", open(sp).read()):
            deps.add(m.group(1))
    return sorted(deps)


def evaluate(task_dir, model, rec, use_graph, max_steps, out_dir=None):
    inst = rec["instance_id"]
    work = tempfile.mkdtemp(prefix=f"eval_{inst}_")
    envdir = tempfile.mkdtemp(prefix=f"env_{inst}_")
    try:
        with tarfile.open(os.path.join(task_dir, "snapshots", inst + ".tgz")) as t:
            t.extractall(work)
        setup_env(work, rec, envdir)

        graph = emb = None
        if use_graph:
            gj = os.path.join(task_dir, "graphs", inst + ".json")
            ej = os.path.join(task_dir, "embeddings", inst + ".npz")
            if os.path.exists(gj):
                graph = json.load(open(gj))
                # fit query encoder on graph node texts (same recipe as embeddings.py)
                import numpy as np
                from sklearn.decomposition import TruncatedSVD
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.preprocessing import normalize
                ids = [n["id"] for n in graph["nodes"]]
                texts = [(n.get("text") or n["id"]) for n in graph["nodes"]]
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                      max_features=200_000)
                X = vec.fit_transform(texts)
                n_comp = min(256, max(2, min(X.shape) - 1))
                svd = TruncatedSVD(n_components=n_comp, random_state=0).fit(X)
                E = svd.transform(X)
                if E.shape[1] < 256:
                    E = np.pad(E, ((0, 0), (0, 256 - E.shape[1])))
                E = normalize(E.astype(np.float32))
                emb = (vec, svd, ids, E, texts)
        tools = Tools(work, graph, emb,
                      pybin=os.path.join(envdir, "venv", "bin"))

        msgs = [{"role": "system", "content": system_prompt(use_graph)},
                {"role": "user", "content": "ISSUE:\n" + rec["problem_statement"][:4000]}]
        t0 = time.time()
        steps = 0
        tool_hist = {}
        while steps < max_steps and not tools.done and time.time() - t0 < 600:
            try:
                reply = chat(model, msgs)
            except Exception as e:
                msgs.append({"role": "user", "content":
                             f"[llm timeout/error: {e}] continue."})
                tool_hist["<llm_err>"] = tool_hist.get("<llm_err>", 0) + 1
                steps += 1
                continue
            msgs.append({"role": "assistant", "content": reply})
            a = parse_action(reply)
            if a is None:
                msgs.append({"role": "user", "content":
                             "Reply with ONE JSON tool call only."})
                tool_hist["<unparsed>"] = tool_hist.get("<unparsed>", 0) + 1
                steps += 1
                continue
            tool_hist[a["tool"]] = tool_hist.get(a["tool"], 0) + 1
            out = tools.run(a)
            msgs.append({"role": "user", "content": str(out)[:6000]})
            steps += 1
        # capture patch
        r = subprocess.run(["git", "add", "-N", "."], cwd=work, capture_output=True)
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=work,
                              capture_output=True, text=True).stdout
        # score: apply test_patch, run tests
        tp = os.path.join(envdir, "_tp.patch")
        open(tp, "w").write(rec["test_patch"])
        subprocess.run(["git", "apply", "--whitespace=fix", tp], cwd=work,
                       capture_output=True)
        tests = " ".join(_test_files(rec))
        py = os.path.join(envdir, "venv", "bin", "python")
        rr = subprocess.run(f"{py} -m pytest -x -q {tests}", shell=True,
                            cwd=work, capture_output=True, text=True, timeout=600)
        passed = rr.returncode == 0
        # save transcript for failure analysis
        tdir = os.path.join(os.path.dirname(os.path.abspath(out_dir or ".")),
                            "transcripts")
        os.makedirs(tdir, exist_ok=True)
        with open(os.path.join(tdir, inst + ".json"), "w") as f:
            json.dump(msgs[2:], f)  # skip system prompt
        return {"instance_id": inst, "steps": steps,
                "seconds": round(time.time() - t0, 1),
                "has_patch": bool(diff.strip()), "diff": diff[:20000],
                "tool_hist": tool_hist, "passed": passed,
                "tail": (rr.stdout + rr.stderr)[-300:]}
    except Exception as e:
        stage = "agent" if "msgs" in dir() and msgs else "setup"
        return {"instance_id": inst, "error": f"{stage}: {e}"[:300],
                "passed": False}
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(envdir, ignore_errors=True)


def _test_files(rec):
    # extract test file paths from the test_patch diff
    out = re.findall(r"\+\+\+ b/(\S+\.py)", rec["test_patch"])
    return out or ["tests"]


def main():
    task_dir, model = sys.argv[1], sys.argv[2]
    use_graph = "--graph" in sys.argv
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 30
    steps = int(sys.argv[sys.argv.index("--max-steps") + 1]) if "--max-steps" in sys.argv else 24
    out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "results.jsonl"
    recs = [json.loads(l) for l in open(os.path.join(task_dir, "tasks.jsonl"))]
    done = set()
    try:
        for l in open(out_path):
            r = json.loads(l)
            if "error" not in r:  # errored rows are retried
                done.add(r["instance_id"])
    except FileNotFoundError:
        pass
    todo = [r for r in recs if r["instance_id"] not in done][:limit]
    print(f"{len(todo)} tasks | model={model} graph={use_graph}", file=sys.stderr)
    fout = open(out_path, "a")
    npass = 0
    for i, r in enumerate(todo):
        res = evaluate(task_dir, model, r, use_graph, steps,
                       out_dir=out_path)
        res["model"] = model
        res["graph"] = use_graph
        fout.write(json.dumps(res) + "\n")
        fout.flush()
        npass += bool(res.get("passed"))
        print(f"[{i+1}/{len(todo)}] {r['instance_id']} steps={res.get('steps')} "
              f"patch={res.get('has_patch')} PASS={res.get('passed')}",
              file=sys.stderr)
    print(f"resolve rate: {npass}/{len(todo)}", file=sys.stderr)


if __name__ == "__main__":
    main()
