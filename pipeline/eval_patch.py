#!/usr/bin/env python3
"""Held-out patch-generation eval: generate diff -> git apply -> run tests.

Phase 1 (GPU): eval_patch.py gen <sft_test.jsonl> <out.jsonl> [--lora DIR]
Phase 2 (CPU): eval_patch.py check <out.jsonl> <tasks_dir> <out_checked.jsonl>

PASS criterion mirrors the competition: model patch applies to the frozen
snapshot AND the task's test files pass after applying test_patch + patch.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify import era_constraints, get_dep_names, pick_python, to_iso, run  # noqa

DIFF_RE = re.compile(r"(```(?:diff)?\n(.*?)```)", re.S)


def gen(test_path, out_path, lora=None, max_new=1024):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    recs = [json.loads(l) for l in open(test_path)]
    done = {json.loads(l)["instance_id"]
            for l in open(out_path)} if os.path.exists(out_path) else set()
    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(
        lora or model_id)
    if lora:
        from peft import PeftModel
        m = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        m = PeftModel.from_pretrained(m, lora)
    else:
        m = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
    m.eval()
    fo = open(out_path, "a")
    for i, r in enumerate(recs):
        if r["instance_id"] in done:
            continue
        msgs = [{"role": "user", "content": r["user"]}]
        x = tok.apply_chat_template(
            msgs, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(m.device)
        with torch.no_grad():
            y = m.generate(**x, max_new_tokens=max_new,
                           do_sample=False, pad_token_id=tok.eos_token_id)
        r["gen"] = tok.decode(
            y[0][x["input_ids"].shape[1]:], skip_special_tokens=True)
        fo.write(json.dumps(r) + "\n")
        fo.flush()
        print(f"[{i+1}/{len(recs)}] {r['instance_id']} "
              f"gen={len(r['gen'])} chars", file=sys.stderr)


def extract_diff(text):
    m = DIFF_RE.search(text)
    body = m.group(2) if m else text
    # keep only unified-diff lines
    lines = [l for l in body.splitlines()
             if l.startswith(("diff ", "index ", "--- ", "+++ ", "@@",
                              "+", "-", " "))]
    if not any(l.startswith("---") for l in lines):
        return None
    # renumber every @@ header from actual hunk content — models often
    # miscount or truncate, and git apply rejects mismatched counts
    out = []
    i = 0
    while i < len(lines):
        l = lines[i]
        if not l.startswith("@@"):
            out.append(l)
            i += 1
            continue
        hm = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)",
                      l)
        if hm is None:
            out.append(l)
            i += 1
            continue
        j = i + 1
        n_old = n_new = 0
        while j < len(lines) and not lines[j].startswith("@@"):
            if lines[j].startswith("diff "):
                break
            n_old += lines[j].startswith(("-", " "))
            n_new += lines[j].startswith(("+", " "))
            j += 1
        out.append(f"@@ -{hm.group(1)},{n_old} "
                   f"+{hm.group(3)},{n_new} @@{hm.group(5)}")
        out.extend(lines[i + 1:j])
        i = j
    if not any(l.startswith("---") for l in out):
        return None
    return "\n".join(out) + "\n"


def check(gen_path, tasks_dir, out_path):
    recs = {json.loads(l)["instance_id"]: json.loads(l)
            for l in open(os.path.join(tasks_dir, "tasks.jsonl"))}
    done = {json.loads(l)["instance_id"]
            for l in open(out_path)} if os.path.exists(out_path) else set()
    fo = open(out_path, "a")
    for line in open(gen_path):
        g = json.loads(line)
        iid = g["instance_id"]
        if iid in done or iid not in recs:
            continue
        t = recs[iid]
        res = {"instance_id": iid, "repo": t["repo"],
               "apply_ok": False, "passed": False}
        diff = extract_diff(g["gen"])
        if not diff:
            res["why"] = "no-diff"
            fo.write(json.dumps(res) + "\n"); fo.flush(); continue
        work = tempfile.mkdtemp(prefix="pe_")
        try:
            with __import__("tarfile").open(
                    os.path.join(tasks_dir, "snapshots", iid + ".tgz")) as tf:
                tf.extractall(work, filter="data")
            with open(os.path.join(work, "_mp.patch"), "w") as f:
                f.write(diff)
            c, _ = run(["git", "apply", "--check", "_mp.patch"], cwd=work)
            if c != 0:
                res["why"] = "apply-fail"
                fo.write(json.dumps(res) + "\n"); fo.flush(); continue
            res["apply_ok"] = True
            # test env: era-pinned deps, repo installed editable
            era = to_iso(t.get("fix_date") or t.get("created_at"))
            venv = os.path.join(work, ".venv")
            run(["uv", "venv", "--python", pick_python(era), venv],
                timeout=120)
            own = t["repo"].split("/")[-1].lower().replace("-", "_")
            self_pt = own == "pytest"
            cf = era_constraints(os.path.join(tasks_dir, "..", "repos", own),
                                 t["base_commit"], era, work,
                                 extra=[] if self_pt else ["pytest"])
            deps = [d for d in get_dep_names(
                        os.path.join(tasks_dir, "..", "repos", own),
                        t["base_commit"])
                    if d.lower() not in ("flask", own)]
            resolved = set()
            if cf:
                for ln in open(cf):
                    resolved.add(ln.split("==")[0].strip().lower()
                                 .replace("-", "_"))
            deps = [d for d in deps
                    if not resolved or d.lower().replace("-", "_") in resolved]
            inst = ["-e", ".", *([] if self_pt else ["pytest"]), *deps] + \
                   (["--constraints", cf] if cf else [])
            c, out = run(["uv", "pip", "install", "--python",
                          os.path.join(venv, "bin", "python"), *inst],
                         cwd=work, timeout=600)
            if c != 0:
                res["why"] = "install-fail"
                fo.write(json.dumps(res) + "\n"); fo.flush(); continue
            run(["git", "apply", "_mp.patch"], cwd=work)
            with open(os.path.join(work, "_tp.patch"), "w") as f:
                f.write(t["test_patch"])
            run(["git", "apply", "_tp.patch"], cwd=work)
            env = dict(os.environ,
                       PATH=os.path.join(venv, "bin") + ":" + os.environ["PATH"],
                       VIRTUAL_ENV=venv)
            c, out = run(["python", "-m", "pytest", "-x", "-q",
                          *t["test_files"]], cwd=work, env=env, timeout=300)
            res["passed"] = (c == 0)
            res["why"] = "" if c == 0 else f"tests rc={c}"
            res["test_tail"] = out[-300:]
        finally:
            __import__("shutil").rmtree(work, ignore_errors=True)
        fo.write(json.dumps(res) + "\n")
        fo.flush()
        print(f"{iid} apply={res['apply_ok']} pass={res['passed']}",
              file=sys.stderr)


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        lora = None
        if "--lora" in sys.argv:
            lora = sys.argv[sys.argv.index("--lora") + 1]
        gen(sys.argv[2], sys.argv[3], lora=lora)
    else:
        check(sys.argv[2], sys.argv[3], sys.argv[4])
