import json, os, subprocess, sys, tarfile, tempfile, shutil
from concurrent.futures import ThreadPoolExecutor

TD = "data/tasks_v3"
PIPE = os.path.abspath("pipeline")
tasks = [json.loads(l) for l in open(f"{TD}/tasks.jsonl")]

def missing(r):
    inst = r["instance_id"]
    return not (os.path.exists(f"{TD}/graphs/{inst}.json")
                and os.path.exists(f"{TD}/embeddings/{inst}.npz"))

def gen(r):
    inst = r["instance_id"]
    work = tempfile.mkdtemp(prefix="g_")
    try:
        with tarfile.open(f"{TD}/snapshots/{inst}.tgz") as t:
            t.extractall(work)
        repo = inst.rsplit("_", 1)[0]
        if subprocess.run([sys.executable, f"{PIPE}/graphs.py", work, repo,
                           f"{TD}/graphs/{inst}.json"],
                          capture_output=True).returncode != 0:
            return inst, "graph-fail"
        if subprocess.run([sys.executable, f"{PIPE}/embeddings.py",
                           f"{TD}/graphs/{inst}.json",
                           f"{TD}/embeddings/{inst}.npz"],
                          capture_output=True).returncode != 0:
            return inst, "embed-fail"
        return inst, "ok"
    finally:
        shutil.rmtree(work, ignore_errors=True)

todo = [r for r in tasks if missing(r)]
print(f"{len(todo)} to generate")
done = 0
with ThreadPoolExecutor(max_workers=3) as ex:
    for inst, st in ex.map(gen, todo):
        done += 1
        if st != "ok":
            print("FAIL", inst, st)
        elif done % 10 == 0:
            print(f"{done}/{len(todo)}", flush=True)
print("PARALLEL GEN DONE")
