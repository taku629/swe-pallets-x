#!/bin/bash
# Expand SWE-Pallets-X: enrich + diff-extract for new repos.
cd /home/tk250127/kaggle_comp
set -x
# subsample big repos for tractability
python3 - <<'EOF'
import json, random
random.seed(0)
for name, cap in [("pytest", 250), ("sympy", 250)]:
    rows = [json.loads(l) for l in open(f"data/candidates/{name}.jsonl")]
    rows.sort(key=lambda r: r["date"])
    # stratified by year: keep a uniform slice across history
    if len(rows) > cap:
        step = len(rows) / cap
        rows = [rows[int(i * step)] for i in range(cap)]
    open(f"data/candidates/{name}_sub.jsonl", "w").write(
        "".join(json.dumps(r) + "\n" for r in rows))
    print(name, len(rows))
EOF

for spec in "jinja2 pallets/jinja2 jinja2" \
            "itsdangerous pallets/itsdangerous itsdangerous" \
            "markupsafe pallets/markupsafe markupsafe" \
            "pytest pytest-dev/pytest pytest_sub" \
            "sympy sympy/sympy sympy_sub"; do
  set -- $spec
  short=$1; full=$2; cand=$3
  python3 pipeline/enrich.py data/candidates/${cand}.jsonl $full \
    data/candidates/${short}_enriched.jsonl
  python3 pipeline/diffs.py data/candidates/${short}_enriched.jsonl \
    data/repos/$short data/candidates/${short}_diff.jsonl
done
echo "ALL ENRICH+DIFF DONE"
