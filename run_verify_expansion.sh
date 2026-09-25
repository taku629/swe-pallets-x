#!/bin/bash
# Verify expansion candidates: jinja2, itsdangerous, markupsafe, pytest, sympy.
cd /home/tk250127/kaggle_comp
for r in jinja2 itsdangerous markupsafe pytest sympy; do
  echo "=== $r ==="
  python3 pipeline/verify.py data/candidates/${r}_diff.jsonl \
    data/repos/$r 3.11 data/verified/${r}.jsonl 6
done
echo "ALL VERIFY DONE"
