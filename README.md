# SWE-Pallets-X

**Verified extension benchmark for small-model software-engineering agents**, in the
[Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent)
dataset format. Companion code for the Kaggle Paper Track writeup.

- **Dataset**: <https://www.kaggle.com/datasets/takumuhata/swe-pallets-x>
  (**476 verified tasks** across 8 repositories — ~3.7× the competition's
  129-task public training set)
- **Verification demo notebook**:
  <https://www.kaggle.com/code/takumuhata/swe-pallets-x-dataset-tour-verification-demo>
- **Paper**: [`paper.md`](paper.md)

## Dataset composition

| Repo | Tasks |
|---|---|
| pytest-dev/pytest | 132 |
| sympy/sympy | 129 |
| pallets/click | 109 |
| pallets/werkzeug | 43 |
| pallets/flask | 29 |
| pallets/jinja2 | 29 |
| pallets/markupsafe | 4 |
| pallets/itsdangerous | 1 |
| **Total** | **476** |

Each task ships with: `problem_statement` (linked GitHub issue/PR or commit
subject), `patch`, `test_patch`, frozen single-commit snapshot tarball,
AST call/dependency graph (NetworkX node-link JSON), and 256-dim node
embeddings (TF-IDF → TruncatedSVD).

## Pipeline

Checkpointed stages, each emitting JSONL and independently resumable:

```
mine.py      # git log scan: commits touching source+test files
enrich.py    # GitHub API: fix commit -> PR -> linked issue
diffs.py     # patch / test_patch extraction
verify.py    # two-phase F2P/P2P execution check in era-pinned per-instance venvs
build_dataset.py  # freeze snapshots, write tasks.jsonl
graphs.py / embeddings.py / parallel_graphs.py  # AST graphs + 256-dim embeddings
check_dataset.py  # integrity check (0 errors / 0 warnings required)
agent_eval.py     # minimal SWE-agent harness for small local models (Ollama)
analyze_eval.py   # resolve rates + failure taxonomy
```

Verification is era-consistent (Python 3.10–3.13 by commit date, dependencies
pinned to pre-commit PyPI releases, era-pinned pytest) and handles self-hosting
repos (verifying pytest uses the repo itself as the runner).

## Results snapshot

| Model | Resolved (181-task initial release) |
|---|---|
| Qwen2.5-Coder-7B | 4/181 (2.2%) |
| Gemma3-4B | 1/181 (0.6%) |
| Qwen2.5-Coder-7B (75-task expansion sample) | 1/75 (1.3%) |

**Utility probes** (84 held-out tasks): statement→graph-node BM25 file
localization reaches **69% recall@5 / 80% @10** (+4–5 pts from one-hop graph
expansion). A QLoRA patch-SFT probe on Qwen2.5-Coder-1.5B is an honest
negative — SFT improves diff syntax but *reduces* context-line fidelity vs
era-frozen snapshots (0.61 vs 0.72 verbatim context match), a failure mode
the dataset makes measurable. See `paper.md` §5.2.

See `paper.md` for the full analysis (per-repo verification yields, failure
taxonomy, ±graph-tools ablation, harness-fidelity notes).

## License

Pipeline code: MIT. Dataset: CC-BY-SA-4.0 (source repos are BSD-3-Clause/MIT —
pallets projects, pytest, sympy).
