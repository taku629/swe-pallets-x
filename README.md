# SWE-Pallets-X

**Verified extension benchmark for small-model software-engineering agents**, in the
[Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent)
dataset format. Companion code for the Kaggle Paper Track writeup.

- **Dataset**: <https://www.kaggle.com/datasets/takumuhata/swe-pallets-x>
  (181 verified tasks → expanding to 3 additional ecosystems: pytest, sympy, pallets tools)
- **Verification demo notebook**:
  <https://www.kaggle.com/code/takumuhata/swe-pallets-x-dataset-tour-verification-demo>
- **Paper**: [`paper.md`](paper.md)

## Pipeline

Checkpointed stages, each emitting JSONL and independently resumable:

```
mine.py      # git log scan: commits touching source+test files
enrich.py    # GitHub API: fix commit -> PR -> linked issue
diffs.py     # patch / test_patch extraction
verify.py    # two-phase F2P/P2P execution check in era-pinned per-instance venvs
build_dataset.py  # freeze snapshots, write tasks.jsonl
graphs.py / embeddings.py / gen_graphs_all.py  # AST graphs + 256-dim embeddings
check_dataset.py  # integrity check (0 errors / 0 warnings required)
agent_eval.py     # minimal SWE-agent harness for small local models (Ollama)
analyze_eval.py   # resolve rates + failure taxonomy
```

## Results snapshot

| Model | Resolved (181 tasks) |
|---|---|
| Qwen2.5-Coder-7B | 4/181 (2.2%) |
| Gemma3-4B | 1/181 (0.6%) |

See `paper.md` for the full analysis (verification yields, failure taxonomy,
±graph-tools ablation, harness-fidelity notes).

## License

Pipeline code: MIT. Dataset: CC-BY-SA-4.0 (source repos are BSD-3-Clause/MIT —
pallets projects, pytest, sympy).
