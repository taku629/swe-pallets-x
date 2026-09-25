# SWE-Pallets-X: A Verified Extension Benchmark for Small-Model Software-Engineering Agents

*An open replication of the Gemma 4 Developer Agent task-curation pipeline, plus a 181-task verified extension dataset.*

## Abstract

Small local models remain weak at multi-turn software engineering, and progress is bottlenecked by evaluation and training resources as much as by models themselves. We replicate the task-curation pipeline described by the Gemma 4 Developer Agent Competition and generalize it into a standalone, reproducible open-source pipeline that converts public GitHub history into verified SWE-bench-style task instances. Applied to three repositories disjoint from the competition training set (Flask, Werkzeug, Click), it yields **181 verified tasks** packaged in competition-native format — `tasks.jsonl`, frozen git snapshots, AST call/dependency graphs, and 256-dim node embeddings — released as a public Kaggle dataset together with the full pipeline source. We report curation statistics, verification yields, failure-mode analysis, and a baseline small-model agent evaluation on the new tasks.

## 1. Introduction

The Gemma 4 Developer Agent Competition evaluates agents on real GitHub bug fixes using a curated dataset: 129 public training tasks built from commit+test co-change mining, fail-to-pass/pass-to-pass (F2P/P2P) execution verification, AST code graphs, and node embeddings; the hidden test set (~120 tasks) was curated from private repositories by the same pipeline. The organizers document the recipe but not its implementation.

Independently reproducing this pipeline matters for three reasons:

1. **Training data.** Post-training methods (trajectory SFT, RL on execution rewards) need far more than 129 instances.
2. **Generalization measurement.** A public, structurally identical extension set enables held-out evaluation without touching the private test set.
3. **Reusability.** A documented, working pipeline lets the community extend the benchmark to new repositories, languages, and task types.

We release **(a)** an open end-to-end implementation of the curation pipeline and **(b)** *SWE-Pallets-X*, a 181-task verified extension dataset in competition-native format, plus **(c)** baseline agent resolve rates on the new tasks.

## 2. Related work

- **SWE-bench** (Jimenez et al., 2023): real GitHub issues + reference patches + execution verification. Our pipeline mirrors its F2P/P2P validation loop.
- **SWE-smith / SWE-Gym**: scale task construction via environment synthesis; small models post-trained on trajectories reach ~20–42% on SWE-bench Verified (e.g. SWE-Protégé), showing small-model agents are trainable *when data exists* — the gap this resource targets.
- **Repository-level code retrieval**: CodexGraph, RANGER, GraphCodeAgent show graph-structured retrieval helps repo-scale tasks — the same toolset the competition exposes (`get_code_neighbors`, `search_similar_code`, `get_code_subgraph`).

## 3. The pipeline

Five checkpointed stages; each emits JSONL and is independently resumable.

**3.1 Mining.** `git log` scan for non-merge commits touching *both* non-test `.py` and test files (`test_*.py`, `*_test.py`, `tests/`). Candidate = (fix_commit, base_commit=parent, src/test file split). Yield: **586 candidates** across 3 repos (2021–present).

**3.2 Issue linking.** Each fix commit maps to its PR via the GitHub API; `fixes/closes #N` references resolve to the original issue, which becomes `problem_statement` (PR body as fallback). Real issue-derived statements (median ~865 chars) make tasks substantially more realistic than commit-subject stubs.

**3.3 Patch extraction.** `patch` = diff of non-test `.py`; `test_patch` = diff of test files. De-noising caps (≤200 changed lines, ≤6 source files) remove large-scale changes, consistent with the organizers' LSC filtering.

**3.4 Two-phase execution verification.** The decisive stage, where most curation effort went:

- *Era-consistent environments.* Per-instance venv with era-matched interpreter (Python 3.10–3.13 by commit year) and era-pinned dependencies — each requirement constrained to the latest PyPI release *before* the commit date, **including pytest itself**. This proved essential: modern pytest breaks historical conftests (e.g. `monkeypatch.notset` removal), and current dependency majors break historical imports (e.g. `werkzeug.url_quote`). Test-only dependency groups (`dependency-groups.tests` in `pyproject.toml`) must also be discovered and pinned.
- *Fail-to-Pass:* at `base_commit` with only `test_patch` applied, touched tests must fail.
- *Pass-to-Pass:* additionally applying `patch`, touched tests must pass.

Of 586 unique mined candidates, 412 reached execution verification (the rest were dropped by enrichment gaps or de-noising caps) and **191 (46% of attempted; 33% of mined) survived**. Failure modes: P2P failures 69% of failures (missing test deps, era-incompatible APIs, or genuinely incomplete fixes), no-F2P-signal 28% (new tests pass even without the fix — tasks that would be trivially solvable or mislabeled), other 3%. **181 instances** passed packaging integrity checks.

**3.5 Packaging.** Each verified instance ships as `snapshots/<id>.tgz` (repo frozen at `base_commit` as a fresh single-commit git repo — `git log/diff` safe, no forward-history leakage), `graphs/<id>.json` (NetworkX node-link JSON; ~1.5–1.9k symbol nodes and ~4.6–5.7k `calls`/`contains` edges per snapshot), and `embeddings/<id>.npz` (256-dim float32 per node; char n-gram TF-IDF → TruncatedSVD-256 → L2-norm — a fully disclosed CPU-reproducible encoder rather than an opaque proprietary one).

## 4. The dataset

| Repo | Tasks | Share |
|---|---|---|
| Click | 109 | 60% |
| Werkzeug | 43 | 24% |
| Flask | 29 | 16% |
| **Total** | **181** | |

At 181 instances the set is ~40% larger than the competition's 129-task public training set.

Median patch: 23 changed lines (max 177). Median problem statement: 865 chars — sourced from the linked GitHub issue for 50% of tasks, from the PR body for a further 46%, and from the commit subject for the remaining 4%. Dates span 2022–2026. The set is disjoint from the competition's public training repositories (fastapi, rich, requests, httpx, among others). Overlap with the undisclosed hidden test set cannot be ruled out a priori; every instance is nevertheless independently derived and execution-verified, so the resource remains valid as training data and as a public held-out benchmark. The dataset is published at <https://www.kaggle.com/datasets/takumuhata/swe-pallets-x>.

## 5. Baseline evaluation

We run small open models through a minimal re-implementation of the competition toolset (`run_command`, `read_file`, `edit_file`, `write_file`, `submit_patch`, ± graph tools) inside each frozen snapshot, capture `git diff HEAD`, and score with the task's own test suite — the same PASS/FAIL criterion as the competition. A public companion notebook (<https://www.kaggle.com/code/takumuhata/swe-pallets-x-dataset-tour-verification-demo>) independently re-verifies all 181 packaged instances on Kaggle infrastructure (schema, companion assets, clean `git apply` of `test_patch`).

Across **all 181 tasks**, Qwen2.5-Coder-7B resolves **4/181 (2.2%)** and Gemma3-4B resolves **1/181 (0.6%)** — the same order as the ~5% early leaderboard of the parent competition, suggesting comparable difficulty. All solved patches are small (2–38 changed lines; dataset median 23).

| Model | Resolved | Mean steps | Dominant failure mode |
|---|---|---|---|
| Qwen2.5-Coder-7B | 4/181 (2.2%) | 12.3 | patch-failed-tests (52) |
| Gemma3-4B | 1/181 (0.6%) | 3.5 | empty-submit (148/181, 82%) |

**Failure taxonomy** (Qwen, from episode transcripts): patch-failed-tests 52, edits-failed-to-apply 50 (`edit_file` `old_string` mismatches), no-edit-attempted 40, empty-submit 35. The two small models fail for *different* reasons — Qwen engages the task (523 `edit_file` calls) but fumbles patch mechanics, while Gemma3 mostly never engages (mean 3.5 steps, 15 `read_file` calls across 181 episodes) — which is precisely the kind of per-model diagnostic a public extension set enables.

**±Graph-tools ablation.** On the 36-task stratified sample, exposing `get_code_neighbors`, `search_similar_code`, and `get_code_subgraph` yields an identical 2/36 with the same solved instances — but the reason is behavioral, not neutral: the model invoked graph tools only 8 times in 36 episodes (all `search_similar_code`; zero graph-traversal calls). At this scale the binding constraint is basic file navigation and edit application, upstream of retrieval. Whether graph tools help *stronger* small models is exactly the kind of question the dataset enables.

**Harness-fidelity caution.** Our first harness iteration silently mis-evaluated — file tools rejected the `/workspace`-prefixed paths the prompt instructed, and `run_command` executed outside the task environment, so all `edit_file` calls failed and reproduction attempts hit import errors. This produced a deceptively low 2.8% resolve rate that reflected harness defects, not model capability — a reminder that small-model agent benchmarks are extremely sensitive to tool-interface fidelity.

## 6. Limitations

- Public-repo mining inherits maintainer conventions; issue↔PR linkage quality varies.
- Graphs use a faithful-schema AST extractor (not the organizers' implementation); embeddings use a disclosed open encoder.
- Verification runs touched test files per task, not full-suite regression.
- Three repos is a start, not coverage — the pipeline is the deliverable and scales by adding repos (each new repo mostly needs dependency-group discovery tuning).

## 7. Conclusion

We release an open replication of the competition's task-curation pipeline and SWE-Pallets-X, 181 verified tasks in competition-native format — immediately usable as extra training data, a public held-out test bed, and a template for further extension. Verification yields and failure-mode analysis show the recipe transfers across repos with modest per-repo dependency tuning.

## References

Jimenez et al., *SWE-bench*, 2023 · Yang et al., *SWE-smith*, 2025 · *SWE-Protégé*, 2026 · Liu et al., *CodexGraph* (NAACL), 2025 · *RANGER*, 2025 · Kaggle competition data-description, rules and evaluation pages, 2026.
