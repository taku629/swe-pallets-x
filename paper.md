# SWE-Pallets-X: A Verified Extension Benchmark for Small-Model Software-Engineering Agents

*An open replication of the Gemma 4 Developer Agent task-curation pipeline, plus a 476-task verified extension dataset spanning eight repositories.*

## Abstract

Small local models remain weak at multi-turn software engineering, and progress is bottlenecked by evaluation and training resources as much as by models themselves. We replicate the task-curation pipeline described by the Gemma 4 Developer Agent Competition and generalize it into a standalone, reproducible open-source pipeline that converts public GitHub history into verified SWE-bench-style task instances. Applied to eight repositories disjoint from the competition training set — Click, Flask, Werkzeug, Jinja2, ItsDangerous, MarkupSafe, pytest, and SymPy — it yields **476 verified tasks** packaged in competition-native format: `tasks.jsonl`, frozen git snapshots, AST call/dependency graphs, and 256-dim node embeddings, released as a public Kaggle dataset together with the full pipeline source. We report curation statistics, per-repository verification yields, failure-mode analysis, and baseline small-model agent evaluations on the new tasks.

## 1. Introduction

The Gemma 4 Developer Agent Competition evaluates agents on real GitHub bug fixes using a curated dataset: 129 public training tasks built from commit+test co-change mining, fail-to-pass/pass-to-pass (F2P/P2P) execution verification, AST code graphs, and node embeddings; the hidden test set (~120 tasks) was curated from private repositories by the same pipeline. The organizers document the recipe but not its implementation.

Independently reproducing this pipeline matters for three reasons:

1. **Training data.** Post-training methods (trajectory SFT, RL on execution rewards) need far more than 129 instances.
2. **Generalization measurement.** A public, structurally identical extension set enables held-out evaluation without touching the private test set.
3. **Reusability.** A documented, working pipeline lets the community extend the benchmark to new repositories, languages, and task types.

We release **(a)** an open end-to-end implementation of the curation pipeline, **(b)** *SWE-Pallets-X*, a 476-task verified extension dataset in competition-native format, and **(c)** baseline agent resolve rates and failure-mode diagnostics on the new tasks.

## 2. Related work

- **SWE-bench** (Jimenez et al., 2023): real GitHub issues + reference patches + execution verification. Our pipeline mirrors its F2P/P2P validation loop.
- **SWE-smith / SWE-Gym**: scale task construction via environment synthesis; small models post-trained on trajectories reach ~20–42% on SWE-bench Verified (e.g. SWE-Protégé), showing small-model agents are trainable *when data exists* — the gap this resource targets.
- **Repository-level code retrieval**: CodexGraph, RANGER, GraphCodeAgent show graph-structured retrieval helps repo-scale tasks — the same toolset the competition exposes (`get_code_neighbors`, `search_similar_code`, `get_code_subgraph`).

## 3. The pipeline

Five checkpointed stages; each emits JSONL and is independently resumable.

**3.1 Mining.** `git log` scan for non-merge commits touching *both* non-test `.py` and test files (`test_*.py`, `*_test.py`, `tests/`). Candidate = (fix_commit, base_commit=parent, src/test file split). Yield: **4,782 raw candidates** across 8 repos (2021–present); large repos (pytest 772, SymPy 3,332) were uniformly subsampled to 250 each to bound verification cost.

**3.2 Issue linking.** Each fix commit maps to its PR via the GitHub API; `fixes/closes #N` references resolve to the original issue, which becomes `problem_statement` (PR body as fallback). Real issue-derived statements (median ~865 chars) make tasks substantially more realistic than commit-subject stubs.

**3.3 Patch extraction.** `patch` = diff of non-test `.py`; `test_patch` = diff of test files. De-noising caps (≤200 changed lines, ≤6 source files) remove large-scale changes, consistent with the organizers' LSC filtering.

**3.4 Two-phase execution verification.** The decisive stage, where most curation effort went:

- *Era-consistent environments.* Per-instance venv with era-matched interpreter (Python 3.10–3.13 by commit year) and era-pinned dependencies — each requirement constrained to the latest PyPI release *before* the fix-commit date, **including pytest itself**. Three failure classes surfaced only at scale: (i) modern pytest breaks historical conftests (e.g. `monkeypatch.notset` removal), so the test runner itself must be era-pinned; (ii) self-hosting repositories — verifying the pytest repo itself means the repo *is* the test runner, so installing external pytest creates an unresolvable constraint; (iii) locally-declared names absent from PyPI (e.g. SymPy's `isympy` script) poison dependency resolution and must be filtered to packages with a resolvable pre-cutoff release. Era pinning must also anchor to the *fix-commit* date, not the issue-creation date: multi-year-old issues fixed recently otherwise pin dependencies to an era before the code existed.
- *Fail-to-Pass:* at `base_commit` with only `test_patch` applied, touched tests must fail.
- *Pass-to-Pass:* additionally applying `patch`, touched tests must pass.

| Repo | Attempted | Verified | Yield |
|---|---|---|---|
| Click | 154 | 110 | 71% |
| SymPy | 250 | 139 | 56% |
| pytest | 250 | 137 | 55% |
| Flask | 74 | 34 | 46% |
| Jinja2 | 74 | 29 | 39% |
| MarkupSafe | 12 | 4 | 33% |
| Werkzeug | 184 | 47 | 26% |
| ItsDangerous | 6 | 1 | 17% |
| **Total** | **1,004** | **501** | **50%** |

Of verification failures: P2P failures dominate (missing test deps, era-incompatible APIs, or genuinely incomplete fixes), no-F2P-signal next (new tests pass even without the fix — tasks that would be trivially solvable or mislabeled), then infrastructure remnants. Yield varies 3.9× across repos (17–71%), driven mainly by test-suite hygiene and dependency complexity — evidence that the recipe transfers but each new repo needs modest dependency-discovery tuning.

**3.5 Packaging.** Each verified instance ships as `snapshots/<id>.tgz` (repo frozen at `base_commit` as a fresh single-commit git repo — `git log/diff` safe, no forward-history leakage), `graphs/<id>.json` (NetworkX node-link JSON; ~1.5–7k symbol nodes and ~4.6–35k `calls`/`contains` edges per snapshot, generated by a suffix-indexed AST resolver), and `embeddings/<id>.npz` (256-dim float32 per node; TF-IDF → TruncatedSVD-256 → L2-norm — a fully disclosed CPU-reproducible encoder rather than an opaque proprietary one).

## 4. The dataset

| Repo | Tasks | Share |
|---|---|---|
| pytest-dev/pytest | 132 | 28% |
| sympy/sympy | 129 | 27% |
| pallets/click | 109 | 23% |
| pallets/werkzeug | 43 | 9% |
| pallets/flask | 29 | 6% |
| pallets/jinja2 | 29 | 6% |
| pallets/markupsafe | 4 | 1% |
| pallets/itsdangerous | 1 | <1% |
| **Total** | **476** | |

At 476 instances the set is ~3.7× the competition's 129-task public training set, and spans repositories ranging from small utility libraries to a 7,000-symbol scientific codebase — a difficulty axis absent from the original release.

Median patch: ~23 changed lines. Median problem statement: ~865 chars — sourced from the linked GitHub issue or PR body for ~96% of tasks, and from the commit subject otherwise. Dates span 2021–2026. The set is disjoint from the competition's public training repositories (fastapi, rich, requests, httpx, among others). Overlap with the undisclosed hidden test set cannot be ruled out a priori; every instance is nevertheless independently derived and execution-verified, so the resource remains valid as training data and as a public held-out benchmark. The dataset is published at <https://www.kaggle.com/datasets/takumuhata/swe-pallets-x>, with pipeline source at <https://github.com/taku629/swe-pallets-x>.

## 5. Baseline evaluation

We run small open models through a minimal re-implementation of the competition toolset (`run_command`, `read_file`, `edit_file`, `write_file`, `submit_patch`, ± graph tools) inside each frozen snapshot, capture `git diff HEAD`, and score with the task's own test suite — the same PASS/FAIL criterion as the competition. A public companion notebook (<https://www.kaggle.com/code/takumuhata/swe-pallets-x-dataset-tour-verification-demo>) independently re-verifies packaged instances on Kaggle infrastructure (schema, companion assets, clean `git apply` of `test_patch`).

Across the 181-task initial release (all Click/Werkzeug/Flask tasks), Qwen2.5-Coder-7B resolves **4/181 (2.2%)** and Gemma3-4B resolves **1/181 (0.6%)** — the same order as the ~5% early leaderboard of the parent competition, suggesting comparable difficulty. On a 75-task stratified sample of the v2 expansion repos (pytest, SymPy, Jinja2, MarkupSafe, ItsDangerous), Qwen resolves **1/75 (1.3%)** — the extension is at least as hard as the original slice, not diluted by easy instances. All solved patches are small (2–38 changed lines; dataset median 23). The set exhibits a difficulty gradient: tasks with ≤10-line patches resolve at ~5% (Qwen) vs ~0% for patches >30 lines, suggesting headroom for both easy-entry and hard exploration in downstream training.

| Model | Resolved | Mean steps | Dominant failure mode |
|---|---|---|---|
| Qwen2.5-Coder-7B | 4/181 (2.2%) | 12.3 | patch-failed-tests (52) |
| Gemma3-4B | 1/181 (0.6%) | 3.5 | empty-submit (148/181, 82%) |

**Failure taxonomy** (Qwen, from episode transcripts): patch-failed-tests 52, edits-failed-to-apply 50 (`edit_file` `old_string` mismatches), no-edit-attempted 40, empty-submit 35. The two small models fail for *different* reasons — Qwen engages the task (523 `edit_file` calls) but fumbles patch mechanics, while Gemma3 mostly never engages (mean 3.5 steps, 15 `read_file` calls across 181 episodes) — which is precisely the kind of per-model diagnostic a public extension set enables.

**±Graph-tools ablation.** On a 36-task stratified sample, exposing `get_code_neighbors`, `search_similar_code`, and `get_code_subgraph` yields an identical 2/36 with the same solved instances — but the reason is behavioral, not neutral: the model invoked graph tools only 8 times in 36 episodes (all `search_similar_code`; zero graph-traversal calls). At this scale the binding constraint is basic file navigation and edit application, upstream of retrieval. Whether graph tools help *stronger* small models is exactly the kind of question the dataset enables.

**Harness-fidelity caution.** Our first harness iteration silently mis-evaluated — file tools rejected the `/workspace`-prefixed paths the prompt instructed, and `run_command` executed outside the task environment, so all `edit_file` calls failed and reproduction attempts hit import errors. This produced a deceptively low 2.8% resolve rate that reflected harness defects, not model capability — a reminder that small-model agent benchmarks are extremely sensitive to tool-interface fidelity. Verification infrastructure showed the same fragility at curation time: three of our four pipeline-level fixes (era-pin anchoring, self-hosting exclusion, unresolvable-dependency filtering) each recovered tens of percentage points of yield.

## 6. Limitations

- Public-repo mining inherits maintainer conventions; issue↔PR linkage quality varies, and ~4% of tasks carry only commit-subject problem statements.
- Graphs use a faithful-schema AST extractor (not the organizers' implementation); embeddings use a disclosed open encoder. Call-edge resolution is heuristic suffix-matching.
- Verification runs touched test files per task, not full-suite regression.
- Baselines were measured on the initial 181-task release; resolve rates on the pytest/SymPy extension may differ (both repos have heavier per-task environments).
- Eight repos is a start, not coverage — the pipeline is the deliverable and scales by adding repos (each new repo mostly needs dependency-group discovery tuning).

## 7. Conclusion

We release an open replication of the competition's task-curation pipeline and SWE-Pallets-X, 476 verified tasks in competition-native format — immediately usable as extra training data, a public held-out test bed, and a template for further extension. Per-repository verification yields (17–71%) and failure-mode analysis show the recipe transfers across repos with modest per-repo dependency tuning, and our baseline measurements confirm the tasks sit at the right difficulty frontier for today's small-model agents.

## References

Jimenez et al., *SWE-bench*, 2023 · Yang et al., *SWE-smith*, 2025 · *SWE-Protégé*, 2026 · Liu et al., *CodexGraph* (NAACL), 2025 · *RANGER*, 2025 · Kaggle competition data-description, rules and evaluation pages, 2026.
