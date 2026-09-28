# Raw evaluation artifacts

Episode-level results behind paper §5. One JSON row per task episode:
`steps`, `tool_hist` (per-tool call counts), `has_patch`, `diff`, `passed`,
`error` (setup/agent failures; errored rows are retried by the harness).

| File | Arm | Tasks |
|---|---|---|
| qwen_nograph_full.jsonl | Qwen2.5-Coder-7B | all 476 |
| qwen_graph_full+A+B+rest.jsonl | Qwen + graph tools | 181 (split shards, concat = arm) |
| qwen_graph_exp31.jsonl | Qwen + graph tools | 31-task extension subset |
| gemma3_nograph_full.jsonl | Gemma3-4B | 181 |
| gemma3_exp40.jsonl | Gemma3-4B | 31-task extension subset |
| loc476_dump.jsonl | localization probe | all 476, per-instance first-hit rank |

Reproduce: `pipeline/agent_eval.py <task_dir> <ollama-model> --out out.jsonl`
and `pipeline/eval_localize_dump.py <task_dir> <input> <out>`.
