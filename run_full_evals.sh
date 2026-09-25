#!/bin/bash
# Full-181-task baselines: gemma3 first (fast), then qwen2.5-coder.
cd /home/tk250127/kaggle_comp
python3 pipeline/agent_eval.py data/tasks gemma3:4b \
  --limit 181 --max-steps 20 \
  --out data/eval_run/gemma3_nograph_full.jsonl \
  >> data/eval_run/run_gemma3_full.log 2>&1
python3 pipeline/agent_eval.py data/tasks qwen2.5-coder:7b \
  --limit 181 --max-steps 20 \
  --out data/eval_run/qwen_nograph_full.jsonl \
  >> data/eval_run/run_qwen_full.log 2>&1
