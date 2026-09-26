#!/usr/bin/env python3
"""QLoRA SFT on SWE-Pallets-X records (issue+oracle files -> diff patch).
Fits a 1.5B model on a 6GB card (4-bit NF4, grad ckpt, seq<=2048).

Usage: train_sft.py <sft_train.jsonl> <out_dir> [model]
"""
import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def main(train_path, out_dir, model_id=MODEL, max_len=2048, epochs=3):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto",
        attn_implementation="eager")
    m = prepare_model_for_kbit_training(m)
    m = get_peft_model(m, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    m.print_trainable_parameters()

    rows = [json.loads(l) for l in open(train_path)]

    def toks(r):
        msgs = [{"role": "user", "content": r["user"]},
                {"role": "assistant", "content": r["assistant"]}]
        text = tok.apply_chat_template(msgs, tokenize=False)
        enc = tok(text, truncation=True, max_length=max_len)
        # mask prompt tokens: train only on the assistant span
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": r["user"]}],
            tokenize=False, add_generation_prompt=True)
        pl = len(tok(prompt, truncation=True, max_length=max_len)
                 ["input_ids"])
        labels = [-100] * pl + enc["input_ids"][pl:]
        enc["labels"] = labels
        return enc

    ds = Dataset.from_list(rows).map(
        toks, remove_columns=rows[0].keys())

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad] * n)
            labels.append(b["labels"] + [-100] * n)
            attn.append(b["attention_mask"] + [0] * n)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}

    tr = Trainer(
        model=m, args=TrainingArguments(
            output_dir=out_dir, per_device_train_batch_size=1,
            gradient_accumulation_steps=8, num_train_epochs=epochs,
            learning_rate=2e-4, lr_scheduler_type="cosine",
            logging_steps=10, save_strategy="epoch",
            bf16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to=[], seed=0),
        train_dataset=ds, data_collator=collate)
    tr.train()
    m.save_pretrained(os.path.join(out_dir, "lora"))
    tok.save_pretrained(os.path.join(out_dir, "lora"))
    print("saved:", out_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         *(sys.argv[3:] or []))
