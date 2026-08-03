from __future__ import annotations

import argparse
import torch
from dataclasses import dataclass
from typing import Dict, List, Any

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    return ap.parse_args()


@dataclass
class SFTCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]
        batch = self.tokenizer.pad(input_features, padding=True, return_tensors="pt")

        max_len = batch["input_ids"].shape[1]
        labels = []
        for f in features:
            x = list(f["labels"])
            x = x + [-100] * (max_len - len(x))
            labels.append(x)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def main():
    args = parse_args()

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)

    ds = load_dataset("json", data_files=args.train_jsonl, split="train")

    def tokenize_row(ex):
        text = ex["text"] if "text" in ex and ex["text"] else (ex["prompt"] + "\n\n" + ex["response"])
        enc = tokenizer(
            text,
            truncation=True,
            max_length=args.max_seq_length,
            add_special_tokens=True,
        )
        enc["labels"] = list(enc["input_ids"])
        return enc

    ds = ds.map(tokenize_row, remove_columns=ds.column_names)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        seed=args.seed,
        fp16=not bf16_ok,
        bf16=bf16_ok,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=SFTCollator(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[ok] saved", args.output_dir)


if __name__ == "__main__":
    main()
