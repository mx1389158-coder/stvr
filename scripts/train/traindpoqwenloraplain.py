from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Dict, List

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
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_prompt_length", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=0.1)
    return ap.parse_args()


def build_example(tokenizer, prompt: str, response: str, max_length: int, max_prompt_length: int):
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_length,
    )["input_ids"]

    max_resp_len = max(8, max_length - len(prompt_ids) - 1)
    response_ids = tokenizer(
        response,
        add_special_tokens=False,
        truncation=True,
        max_length=max_resp_len,
    )["input_ids"]

    eos_id = tokenizer.eos_token_id
    input_ids = prompt_ids + response_ids + ([eos_id] if eos_id is not None else [])
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + response_ids + ([eos_id] if eos_id is not None else [])

    return {
        "input_ids": input_ids[:max_length],
        "attention_mask": attention_mask[:max_length],
        "labels": labels[:max_length],
    }


@dataclass
class DPOCollator:
    tokenizer: Any

    def _pad_labels(self, features: List[Dict[str, Any]], key: str, max_len: int):
        rows = []
        for f in features:
            x = list(f[key])
            x = x + [-100] * (max_len - len(x))
            rows.append(x)
        return torch.tensor(rows, dtype=torch.long)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen_inputs = [
            {"input_ids": f["chosen_input_ids"], "attention_mask": f["chosen_attention_mask"]}
            for f in features
        ]
        rejected_inputs = [
            {"input_ids": f["rejected_input_ids"], "attention_mask": f["rejected_attention_mask"]}
            for f in features
        ]

        chosen_batch = self.tokenizer.pad(chosen_inputs, padding=True, return_tensors="pt")
        rejected_batch = self.tokenizer.pad(rejected_inputs, padding=True, return_tensors="pt")

        chosen_labels = self._pad_labels(features, "chosen_labels", chosen_batch["input_ids"].shape[1])
        rejected_labels = self._pad_labels(features, "rejected_labels", rejected_batch["input_ids"].shape[1])

        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
            "rejected_labels": rejected_labels,
        }


def sequence_logp(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~mask, 0)
    token_logps = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    seq_logps = (token_logps * mask).sum(dim=-1)
    return seq_logps


class DPOTrainerMin(Trainer):
    def __init__(self, *args, ref_model=None, beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        chosen_out = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
            use_cache=False,
        )
        rejected_out = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
            use_cache=False,
        )

        pi_c = sequence_logp(chosen_out.logits, inputs["chosen_labels"])
        pi_r = sequence_logp(rejected_out.logits, inputs["rejected_labels"])

        with torch.no_grad():
            ref_c_out = self.ref_model(
                input_ids=inputs["chosen_input_ids"],
                attention_mask=inputs["chosen_attention_mask"],
                use_cache=False,
            )
            ref_r_out = self.ref_model(
                input_ids=inputs["rejected_input_ids"],
                attention_mask=inputs["rejected_attention_mask"],
                use_cache=False,
            )

            ref_c = sequence_logp(ref_c_out.logits, inputs["chosen_labels"])
            ref_r = sequence_logp(ref_r_out.logits, inputs["rejected_labels"])

        logits = (pi_c - pi_r) - (ref_c - ref_r)
        loss = -F.logsigmoid(self.beta * logits).mean()

        if return_outputs:
            return loss, {"dpo_logits": logits.detach()}
        return loss


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

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.config.use_cache = False
    ref_model.eval()

    ds = load_dataset("json", data_files=args.train_jsonl, split="train")

    def preprocess(ex):
        chosen = build_example(tokenizer, ex["prompt"], ex["chosen"], args.max_length, args.max_prompt_length)
        rejected = build_example(tokenizer, ex["prompt"], ex["rejected"], args.max_length, args.max_prompt_length)
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels": rejected["labels"],
        }

    ds = ds.map(preprocess, remove_columns=ds.column_names)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        seed=args.seed,
        fp16=not bf16_ok,
        bf16=bf16_ok,
        remove_unused_columns=False,
    )

    trainer = DPOTrainerMin(
        model=model,
        ref_model=ref_model,
        beta=args.beta,
        args=train_args,
        train_dataset=ds,
        data_collator=DPOCollator(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[ok] saved", args.output_dir)


if __name__ == "__main__":
    main()
