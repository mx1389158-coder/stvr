from __future__ import annotations

import argparse
import torch
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    return ap.parse_args()


def main():
    args = parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    merged = model.merge_and_unload()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    merged.save_pretrained(out)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, trust_remote_code=True)
    tok.save_pretrained(out)

    print("[ok] merged to", out)


if __name__ == "__main__":
    main()
