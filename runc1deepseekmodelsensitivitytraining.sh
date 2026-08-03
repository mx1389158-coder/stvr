#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${UTPLM_PROJECT_ROOT:-/root/autodl-tmp/utplm}"
PY="$PROJECT_ROOT/conda_envs/utplm-eval/bin/python"
BASE="$PROJECT_ROOT/models/base/deepseek-coder-6.7b-instruct"
FORMAL_DIR="$PROJECT_ROOT/data/processed/c1formalv2c"
CONTROL_DIR="$PROJECT_ROOT/data/processed/c1controlsv1"
OUT_DIR="$PROJECT_ROOT/models/adapters/c1_model_sensitivity_deepseek_v1"
SEEDS="${SEEDS:-11 22 33}"

mkdir -p "$OUT_DIR"

for SEED in $SEEDS; do
  "$PY" "$PROJECT_ROOT/stvr/scripts/train/traindpoqwenloraplain.py" \
    --model_name_or_path "$BASE" \
    --train_jsonl "$FORMAL_DIR/hh.jsonl" \
    --output_dir "$OUT_DIR/HH_DPO_seed${SEED}" \
    --seed "$SEED" \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 2048 \
    --max_prompt_length 1024 \
    --beta 0.1

done

for SEED in $SEEDS; do
  "$PY" "$PROJECT_ROOT/stvr/scripts/train/trainsftqwenloraplain.py" \
    --model_name_or_path "$BASE" \
    --train_jsonl "$CONTROL_DIR/chosenonlysfthh.jsonl" \
    --output_dir "$OUT_DIR/chosen_only_sft_HH_seed${SEED}" \
    --seed "$SEED" \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 2048

done

for CONTROL in label_shuffled_dpo_HH surface_matched_random_negative_dpo_HH; do
  for SEED in $SEEDS; do
    "$PY" "$PROJECT_ROOT/stvr/scripts/train/traindpoqwenloraplain.py" \
      --model_name_or_path "$BASE" \
      --train_jsonl "$CONTROL_DIR/${CONTROL}.jsonl" \
      --output_dir "$OUT_DIR/${CONTROL}_seed${SEED}" \
      --seed "$SEED" \
      --num_train_epochs 3 \
      --learning_rate 1e-5 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --max_length 2048 \
      --max_prompt_length 1024 \
      --beta 0.1
  done
done
