#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${UTPLM_PROJECT_ROOT:-/root/autodl-tmp/utplm}"
PY="$PROJECT_ROOT/conda_envs/utplm-eval/bin/python"
BASE="$PROJECT_ROOT/models/base/qwen2.5-7b-instruct"
ADAPTER_ROOT="$PROJECT_ROOT/models/adapters/c1controlsv1"
INPUT_CSV="$PROJECT_ROOT/data/splits/v1cleaned/maintestv1.csv"

export UTPLM_PROJECT_ROOT="$PROJECT_ROOT"

for CONTROL in chosen_only_sft_HH label_shuffled_dpo_HH surface_matched_random_negative_dpo_HH; do
  for SEED in 11 22 33; do
    RUN_ID="c1ctrl_${CONTROL}_seed${SEED}_main"
    "$PY" "$PROJECT_ROOT/stvr/scripts/generation/generatecandidatesqwenfinal.py"       --input_csv "$INPUT_CSV"       --run_id "$RUN_ID"       --model_name_or_path "$BASE"       --adapter_dir "$ADAPTER_ROOT/${CONTROL}_seed${SEED}"       --preset_names greedy       --prompt_variant_names balanced       --max_new_tokens 768       --overwrite

    export UTPLM_RUN_ID="$RUN_ID"
    "$PY" "$PROJECT_ROOT/stvr/scripts/eval/runexecution.py"
    "$PY" "$PROJECT_ROOT/stvr/scripts/eval/collectcoverage.py"
    "$PY" "$PROJECT_ROOT/stvr/scripts/eval/runmutation.py"
    "$PY" "$PROJECT_ROOT/stvr/scripts/eval/extractassertionfeatures.py"
    "$PY" "$PROJECT_ROOT/stvr/scripts/eval/buildfailuretaxonomy.py"
    "$PY" "$PROJECT_ROOT/stvr/scripts/analysis/summarizeevalrun.py"       --project_root "$PROJECT_ROOT"       --run_id "$RUN_ID"       --group "$CONTROL"       --seed "$SEED"       --split_name main_test
  done
done
