from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path('/root/autodl-tmp/utplm')
EVAL_ROOT = PROJECT_ROOT / 'outputs' / 'evaluations' / 'b1'
OUT_DIR = PROJECT_ROOT / 'outputs' / 'evaluations' / 'c1_model_sensitivity_deepseek_v1'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS = [
    ('DeepSeek-Coder-6.7B-Instruct', 'HH DPO', 'deepseek_HH_DPO'),
    ('DeepSeek-Coder-6.7B-Instruct', 'chosen-only SFT', 'deepseek_chosen_only_sft_HH'),
    ('DeepSeek-Coder-6.7B-Instruct', 'label-shuffled DPO', 'deepseek_label_shuffled_dpo_HH'),
    ('DeepSeek-Coder-6.7B-Instruct', 'surface-matched random-negative DPO', 'deepseek_surface_matched_random_negative_dpo_HH'),
]

MAIN_QWEN = PROJECT_ROOT / 'outputs' / 'evaluations' / 'c1controlsv1' / 'c1formaluppercasepluscontrolsattemptlevelaggregate.csv'

METRICS = [
    'execution_pass_rate',
    'valid_test_rate',
    'valid_tests_only_mutation_score_mean',
    'execution_pass_x_mutation_score_mean',
    'valid_tests_only_boundary_coverage_mean',
    'execution_pass_x_boundary_coverage_mean',
    'failure_log_count_mean',
]


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sd(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if len(xs) <= 1:
        return 0.0 if xs else None
    return statistics.stdev(xs)


def f(value: Any) -> Optional[float]:
    try:
        if value is None or value == '':
            return None
        x = float(value)
        return None if math.isnan(x) else x
    except Exception:
        return None


def read_deepseek_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model, setting, group in SETTINGS:
        summaries = []
        for seed in (11, 22, 33):
            run_id = f"c1ds_{group.replace('deepseek_', '')}_seed{seed}_main"
            obj = load_json(EVAL_ROOT / run_id / 'summary.json')
            if obj is not None:
                summaries.append(obj)
        row: Dict[str, Any] = {'model': model, 'setting': setting, 'group': group, 'n_runs': len(summaries)}
        for metric in METRICS:
            vals = [f(s.get(metric)) for s in summaries]
            row[f'{metric}_mean'] = mean(vals)
            row[f'{metric}_sd'] = sd(vals)
        rows.append(row)
    return rows


def read_qwen_rows() -> List[Dict[str, Any]]:
    if not MAIN_QWEN.exists():
        return []
    mapping = {
        'HH': 'HH DPO',
        'chosen-only SFT': 'chosen-only SFT',
        'label-shuffled DPO': 'label-shuffled DPO',
        'surface-matched random-negative DPO': 'surface-matched random-negative DPO',
    }
    out = []
    with MAIN_QWEN.open(newline='', encoding='utf-8') as fp:
        for r in csv.DictReader(fp):
            if r['group'] not in mapping:
                continue
            out.append({
                'model': 'Qwen2.5-7B-Instruct',
                'setting': mapping[r['group']],
                'group': r['group'],
                'n_runs': int(float(r.get('n_runs') or 0)),
                'execution_pass_rate_mean': f(r.get('execution_pass_rate_mean')),
                'execution_pass_rate_sd': f(r.get('execution_pass_rate_sd')),
                'valid_test_rate_mean': f(r.get('valid_test_rate_mean')),
                'valid_test_rate_sd': f(r.get('valid_test_rate_sd')),
                'valid_tests_only_mutation_score_mean_mean': f(r.get('valid_tests_only_mutation_score_mean')),
                'valid_tests_only_mutation_score_mean_sd': f(r.get('valid_tests_only_mutation_score_sd')),
                'execution_pass_x_mutation_score_mean_mean': f(r.get('execution_pass_x_mutation_score_mean')),
                'execution_pass_x_mutation_score_mean_sd': f(r.get('execution_pass_x_mutation_score_sd')),
                'valid_tests_only_boundary_coverage_mean_mean': f(r.get('valid_tests_only_boundary_coverage_mean')),
                'valid_tests_only_boundary_coverage_mean_sd': f(r.get('valid_tests_only_boundary_coverage_sd')),
                'execution_pass_x_boundary_coverage_mean_mean': f(r.get('execution_pass_x_boundary_coverage_mean')),
                'execution_pass_x_boundary_coverage_mean_sd': f(r.get('execution_pass_x_boundary_coverage_sd')),
                'failure_log_count_mean_mean': f(r.get('failure_log_count_mean')),
                'failure_log_count_mean_sd': f(r.get('failure_log_count_sd')),
            })
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = ['model', 'setting', 'group', 'n_runs']
    for metric in METRICS:
        fields += [f'{metric}_mean', f'{metric}_sd']
    # Qwen rows from candidate-level aggregate use doubled names for metrics ending in _mean.
    normalized = []
    for r in rows:
        nr = {k: r.get(k) for k in fields}
        for metric in METRICS:
            if nr.get(f'{metric}_mean') is None:
                nr[f'{metric}_mean'] = r.get(f'{metric}_mean_mean')
            if nr.get(f'{metric}_sd') is None:
                nr[f'{metric}_sd'] = r.get(f'{metric}_mean_sd')
        normalized.append(nr)
    with path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)


def main() -> None:
    deepseek_rows = read_deepseek_rows()
    qwen_rows = read_qwen_rows()
    write_csv(OUT_DIR / 'modellevelsensitivitydeepseekonly.csv', deepseek_rows)
    write_csv(OUT_DIR / 'modellevelsensitivityqwenplusdeepseek.csv', qwen_rows + deepseek_rows)
    manifest = {
        'script': str(Path(__file__)),
        'eval_root': str(EVAL_ROOT),
        'output_dir': str(OUT_DIR),
        'settings': SETTINGS,
        'metrics': METRICS,
        'note': 'Model-level sensitivity check, not a model comparison. DeepSeek rows appear as n_runs become available.',
    }
    (OUT_DIR / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
