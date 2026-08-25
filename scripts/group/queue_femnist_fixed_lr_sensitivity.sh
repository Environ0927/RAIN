#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${RAIN_PYTHON:-/home/yuhang.li/miniconda3/envs/rain-artifact/bin/python}"
output_root="$root/outputs/convergence"
manifest="$output_root/sensitivity-femnist-fixedlr-manifest.json"
mkdir -p "$output_root"

"$python_bin" -c '
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
value = {
    "schema_version": 1,
    "dataset": "femnist",
    "role": "user-specified fixed-learning-rate sensitivity experiment",
    "suitable_for_primary_fair_comparison": False,
    "seeds": [1, 2],
    "learning_rates": {
        "fedavg": 0.02,
        "rain": 0.0002,
        "signsgd": 0.00005,
        "flod": 0.00005,
    },
    "selection_note": "Values were specified after validation results and were not validation-optimal for every method.",
}
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "$manifest"

echo "[$(date -Is)] starting two-seed fixed-LR FEMNIST sensitivity runs"
"$root/scripts/group/run_femnist_fixed_lr_sensitivity.sh" primary 0 &
primary_pid=$!
"$root/scripts/group/run_femnist_fixed_lr_sensitivity.sh" baselines 1 &
baseline_pid=$!
wait "$primary_pid"
wait "$baseline_pid"

"$python_bin" -m rain.cli.summarize \
    --inputs "$output_root"/sensitivity-femnist-fixedlr-*-seed*/rounds.jsonl \
    --output "$output_root/sensitivity-femnist-fixedlr-summary.json"

plot_inputs=()
for method in rain signsgd fedavg flod; do
    case "$method" in
        rain) label="RAIN" ;;
        signsgd) label="SignSGD" ;;
        fedavg) label="FedAvg" ;;
        flod) label="FLOD" ;;
    esac
    for seed in 1 2; do
        plot_inputs+=("$label=$output_root/sensitivity-femnist-fixedlr-${method}-seed${seed}/rounds.jsonl")
    done
done
"$python_bin" -m rain.cli.plot_convergence \
    --inputs "${plot_inputs[@]}" \
    --metric validation_accuracy \
    --output "$output_root/sensitivity-femnist-fixedlr-validation.png" \
    --title "FEMNIST fixed-LR sensitivity (mean +/- std; 2 seeds)"

echo "[$(date -Is)] fixed-LR FEMNIST sensitivity suite complete"
