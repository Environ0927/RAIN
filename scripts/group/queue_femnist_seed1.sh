#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${RAIN_PYTHON:-/home/yuhang.li/miniconda3/envs/rain-artifact/bin/python}"
output_root="$root/outputs/convergence"
calibration="$root/calibration/convergence_femnist_sigma0.json"
selection="$output_root/femnist-tuning-selection.json"
mkdir -p "$output_root"

if [[ ! -f "$calibration" ]]; then
    echo "[$(date -Is)] calibrating FEMNIST sigma-zero threshold on CUDA index 1"
    CUDA_VISIBLE_DEVICES=1 "$python_bin" -u -m rain.cli.calibrate_large \
        --config "$root/configs/convergence/femnist_rain.json" \
        --output "$calibration" --device cuda
fi

echo "[$(date -Is)] starting FEMNIST baseline pilots on CUDA index 1"
"$root/scripts/group/run_femnist_pilots_seed1.sh" baselines 1 &
baseline_pid=$!

echo "[$(date -Is)] waiting for the existing CIFAR-10 SignSGD track to release CUDA index 0"
while pgrep -f 'configs/convergence/cifar10_signsgd.json' >/dev/null; do
    sleep 30
done
echo "[$(date -Is)] starting FEMNIST primary pilots on CUDA index 0"
"$root/scripts/group/run_femnist_pilots_seed1.sh" primary 0 &
primary_pid=$!

wait "$baseline_pid"
wait "$primary_pid"

echo "[$(date -Is)] selecting FEMNIST learning rates using validation accuracy only"
"$python_bin" -m rain.cli.select_tuning \
    --inputs "$output_root"/tuning-femnist-*-seed1-lr*/rounds.jsonl \
    --output "$selection" --metric validation_accuracy \
    --expected-candidates 3 --minimum-rounds 100 \
    --methods rain,signsgd,fedavg,flod

echo "[$(date -Is)] starting single-seed FEMNIST formal runs"
"$root/scripts/group/run_femnist_formal_seed1.sh" primary "$selection" 0 &
primary_pid=$!
"$root/scripts/group/run_femnist_formal_seed1.sh" baselines "$selection" 1 &
baseline_pid=$!
wait "$primary_pid"
wait "$baseline_pid"

"$python_bin" -m rain.cli.summarize \
    --inputs "$output_root"/femnist-*-seed1/rounds.jsonl \
    --output "$output_root/femnist-seed1-summary.json"

"$python_bin" -m rain.cli.plot_convergence \
    --inputs \
    RAIN="$output_root/femnist-rain-seed1/rounds.jsonl" \
    SignSGD="$output_root/femnist-signsgd-seed1/rounds.jsonl" \
    FedAvg="$output_root/femnist-fedavg-seed1/rounds.jsonl" \
    FLOD="$output_root/femnist-flod-seed1/rounds.jsonl" \
    --metric validation_accuracy \
    --output "$output_root/femnist-seed1-validation.png" \
    --title "FEMNIST / CNN (plaintext, no attack; seed 1)"

echo "[$(date -Is)] FEMNIST seed-1 suite complete"
