#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${RAIN_PYTHON:-/home/yuhang.li/miniconda3/envs/rain-artifact/bin/python}"
output_root="$root/outputs/convergence"
selection="$output_root/cifar10-tuning-selection.json"
mkdir -p "$output_root"

echo "[$(date -Is)] starting equal-budget CIFAR-10 tuning refinement"
"$root/scripts/group/run_cifar10_refinement_seed1.sh" primary 0 &
primary_pid=$!
"$root/scripts/group/run_cifar10_refinement_seed1.sh" baselines 1 &
baseline_pid=$!
wait "$primary_pid"
wait "$baseline_pid"

echo "[$(date -Is)] selecting learning rates using validation accuracy only"
"$python_bin" -m rain.cli.select_tuning \
    --inputs "$output_root"/tuning-cifar10-*-seed1-lr*/rounds.jsonl \
    --output "$selection" --metric validation_accuracy \
    --expected-candidates 5 --minimum-rounds 200 \
    --methods rain,signsgd,fedavg,flod

echo "[$(date -Is)] starting three-seed CIFAR-10 formal runs"
"$root/scripts/group/run_cifar10_formal.sh" primary "$selection" 0 &
primary_pid=$!
"$root/scripts/group/run_cifar10_formal.sh" baselines "$selection" 1 &
baseline_pid=$!
wait "$primary_pid"
wait "$baseline_pid"

"$python_bin" -m rain.cli.summarize \
    --inputs "$output_root"/cifar10-*-seed*/rounds.jsonl \
    --output "$output_root/cifar10-formal-summary.json"

plot_inputs=()
for method in rain signsgd fedavg flod; do
    case "$method" in
        rain) label="RAIN" ;;
        signsgd) label="SignSGD" ;;
        fedavg) label="FedAvg" ;;
        flod) label="FLOD" ;;
    esac
    for seed in 1 2 3; do
        plot_inputs+=("$label=$output_root/cifar10-${method}-seed${seed}/rounds.jsonl")
    done
done
"$python_bin" -m rain.cli.plot_convergence \
    --inputs "${plot_inputs[@]}" \
    --metric validation_accuracy \
    --output "$output_root/cifar10-formal-validation.png" \
    --title "CIFAR-10 / ResNet-18 (plaintext, no attack; mean +/- std)"

echo "[$(date -Is)] CIFAR-10 formal suite complete"
