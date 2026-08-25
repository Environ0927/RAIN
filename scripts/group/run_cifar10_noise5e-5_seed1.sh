#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-1}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${RAIN_PYTHON:-/home/yuhang.li/miniconda3/envs/rain-artifact/bin/python}"
output_root="$root/outputs/convergence"
calibration="$root/calibration/convergence_cifar10_local_sgd_sigma5e-5.json"
mkdir -p "$output_root"

if [[ ! -f "$calibration" ]]; then
    echo "[$(date -Is)] calibrating the shared sigma=5e-5 threshold on CUDA index $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m rain.cli.calibrate_large \
        --config "$root/configs/convergence/cifar10_noise5e-5_rain.json" \
        --output "$calibration" --device cuda
fi

for method in rain signsgd fedavg flod; do
    output="$output_root/explore-cifar10-noise5e-5-${method}-seed1"
    raw="$output/rounds.jsonl"
    rows=0
    if [[ -f "$raw" ]]; then rows="$(wc -l < "$raw")"; fi
    if (( rows >= 1000 )); then
        echo "[$(date -Is)] skipping completed noisy $method seed=1"
        continue
    fi
    resume_args=()
    if [[ -f "$output/checkpoint.pt" ]]; then
        resume_args=(--resume "$output/checkpoint.pt")
    fi
    echo "[$(date -Is)] starting noisy $method seed=1 on CUDA index $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m rain.cli.train \
        --config "$root/configs/convergence/cifar10_noise5e-5_${method}.json" \
        --output "$output" --device cuda "${resume_args[@]}" \
        >> "$output_root/explore-cifar10-noise5e-5-${method}-seed1.log" 2>&1
done

echo "[$(date -Is)] single-seed sigma=5e-5 exploratory suite complete"
