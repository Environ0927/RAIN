#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: run_cifar10_refinement_seed1.sh primary-or-baselines [gpu-index]" >&2
    exit 2
fi
track="$1"
gpu="${2:-0}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${RAIN_PYTHON:-/home/yuhang.li/miniconda3/envs/rain-artifact/bin/python}"
output_root="$root/outputs/convergence"
mkdir -p "$output_root"

run_candidate() {
    local method="$1"
    local learning_rate="$2"
    local label="${learning_rate//./p}"
    local output="$output_root/tuning-cifar10-${method}-seed1-lr${label}"
    local raw="$output/rounds.jsonl"
    local rows=0
    if [[ -f "$raw" ]]; then rows="$(wc -l < "$raw")"; fi
    if (( rows >= 200 )); then
        echo "[$(date -Is)] skipping completed $method lr=$learning_rate"
        return
    fi
    local resume_args=()
    if [[ -f "$output/checkpoint.pt" ]]; then
        resume_args=(--resume "$output/checkpoint.pt")
    fi
    echo "[$(date -Is)] starting $method lr=$learning_rate on CUDA index $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m rain.cli.train \
        --config "$root/configs/convergence/cifar10_${method}.json" \
        --learning-rate "$learning_rate" --stop-after 200 \
        --output "$output" --device cuda "${resume_args[@]}" \
        >> "$output_root/tuning-cifar10-${method}-seed1-lr${label}.log" 2>&1
}

case "$track" in
    primary)
        for lr in 0.001 0.002; do run_candidate rain "$lr"; done
        for lr in 0.001 0.002; do run_candidate signsgd "$lr"; done
        ;;
    baselines)
        for lr in 0.75 1.25; do run_candidate fedavg "$lr"; done
        for lr in 0.001 0.002; do run_candidate flod "$lr"; done
        ;;
    *)
        echo "unknown track: $track (expected primary or baselines)" >&2
        exit 2
        ;;
esac
