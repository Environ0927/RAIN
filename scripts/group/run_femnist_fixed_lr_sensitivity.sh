#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: run_femnist_fixed_lr_sensitivity.sh primary-or-baselines [gpu-index]" >&2
    exit 2
fi
track="$1"
gpu="${2:-0}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${RAIN_PYTHON:-/home/yuhang.li/miniconda3/envs/rain-artifact/bin/python}"
output_root="$root/outputs/convergence"
mkdir -p "$output_root"

learning_rate_for() {
    case "$1" in
        fedavg) echo "0.02" ;;
        rain) echo "0.0002" ;;
        signsgd) echo "0.00005" ;;
        flod) echo "0.00005" ;;
        *) echo "unknown method: $1" >&2; return 2 ;;
    esac
}

run_method() {
    local method="$1"
    local learning_rate
    learning_rate="$(learning_rate_for "$method")"
    for seed in 1 2; do
        local output="$output_root/sensitivity-femnist-fixedlr-${method}-seed${seed}"
        local raw="$output/rounds.jsonl"
        local rows=0
        if [[ -f "$raw" ]]; then rows="$(wc -l < "$raw")"; fi
        if (( rows >= 1000 )); then
            echo "[$(date -Is)] skipping completed fixed-LR FEMNIST $method seed=$seed"
            continue
        fi
        local resume_args=()
        if [[ -f "$output/checkpoint.pt" ]]; then
            resume_args=(--resume "$output/checkpoint.pt")
        fi
        echo "[$(date -Is)] starting fixed-LR FEMNIST $method seed=$seed lr=$learning_rate on CUDA index $gpu"
        CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m rain.cli.train \
            --config "$root/configs/convergence/femnist_${method}.json" \
            --learning-rate "$learning_rate" --seed "$seed" \
            --output "$output" --device cuda "${resume_args[@]}" \
            >> "$output_root/sensitivity-femnist-fixedlr-${method}-seed${seed}.log" 2>&1
    done
}

case "$track" in
    primary)
        run_method rain
        run_method signsgd
        ;;
    baselines)
        run_method fedavg
        run_method flod
        ;;
    *)
        echo "unknown track: $track (expected primary or baselines)" >&2
        exit 2
        ;;
esac
