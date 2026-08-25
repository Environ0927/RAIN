#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: run_femnist_formal_seed1.sh primary-or-baselines selection.json [gpu-index]" >&2
    exit 2
fi
track="$1"
selection="$2"
gpu="${3:-0}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${RAIN_PYTHON:-/home/yuhang.li/miniconda3/envs/rain-artifact/bin/python}"
output_root="$root/outputs/convergence"
mkdir -p "$output_root"

selected_lr() {
    "$python_bin" -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"][sys.argv[2]]["learning_rate"])' \
        "$selection" "$1"
}

run_method() {
    local method="$1"
    local learning_rate
    learning_rate="$(selected_lr "$method")"
    local output="$output_root/femnist-${method}-seed1"
    local raw="$output/rounds.jsonl"
    local rows=0
    if [[ -f "$raw" ]]; then rows="$(wc -l < "$raw")"; fi
    if (( rows >= 1000 )); then
        echo "[$(date -Is)] skipping completed FEMNIST $method seed=1"
        return
    fi
    local resume_args=()
    if [[ -f "$output/checkpoint.pt" ]]; then
        resume_args=(--resume "$output/checkpoint.pt")
    fi
    echo "[$(date -Is)] starting formal FEMNIST $method seed=1 lr=$learning_rate on CUDA index $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m rain.cli.train \
        --config "$root/configs/convergence/femnist_${method}.json" \
        --learning-rate "$learning_rate" --seed 1 \
        --output "$output" --device cuda "${resume_args[@]}" \
        >> "$output_root/femnist-${method}-seed1.log" 2>&1
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
