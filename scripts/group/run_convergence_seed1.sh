#!/usr/bin/env bash
set -euo pipefail

track="${1:?usage: run_convergence_seed1.sh {primary|baselines} [gpu-index]}"
gpu="${2:-0}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${RAIN_PYTHON:-/home/yuhang.li/miniconda3/envs/rain-artifact/bin/python}"
output_root="$root/outputs/convergence"

run_method() {
    local method="$1"
    local config="$2"
    local learning_rate="$3"
    local output="$output_root/cifar100-${method}-seed1"
    local checkpoint="$output/checkpoint.pt"
    if [[ ! -f "$checkpoint" ]]; then
        echo "missing pilot checkpoint: $checkpoint" >&2
        return 2
    fi
    echo "[$(date -Is)] resuming $method on CUDA index $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m rain.cli.train \
        --config "$root/$config" \
        --learning-rate "$learning_rate" \
        --stop-after 1000 \
        --resume "$checkpoint" \
        --output "$output" \
        --device cuda \
        >> "$output_root/cifar100-${method}-seed1.log" 2>&1
    echo "[$(date -Is)] completed $method"
}

case "$track" in
    primary)
        run_method rain configs/convergence/cifar100_rain.json 0.0002
        run_method signsgd configs/convergence/cifar100_signsgd.json 0.0001
        ;;
    baselines)
        run_method fedavg configs/convergence/cifar100_fedavg.json 0.1
        run_method flod configs/convergence/cifar100_flod.json 0.0002
        ;;
    *)
        echo "unknown track: $track (expected primary or baselines)" >&2
        exit 2
        ;;
esac
