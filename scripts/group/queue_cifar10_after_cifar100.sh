#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${RAIN_PYTHON:-/home/yuhang.li/miniconda3/envs/rain-artifact/bin/python}"
output_root="$root/outputs/convergence"
calibration="$root/calibration/convergence_cifar10_local_sgd_sigma0.json"
mkdir -p "$output_root" "$root/calibration"

cifar100_complete() {
    "$python_bin" - "$output_root" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
methods = ("rain", "signsgd", "fedavg", "flod")
complete = all(
    sum(1 for _ in (root / f"cifar100-{method}-seed1" / "rounds.jsonl").open()) >= 1000
    if (root / f"cifar100-{method}-seed1" / "rounds.jsonl").is_file()
    else False
    for method in methods
)
raise SystemExit(0 if complete else 1)
PY
}

echo "[$(date -Is)] waiting for all four CIFAR-100 runs"
until cifar100_complete; do
    sleep 30
done
echo "[$(date -Is)] CIFAR-100 complete"

if [[ ! -f "$calibration" ]]; then
    echo "[$(date -Is)] calibrating CIFAR-10 local-update threshold on CUDA index 0"
    CUDA_VISIBLE_DEVICES=0 "$python_bin" -u -m rain.cli.calibrate_large \
        --config "$root/configs/convergence/cifar10_rain.json" \
        --output "$calibration" \
        --device cuda
fi

echo "[$(date -Is)] starting equal-budget CIFAR-10 pilots"
"$root/scripts/group/run_cifar10_pilots_seed1.sh" primary 0 &
primary_pid=$!
"$root/scripts/group/run_cifar10_pilots_seed1.sh" baselines 1 &
baselines_pid=$!
wait "$primary_pid"
wait "$baselines_pid"
echo "[$(date -Is)] CIFAR-10 pilots complete"
