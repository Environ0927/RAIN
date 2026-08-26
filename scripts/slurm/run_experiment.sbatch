#!/bin/bash -l
# Generic RAIN training/calibration job.
# Submit with CONFIG and MODE exported; training additionally requires OUTPUT.
#SBATCH --job-name=rain-experiment
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this job from the RAIN repository root}"
module purge
module load shangwang
module load python/pytorch

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLBACKEND=Agg

: "${CONFIG:?submit with --export=ALL,CONFIG=path/to/config.json}"
MODE="${MODE:-train}"

python --version
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

case "$MODE" in
  train)
    : "${OUTPUT:?training requires OUTPUT=outputs/run-name}"
    mkdir -p "$OUTPUT"
    command=(python -m rain.cli.train --config "$CONFIG" --output "$OUTPUT" --device cuda)
    if [[ -n "${RESUME:-}" ]]; then
      command+=(--resume "$RESUME")
    fi
    "${command[@]}"
    ;;
  calibrate)
    command=(python -m rain.cli.calibrate_large --config "$CONFIG" --device cuda)
    if [[ -n "${CALIBRATION_OUTPUT:-}" ]]; then
      command+=(--output "$CALIBRATION_OUTPUT")
    fi
    "${command[@]}"
    ;;
  *)
    echo "MODE must be train or calibrate" >&2
    exit 2
    ;;
esac
