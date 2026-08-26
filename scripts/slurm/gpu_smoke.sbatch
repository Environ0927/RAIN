#!/bin/bash -l
#SBATCH --job-name=rain-gpu-smoke
#SBATCH --partition=4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this job from the RAIN repository root}"
module purge
module load python/pytorch

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

nvidia-smi

python - <<'PY'
import platform
import torch

print("host", platform.node())
print("torch", torch.__version__)
print("cuda_build", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("Slurm allocated no CUDA device")
print("gpu", torch.cuda.get_device_name(0))
print("gpu_count", torch.cuda.device_count())
x = torch.randn(2048, 2048, device="cuda")
print("cuda_matmul_checksum", float((x @ x).mean()))
PY

python -m pytest -q
python -m rain.cli.demo_round --clients 4 --dimension 10000 --chunk-size 4096 --seed 7
