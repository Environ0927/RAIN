#!/bin/bash -l
#SBATCH --job-name=rain-cpu-smoke
#SBATCH --partition=9654
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this job from the RAIN repository root}"
module purge
module load python/pytorch

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -m pytest -q
python -m rain.cli.large_benchmark \
  --clients 2 \
  --dimensions 100000,1000000 \
  --chunk-sizes 262144 \
  --output outputs/server-cpu-smoke/rows.jsonl
