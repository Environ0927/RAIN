# Running RAIN on Zhongke Suanlian Cloud

This repository has been verified on both the Northwest-1 and East-China
cluster login nodes. Their filesystems are independent.

```text
Cluster        SSH alias       Login host                    GPU partition
Northwest-1    rain-hpc        c1.hpcmaster.com:50888        4090
East-China     rain-hpc-east   dl01.hpcmaster.com:50888      A800-N

Repository on each cluster: $HOME/yuhangli/RAIN
Runtime module: python/pytorch
```

The observed module provides Python 3.12.2, PyTorch 2.4.1+cu118,
torchvision 0.19.1+cu118, and NumPy 1.26.4. The repository test suite passed on
the login node. CUDA is expected to be unavailable on the login node and must
only be used inside a Slurm GPU allocation.

## Login and update

```bash
ssh rain-hpc
cd "$HOME/yuhangli/RAIN"
git pull --ff-only
```

Use `ssh rain-hpc-east` instead for East China. Do not copy an absolute data
path from one cluster to the other: code, datasets, calibration files, and
outputs must exist on the selected cluster's own storage.

Do not run training directly on the login node. First submit the bounded GPU
smoke job:

```bash
sbatch scripts/slurm/gpu_smoke.sbatch
squeue -u "$USER"
tail -f slurm-rain-gpu-smoke-JOBID.out
```

On East China, override the script's Northwest-1 default partition:

```bash
sbatch --partition=A800-N scripts/slurm/gpu_smoke.sbatch
```

The smoke job checks the allocated GPU, performs a CUDA matrix multiplication,
runs the complete test suite, and executes a 10,000-coordinate RAIN round.

The protocol-only path can be checked independently on the CPU partition:

```bash
sbatch scripts/slurm/cpu_protocol_smoke.sbatch
```

This job runs the test suite and exact protocol rounds at 100,000 and 1,000,000
coordinates. It does not use a GPU or download a dataset.

## Quick training jobs

CIFAR-100 is downloaded automatically. Submit its one-round configuration
before attempting a full run:

```bash
sbatch --export=ALL,MODE=train,CONFIG=configs/quick/cifar100_resnet34.json,OUTPUT=outputs/cifar100-quick \
  scripts/slurm/run_experiment.sbatch
```

Add `--partition=A800-N` before `--export` when submitting on East China.

FEMNIST and Tiny-ImageNet are not downloaded automatically. Upload them to the
same Northwest-1 cluster storage used by the job:

```text
data/femnist/data/train/*.json
data/femnist/data/test/*.json
data/tiny-imagenet-200/wnids.txt
data/tiny-imagenet-200/train/
data/tiny-imagenet-200/val/
```

Then use the corresponding quick config by changing `CONFIG` and `OUTPUT`.

## Calibration and full runs

Every full large-data config references an independently generated calibration
file. Generate it first:

```bash
sbatch --export=ALL,MODE=calibrate,CONFIG=configs/large/cifar100_resnet34.json \
  scripts/slurm/run_experiment.sbatch
```

After calibration succeeds:

```bash
sbatch --export=ALL,MODE=train,CONFIG=configs/large/cifar100_resnet34.json,OUTPUT=outputs/cifar100-r34-full \
  scripts/slurm/run_experiment.sbatch
```

Resume a checkpointed run with:

```bash
sbatch --export=ALL,MODE=train,CONFIG=configs/large/cifar100_resnet34.json,OUTPUT=outputs/cifar100-r34-full,RESUME=outputs/cifar100-r34-full/checkpoint.pt \
  scripts/slurm/run_experiment.sbatch
```

Use `squeue -u "$USER"`, `scontrol show job JOBID`, and the generated
`slurm-*.out`/`slurm-*.err` files for monitoring. The protocol simulator is
single-process CPU/NumPy code even when model gradients use the GPU, so report
GPU training time separately from protocol wall time.
