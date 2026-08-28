# Running RAIN on a Slurm cluster

This walkthrough deliberately uses site-neutral placeholders so it can be
included in the anonymous review artifact. Replace `<gpu-partition>` with the
partition name shown by your cluster and clone the artifact under any directory
owned by the current user.

The job scripts load a site-provided PyTorch module. Set
`RAIN_PYTORCH_MODULE` when the module has a different name. CUDA is expected
to be unavailable on the login node and must only be used inside a Slurm GPU
allocation.

## Login and update

```bash
ssh <cluster-login-alias>
cd <artifact-directory>
git pull --ff-only
```

Do not copy an absolute data path from one cluster to another: code, datasets,
calibration files, and outputs must exist on the selected cluster's storage.

Do not run training directly on the login node. First submit the bounded GPU
smoke job:

```bash
sbatch scripts/slurm/gpu_smoke.sbatch
squeue -u "$USER"
tail -f slurm-rain-gpu-smoke-JOBID.out
```

When the script's default partition does not match the site, override it:

```bash
sbatch --partition=<gpu-partition> scripts/slurm/gpu_smoke.sbatch
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

Override the default partition with `--partition=<gpu-partition>` when needed.

Tiny-ImageNet is not downloaded automatically. For FEMNIST, connected hosts
can run `python -m rain.cli.prepare_femnist --root ./data`; offline compute
nodes must receive the pinned TFF archive/HDF5 files or LEAF JSON shards in
the same cluster storage used by the job:

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
