# Plaintext convergence experiment

This experiment isolates optimization behavior from the two-server execution
layer.  It does not run shuffling, Boolean sharing, secure comparison, or
logical communication accounting.  All methods use the same model
initialization, data partition, sampled client batches, evaluation schedule,
and attack-free setting.

The primary comparison is RAIN, SignSGD, and FedAvg.  FLOD is an optional
additional reference-guided baseline.  It must receive the same data and
tuning budget as the other methods; a deliberately weakened baseline is not a
valid experiment.

## Experiment order

The current large-benchmark order is CIFAR-10, FEMNIST, then Tiny-ImageNet.
CIFAR-10 is the first fully trained convergence experiment; the earlier
CIFAR-100 seed-1 run is retained as an under-trained pipeline pilot rather than
used as the final utility result.

The CIFAR-10 configs use a CIFAR ResNet-18, 100 fixed Dirichlet-alpha-0.5
clients, 20 sampled clients per round, and five sequential local SGD steps per
client. Client SGD uses learning rate 0.05, momentum 0.9, Nesterov momentum,
and weight decay 5e-4 for every aggregation method. With batch size 64 and
1,000 rounds, this exposes the optimizer to about 150 training-set equivalents.
The 500-example reference set, 2,000-example threshold-calibration set, and
5,000-example tuning-validation set are mutually disjoint. Test accuracy is
withheld until round 1,000.

Generate the common sigma-zero RAIN/FLOD threshold before tuning:

```bash
python -m rain.cli.calibrate_large \
  --config configs/convergence/cifar10_rain.json --device cuda
```

Then run the equal-budget 200-round learning-rate pilots on two GPUs:

```bash
scripts/group/run_cifar10_pilots_seed1.sh primary 0
scripts/group/run_cifar10_pilots_seed1.sh baselines 1
```

On the group server, the handoff can be queued while the retained CIFAR-100
runs are still finishing. It waits until all four raw logs contain 1,000 rows,
then calibrates once and starts the two pilot tracks:

```bash
nohup scripts/group/queue_cifar10_after_cifar100.sh \
  > outputs/convergence/cifar10-queue.log 2>&1 &
```

The initial three server-step candidates are `1e-4`, `2e-4`, and `5e-4` for
RAIN, SignSGD, and FLOD; FedAvg uses server interpolation candidates 0.5, 1.0,
and 1.5. Because the seed-1 pilot placed all three sign-based methods at the
upper grid boundary, the preregistered refinement adds `1e-3` and `2e-3` to
each of those methods. FedAvg receives the same total candidate count by adding
0.75 and 1.25. This produces five retained candidates per method.

The paper's Figure 4 comparison is MNIST/FMNIST with Shuffle-DP at
`epsilon_0=10`. It does not imply that SignSGD must be weak in this separate
CIFAR-10 plaintext, sigma-zero experiment. Do not change a baseline merely to
force the Figure 4 ordering.

Start the auditable refinement-to-formal pipeline on the two-GPU group server:

```bash
nohup scripts/group/queue_cifar10_formal.sh \
  > outputs/convergence/cifar10-formal-queue.log 2>&1 &
```

The queue selects each method's learning rate by validation accuracy only,
then runs seeds 1, 2, and 3 for 1,000 rounds. `data.partition_seed=1` keeps the
root, calibration, validation, and client partitions identical across seeds;
the training seed still changes initialization, client sampling, minibatches,
and augmentation. Test accuracy remains withheld until the final round. The
queue writes `cifar10-formal-summary.json` and a validation-convergence figure
with mean and one-standard-deviation bands. Raw logs and the complete tuning
selection record are retained.

### Single-seed noise probe

The optional exploratory probe applies the same `L2` clipping (`C=1`) and
coordinate-wise Gaussian perturbation (`noise_multiplier=5e-5`) to all four
methods. It freezes the clean-validation-selected learning rates and runs seed
1 only. Because this very small perturbation does not provide a meaningful DP
budget, these outputs are utility-sensitivity evidence and must not be labeled
as a formal Shuffle-DP result.

```bash
nohup scripts/group/run_cifar10_noise5e-5_seed1.sh 1 \
  > outputs/convergence/explore-cifar10-noise5e-5-queue.log 2>&1 &
```

## Method definitions

- `rain`: reference-Hamming ReLU weights followed by a coordinate sign.
- `signsgd`: unweighted coordinate majority sign, with zero on a tied vote.
- `fedavg`: data-size-weighted full-precision gradient average. With one local
  step, this is the gradient form of one-step FedAvg.
- `flod`: reference-Hamming ReLU weights followed by a normalized weighted
  sign vector, preserving magnitude below one.

## Retained CIFAR-100 pilot

The CIFAR-100 configurations use ResNet-34, ten fixed Dirichlet clients,
one local step, no malicious clients, and no DP perturbation.  They are
therefore labeled `plaintext, sigma=0`; they must not be described as a
Shuffle-DP result.  RAIN and FLOD share one independently generated threshold
file:

```bash
python -m rain.cli.calibrate_large \
  --config configs/convergence/cifar100_rain.json \
  --device cuda
```

Run the four methods into separate output directories:

```bash
python -m rain.cli.train --config configs/convergence/cifar100_rain.json \
  --output outputs/convergence/cifar100-rain-seed1 --device cuda
python -m rain.cli.train --config configs/convergence/cifar100_signsgd.json \
  --output outputs/convergence/cifar100-signsgd-seed1 --device cuda
python -m rain.cli.train --config configs/convergence/cifar100_fedavg.json \
  --output outputs/convergence/cifar100-fedavg-seed1 --device cuda
python -m rain.cli.train --config configs/convergence/cifar100_flod.json \
  --output outputs/convergence/cifar100-flod-seed1 --device cuda
```

Do not select learning rates using test accuracy. The 500 reserved calibration
examples also form a held-out validation loader, and the convergence configs
write `validation_accuracy` every ten rounds while withholding test accuracy
until round 1000. The checked-in values are initial presets. Before a paper
result, apply the same small candidate-grid size to every method, freeze the
selected values, then run at least three seeds. Report every method, including
a FLOD result that is stronger than RAIN.

Use `--stop-after 100 --learning-rate VALUE` for each pilot. The resolved
learning rate is written into the run's `config.json`, the pilot stops with a
resumable checkpoint, and test accuracy remains unevaluated because round 100
is not the configured final round. Use the same number of candidate values for
every method.

Plot only from raw JSONL logs:

```bash
python -m rain.cli.plot_convergence --inputs \
  RAIN=outputs/convergence/cifar100-rain-seed1/rounds.jsonl \
  SignSGD=outputs/convergence/cifar100-signsgd-seed1/rounds.jsonl \
  FedAvg=outputs/convergence/cifar100-fedavg-seed1/rounds.jsonl \
  FLOD=outputs/convergence/cifar100-flod-seed1/rounds.jsonl \
  --output outputs/convergence/cifar100-seed1.png \
  --title "CIFAR-100 / ResNet-34 (plaintext, sigma=0)"
```

During learning-rate selection, add `--metric validation_accuracy`. Final
paper curves use the test metric after hyperparameters have been frozen.

On the two-GPU group server, the selected seed-1 checkpoints can be resumed in
two serial tracks so no two processes share one GPU:

```bash
scripts/group/run_convergence_seed1.sh primary 0
scripts/group/run_convergence_seed1.sh baselines 1
```

The current validation-selected learning rates embedded in that launcher are
RAIN `2e-4`, SignSGD `1e-4`, FedAvg `0.1`, and FLOD `2e-4`. Preserve the raw
pilot directories and validation logs as tuning evidence.

## FEMNIST continuation

The FEMNIST convergence configs use the official 62-class TFF release, a
3,246,270-parameter two-convolution CNN, 1,000 natural writer clients in the
fixed population, and 100 randomly participating writers per round. They use
the same four plaintext aggregation definitions, 1,000-round schedule,
validation-only tuning, and final-only test evaluation as CIFAR-100. The 124
root examples, 3,200 threshold-calibration examples, and 10,000 tuning-
validation examples are writer-disjoint from participating clients and
mutually disjoint from one another. FEMNIST is naturally writer-partitioned
and must not be described as Dirichlet data.

Prepare the data before the GPUs become available:

```bash
python -m rain.cli.prepare_femnist --root ./data
```

Then calibrate the sigma-zero RAIN/FLOD threshold once:

```bash
python -m rain.cli.calibrate_large \
  --config configs/convergence/femnist_rain.json --device cuda
```

After CIFAR-100 completes, use equal three-value learning-rate budgets on the
two GPUs:

```bash
scripts/group/run_femnist_pilots_seed1.sh primary 0
scripts/group/run_femnist_pilots_seed1.sh baselines 1
```

Select by round-100 held-out `validation_accuracy`, retain all pilot logs, and
only then freeze the four learning rates for the 1,000-round runs.

For the single-seed run on the group server, the queue calibrates once, starts
the baseline pilots on GPU 1 immediately, waits for the existing CIFAR-10
SignSGD track to release GPU 0, then performs equal-budget selection and the
four 1,000-round seed-1 runs:

```bash
nohup scripts/group/queue_femnist_seed1.sh \
  > outputs/convergence/femnist-seed1-queue.log 2>&1 &
```
