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

The three server-step candidates are `1e-4`, `2e-4`, and `5e-4` for RAIN,
SignSGD, and FLOD; FedAvg uses server interpolation candidates 0.5, 1.0, and
1.5. All methods share the client optimizer and candidate-count budget.

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
validation-only tuning, and final-only test evaluation as CIFAR-100. FEMNIST
is naturally writer-partitioned and must not be described as Dirichlet data.

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
