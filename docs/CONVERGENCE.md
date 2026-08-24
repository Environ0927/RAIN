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

## Method definitions

- `rain`: reference-Hamming ReLU weights followed by a coordinate sign.
- `signsgd`: unweighted coordinate majority sign, with zero on a tied vote.
- `fedavg`: data-size-weighted full-precision gradient average. With one local
  step, this is the gradient form of one-step FedAvg.
- `flod`: reference-Hamming ReLU weights followed by a normalized weighted
  sign vector, preserving magnitude below one.

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

Do not select learning rates using test accuracy.  The checked-in values are
initial presets.  Before a paper result, apply the same small candidate grid
to every method using a held-out validation split, freeze the selected values,
then run at least three seeds.  Report every method, including a FLOD result
that is stronger than RAIN.

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
