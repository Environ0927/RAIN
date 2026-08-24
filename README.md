# RAIN research artifact

This repository contains a protocol-level simulator for RAIN under two
semi-honest, non-colluding servers and Byzantine clients. It implements client
clipping/noise/sign sharing, paper Algorithms 2–3, a shuffled-Gaussian RDP
accountant, offline threshold calibration, PyTorch training, attacks, exact
logical-message accounting, and reviewer/full configurations. It is an
experimental artifact, not a production cryptographic or network system.

## Reviewer quick start

Use Python 3.10 or 3.11. The protocol-only path needs NumPy and normally takes
well under five minutes on a laptop CPU:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m rain.cli.demo_shuffle --clients 8 --dimension 32 --chunk-size 16 --seed 7
python -m rain.cli.demo_round --clients 4 --dimension 31 --chunk-size 16 --seed 7
python -m rain.cli.privacy --clients 30 --rounds 100 --clip-norm 1 --noise-multiplier 4
```

Expected: all tests pass; shuffle prints `oracle_match: true` and
`multiset_preserved: true`; the complete round prints a 31-coordinate output
plus nonzero client-to-server and server-to-server byte counts. Oracles are
used by tests and the shuffle demonstration only, never by training.

For the CPU ML quick path (dataset download required):

```bash
python -m pip install -r requirements-cpu.txt
python -m pip install -e .
python -m rain.cli.train --config configs/quick/mnist_lr.json --output outputs/mnist-quick
```

The run writes `config.json`, `environment.json`, `data_split.json`,
`reference_set.json`, `privacy_accountant.json`, `rounds.jsonl`, and
`checkpoint.pt`. Resume with `--resume outputs/mnist-quick/checkpoint.pt`.
The quick Fashion-MNIST and CIFAR shape/protocol checks are:

```bash
python -m rain.cli.train --config configs/quick/fashion_fnet.json --output outputs/fashion-quick
python -m rain.cli.train --config configs/quick/cifar10_resnet18.json --output outputs/cifar-quick --device cuda
```

CIFAR ResNet-18 has 11,173,962 parameters. Its secure bit protocol is
computation-heavy; the quick config uses two clients, one round, and a large
chunk. A GPU accelerates gradient computation but not the NumPy protocol
simulator.

### Larger datasets and models

The artifact now includes an index-streamed large-data path: writer-partitioned
FEMNIST with a 3.25M-parameter CNN, CIFAR-100 with a 21.33M-parameter
ResNet-34, and Tiny-ImageNet with a 23.91M-parameter ResNet-50:

```bash
python -m rain.cli.train --config configs/quick/femnist_cnn.json --output outputs/femnist-quick --device cuda
python -m rain.cli.train --config configs/quick/cifar100_resnet34.json --output outputs/cifar100-quick --device cuda
python -m rain.cli.train --config configs/quick/tinyimagenet_resnet50.json --output outputs/tiny-quick --device cuda
python -m rain.cli.large_benchmark --clients 2,5,10 --dimensions 3246270,11173962,21328292,23910152 --chunk-sizes 262144
```

See [large-scale experiments](docs/LARGE_SCALE.md) before starting the full
configs; FEMNIST uses standard LEAF JSON shards, Tiny-ImageNet is a manual
download, and full runs require an independent calibration file.

## Main entry point and compatibility

The retained trainer also routes `--aggregation rain` to the new
`rain.training.adapter` path. It requires a real root set and either a fixed
calibration file or an explicit threshold:

```bash
python main.py --dataset MNIST --net lr --nworkers 4 --niter 1 --server_pc 40 \
  --aggregation rain --dp_clip 1 --dp_sigma 4 --tau_override 0.4 \
  --rain_chunk_size 4096 --output_dir outputs/main-quick
```

Only `rain` selects the artifact protocol. The deprecated `rainy` and
`rainy_tssc` prototypes and their platform-specific shuffling helpers have
been removed; there is no automatic fallback to a plaintext legacy path.

## Calibration, privacy, and benchmarks

Calibration consumes independent benign updates and a reference bit vector:

```bash
python -m rain.cli.calibrate --updates calibration_updates.npy \
  --reference reference_bits.npy --clip-norm 1 --noise-multiplier 4 \
  --quantile 0.95 --seed 1 --output calibration/tau-sigma4.json
```

Online training reads the resulting public `tau`; it never derives the default
from the current client batch. Protocol and chunk benchmarks use:

```bash
python -m rain.cli.benchmark --clients 8 --dimension 10000 --chunks 1024,4096,16384
```

Communication is the exact deterministic message header plus serialized
payload, excluding TCP/IP framing and network delay. Timings describe the
single-process protocol simulator, not a distributed deployment.

Figures are generated only from raw rows, for example:

```bash
python -m rain.cli.plot --input outputs/mnist-quick/rounds.jsonl --output outputs/mnist-quick/curves.png
```

## Full experiment configurations

- `configs/full/fashion_fnet.json`: 30 clients, 100 rounds, CPU/GPU training;
  expected hours, depending strongly on the protocol chunk and CPU.
- `configs/full/cifar10_resnet18.json`: 30 clients, 200 rounds; multi-day
  protocol simulation and a CUDA-capable GPU are recommended.
- `configs/full/privacy_curves.json`: inputs for privacy-curve sweeps.
- `configs/ablation/`: malicious ratio, root size/bias, communication, and
  chunk-size sweeps.

Run a concrete training config with `python -m rain.cli.train --config ...`.
Sweep JSON files specify axes and are intentionally raw experiment manifests;
they contain no hard-coded paper results. Output ranges depend on the attack,
partition, seed, hardware, and dataset version. Repeated seeds should reproduce
protocol directions and partitions; wall time can vary.

## Artifact map

```text
rain/client/       clipping, noise, dense signs, Boolean sharing
rain/protocol/     isolated parties, shuffle, B2A, Beaver, comparison, aggregation
rain/privacy/      versioned calibration and analytical RDP accountant
rain/training/     reference provider, coordinator, adapter, unified attacks
rain/cli/          demos, accountant, calibration, benchmark, trainer
models/            LR, SimpleCNN, FashionNet, CIFAR ResNet-18
configs/           quick, full, and ablation manifests
tests/             protocol, validation, privacy, model, attack, training tests
docs/              assumptions, design, metrics, alignment, limitations
```

See [the walkthrough](docs/ARTIFACT_WALKTHROUGH.md),
[protocol design](docs/PROTOCOL.md), [threat model](docs/THREAT_MODEL.md), and
[frozen model definitions](docs/MODELS.md) before interpreting results. Current
implementation limits are listed in [known limitations](docs/KNOWN_LIMITATIONS.md),
and the accountant convention is fixed in [privacy accounting](docs/PRIVACY.md).
