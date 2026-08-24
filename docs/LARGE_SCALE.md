# Large-dataset and large-model experiments

## Added experiment tiers

The scalable path is separate from the retained tensor-materializing loader.
It stores only sample indices, loads client mini-batches on demand, keeps the
audited root set disjoint, and evaluates the reference gradient in bounded
mini-batches. It supports:

| Dataset | Model | Classes | Parameters | Input |
|---|---|---:|---:|---:|
| CIFAR-10 | small-image ResNet-18 | 10 | 11,173,962 | 3x32x32 |
| FEMNIST | two-convolution CNN | 62 | 3,246,270 | 1x28x28 |
| CIFAR-100 | small-image ResNet-34 | 100 | 21,328,292 | 3x32x32 |
| Tiny-ImageNet | small-image ResNet-50 | 200 | 23,910,152 | 3x64x64 |

The ResNets use a 3x3 stride-one stem without the ImageNet max-pool. Full
configs use mixed-precision forward/backward on CUDA, two independent local
mini-batches per client, cosine server step decay, periodic evaluation, and
checkpointing. Protocol arithmetic remains exact NumPy integer/bit arithmetic.

## Data preparation

CIFAR-10 and CIFAR-100 download automatically through torchvision. FEMNIST can be
downloaded from the SHA-256-pinned TensorFlow Federated release and converted
into the bounded-memory cache with:

```bash
python -m rain.cli.prepare_femnist --root ./data
```

This obtains the 62-class writer-partitioned release (3,400 writers, 671,585
training examples, and 77,483 test examples). It stores TFF's two HDF5 source
files under `data/femnist/`, corrects TFF's background/ink orientation during
cache conversion, and does not require TensorFlow. On an offline server, copy
the archive or extracted HDF5 files first, extract the archive if needed, and
run the command with `--offline`.

The earlier standard LEAF JSON representation remains supported in either of
these equivalent layouts:

```text
data/femnist/data/train/*.json
data/femnist/data/test/*.json

data/femnist/train/*.json
data/femnist/test/*.json
```

Each JSON shard must contain `users`, `num_samples`, and `user_data`, with
flattened 784-pixel `x` rows and labels `y` in `[0, 61]`. On first use the
loader converts the JSON into compressed uint8 shards under
`data/femnist/processed/rain-femnist-v1/`. The cache is automatically rebuilt
when a source shard's path, size, or modification time changes. It retains the
original writer IDs and never converts FEMNIST into a synthetic Dirichlet
partition. The HDF5 path has the same writer-preservation guarantee.

Tiny-ImageNet must be downloaded and extracted so one of these paths exists:

```text
data/tiny-imagenet-200/wnids.txt
data/tiny-imagenet-200/train/
data/tiny-imagenet-200/val/
```

The original Tiny-ImageNet validation annotation layout is supported directly;
it does not need to be reorganized into class directories.

CIFAR-10, CIFAR-100, and Tiny-ImageNet use a seeded per-class Dirichlet allocation.
FEMNIST deterministically selects `population_clients` natural writers, then
samples `clients` participants without replacement each round; root and
calibration examples come only from writers outside that population. The full
config uses a 1,000-writer population and 100 participants per round. Thus
FEMNIST has sample-level and writer-level separation between public data and
the training population. Every run writes indices, selected writer IDs,
partition kind, unused sample count, and per-round participant indices.
`dirichlet_alpha=0.5` is
the default heterogeneous setting for the two image benchmarks; use larger
values for more IID-like partitions and report the value.

## Staged commands

First validate model/data plumbing:

```bash
python -m rain.cli.train --config configs/quick/femnist_cnn.json \
  --output outputs/femnist-cnn-quick --device cuda

python -m rain.cli.train --config configs/quick/cifar100_resnet34.json \
  --output outputs/cifar100-r34-quick --device cuda

python -m rain.cli.train --config configs/quick/tinyimagenet_resnet50.json \
  --output outputs/tiny-r50-quick --device cuda
```

Before a full run, create the independent noise-aware calibration file at the
path named by the config. The full configs reserve `calibration_pc` examples
that are excluded from both the root set and all client partitions:

```bash
python -m rain.cli.calibrate_large \
  --config configs/large/femnist_cnn.json --device cuda

python -m rain.cli.calibrate_large \
  --config configs/large/cifar100_resnet34.json --device cuda

python -m rain.cli.calibrate_large \
  --config configs/large/tinyimagenet_resnet50.json --device cuda
```

The tool stores the threshold record plus a manifest containing every root and
calibration index. A fixed `tau=0.4` is present only in quick smoke configs and
must not be reported as the calibrated paper setting.

Full utility runs:

```bash
python -m rain.cli.train --config configs/large/femnist_cnn.json \
  --output outputs/femnist-cnn-full --device cuda

python -m rain.cli.train --config configs/large/cifar100_resnet34.json \
  --output outputs/cifar100-r34-full --device cuda

python -m rain.cli.train --config configs/large/tinyimagenet_resnet50.json \
  --output outputs/tiny-r50-full --device cuda
```

Resume by adding `--resume OUTPUT/checkpoint.pt`. The trainer rejects a
checkpoint whose embedded config differs from `--config`.

Expand the base config into a three-seed robustness matrix without hand-editing
dozens of files:

```bash
python -m rain.cli.make_sweep \
  --base configs/large/femnist_cnn.json \
  --output-dir configs/generated/femnist-cnn \
  --seeds 1,2,3 --attacks none,scaling,attack-dpfl,raa,rsca \
  --ratios 0.1,0.2,0.4 --hamming-fraction 0.25

python -m rain.cli.make_sweep \
  --base configs/large/cifar100_resnet34.json \
  --output-dir configs/generated/cifar100-r34 \
  --seeds 1,2,3 --attacks none,scaling,attack-dpfl,raa,rsca \
  --ratios 0.1,0.2,0.4 --hamming-fraction 0.25
```

The generated manifest contains the exact command for every run. RAA/RSCA
budgets are converted from the public fraction to an exact model-coordinate
count and remain below the filtering threshold.

After the runs finish, aggregate final evaluated rows across seeds directly
from raw JSONL files:

```bash
python -m rain.cli.summarize \
  --inputs "outputs/seed*/rounds.jsonl" \
  --output outputs/large-summary.json
```

The summary reports run count, mean, and standard deviation for accuracy,
balanced accuracy, ASR, runtime, communication, and epsilon.

## Protocol-only scaling matrix

Run protocol scalability independently from costly model training:

```bash
python -m rain.cli.large_benchmark \
  --clients 2,5,10,20 \
  --dimensions 3246270,11173962,21328292,23910152 \
  --chunk-sizes 131072,262144,524288 \
  --output outputs/large-protocol/rows.jsonl
```

The command is resumable and records exact serialized C-P/P-P bytes, primitive
counts, offline/online time, peak Python allocation, and a final-direction hash.
It never substitutes a plaintext oracle. On the development machine (Intel
i7-14650HX, NumPy 1.26.4), the verified 2-client, 11,173,962-coordinate,
262,144-chunk round took 104.52 seconds and exchanged 1,086,917,198 logical
bytes. The 21,328,292-coordinate ResNet-34 point took 213.79 seconds and
exchanged 2,101,321,412 logical bytes with a 756,645,478-byte peak Python
allocation. Treat these as local smoke measurements, not paper results or
network benchmarks.

## Recommended experiment matrix

For publishable results, use at least three seeds and report mean plus standard
deviation for:

1. benign convergence and final accuracy;
2. Scaling ASR, Attack-DPFL, RAA, and RSCA at 10%, 20%, and 40% malicious clients;
3. noise multipliers 2, 4, 6 and their composed epsilon/delta;
4. root sizes 100, 200, 400 and root-distribution bias;
5. dimensions/models, clients, and chunk sizes in the protocol-only matrix;
6. GPU training time separately from protocol simulator time.

For FEMNIST, additionally report the total writer population, selected writers
per run, samples per writer, and the natural client-size distribution. Do not
label FEMNIST as Dirichlet non-IID.

Do not mix protocol simulator wall time with CUDA model-training time, and do
not present logical bytes as measured network traffic.
