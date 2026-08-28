# Baseline reproduction

`paper/baseline_registry.json` is the machine-readable source of truth. It
classifies every paper baseline as an in-tree aggregation, an adapter-backed
implementation behind the unified trainer, an analytical accountant, or a
pinned external artifact. The artifact does not relabel one implementation's
measurements as another method's result.

## Unified robustness experiments

The unified `rain-train` path supports all methods used in the paper's
robustness comparison:

| Family | Methods |
|---|---|
| Geometry/statistics | Krum, Trim-Mean, FoundationFL, FLOD, RFLPA, ShieldFL, SignGuard |
| Trust/reputation | FLTrust, FoolsGold, Divide-and-Conquer, CONTRA, ROMOA, FLARE |

FedAvg, coordinate median and SignSGD are also available. Every method uses
the same selected clients, client updates, malicious-client indices and
validation split for a given dataset/seed/attack condition. For plaintext
baselines, `protocol.preprocessing=clip_noise` applies the same clipping and
local noise before aggregation. Set it to `none` only for an explicitly
labelled ablation.

ShieldFL, FoolsGold, CONTRA and ROMOA keep cross-round state. Their state is
saved in `checkpoint.pt`; resume fails instead of silently resetting state when
the required aggregation state is missing. FLARE consumes only the frozen
public reference split and never test labels.

FoundationFL uses its Median instantiation with `synthetic_ratio=1.0` (one
synthetic copy per selected client). RFLPA's utility rule is the cosine trust
score and norm normalization defined in its paper and is therefore numerically
the same aggregation rule as FLTrust; RFLPA's packed-share protocol is kept as
a pinned upstream implementation and its cryptographic cost must not be
reported from the plaintext utility run.

Generate the complete 13-method attack matrix deterministically:

```bash
METHODS=krum,trim-mean,foundationfl,flod,rflpa,shieldfl,signguard,fltrust,foolsgold,divide-and-conquer,contra,romoa,flare
python -m rain.cli.make_sweep --base configs/large/cifar10_resnet18.json \
  --output-dir generated/robustness/cifar10 --aggregations "$METHODS" \
  --seeds 1,2,3 --attacks krum,min-max,scaling,attack-dpfl,raa,woaa \
  --ratios 0.1,0.2,0.4
```

Repeat with `configs/large/femnist_cnn.json` and
`configs/large/tinyimagenet_resnet18.json`. The generated `manifest.json`
contains the exact command for every run. This full Cartesian product is
large; the paper figure subset is defined in `paper/experiment_matrix.json`.

## Approx+Shuffling

`rain/privacy/approx_shuffling.py` implements Theorem III.1 of
arXiv:2001.03618 for attribute-fragmented k-RAPPOR. It reports central
*removal* DP and rejects parameters outside the theorem domain rather than
extrapolating the equation.

```bash
python -m rain.cli.approx_shuffling \
  --clients 1000000 --local-epsilon 2 --delta 1e-5 \
  --output outputs/privacy/approx-shuffling.json
```

This bound is not the RAIN shuffled-Gaussian accountant. Plots must retain the
mechanism and adjacency fields so the two privacy notions are not presented as
identical.

## Camel and FLGuard

Camel and FLGuard use different protocols and training stacks, so they remain
in their authors' repositories. Exact commits are mandatory:

```bash
python scripts/install_external_baselines.py --only camel --only flguard \
  --manifest external/install_manifest.json
python scripts/install_external_baselines.py --verify-only \
  --only camel --only flguard
```

Run an upstream command through the provenance wrapper. For example, the
Camel repository documents Fashion-MNIST as:

```bash
python scripts/run_external_baseline.py --name camel \
  --output-dir outputs/external/camel-fmnist-seed1 -- \
  python train_fmnist.py --epsilon 2 --epochs 1000 --lr 0.1 \
  --clip-val 0.5 --batch-size 12800 --device cuda
```

FLGuard is invoked with its upstream YAML configuration:

```bash
python scripts/run_external_baseline.py --name flguard \
  --output-dir outputs/external/flguard-cifar10-seed1 -- \
  python train_fl.py --config example_config.yaml
```

The wrapper records the exact commit, command, Python version, UTC timestamps,
return code and raw log. Normalize measured values using
`paper/external_result_template.json`, then import them with:

```bash
python -m rain.cli.import_external_results --baseline camel \
  --input measured-camel.json --output outputs/external/camel.jsonl
```

The importer rejects a wrong commit or unlabeled dataset. Camel upstream
provides MNIST and Fashion-MNIST training plus a secure-shuffle benchmark;
FLGuard upstream provides MNIST, CIFAR-10 and FEMNIST. Neither upstream
artifact provides Tiny-ImageNet. A Tiny-ImageNet port must therefore be
reported as a port, not as an upstream reproduction.

## Review checks

```bash
python -m rain.cli.audit_artifact --root .
python -m rain.cli.audit_artifact --root . --strict-external
```

The first command verifies in-tree implementations, source files, configs and
the complete baseline registry. The strict form additionally verifies that
Camel and FLGuard are clean Git checkouts at the exact commits. Directory
existence alone is insufficient.
