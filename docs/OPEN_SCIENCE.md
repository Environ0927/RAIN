# Open-science and artifact availability

This document is the reviewer-facing inventory for the anonymous source
artifact. The machine-readable version is `paper/artifact_inventory.json`.
Nothing marked as pending is claimed as a completed experimental result.

## Included materials

- The RAIN protocol simulator, integrity layer, privacy accountant, threshold
  calibration, training pipeline, attacks, and in-tree baseline integrations.
- Locked dataset/model/training configurations and the experiment matrix.
- Tests, protocol demonstrations, external-baseline installers, provenance
  wrappers, and result-import validation.
- A source manifest (`ARTIFACT_MANIFEST.sha256`) generated for each review
  package.

The anonymous package intentionally excludes Git history, caches, private
cluster operations, exploratory stress configurations, datasets, checkpoints,
and incomplete or selectively chosen experimental outputs.

## Data availability

CIFAR-10 is acquired through `torchvision` by the training command. FEMNIST is
prepared with `python -m rain.cli.prepare_femnist --root ./data` or supplied as
standard LEAF JSON shards. Tiny-ImageNet must be obtained by the reviewer under
its distributor's terms and placed under `data/tiny-imagenet-200`. The datasets
are not redistributed because of size and upstream distribution terms. Exact
directory layouts and split rules are documented in `docs/LARGE_SCALE.md`.

Each run archives the resolved configuration and split metadata. Reference,
calibration, validation, and test examples are disjoint. Dataset-dependent RAIN
threshold files must be produced with `rain-calibrate-large`; a threshold from
another dataset, partition, seed, or randomization setting is rejected.
CIFAR-10 and FEMNIST Figure 4 calibration records and their split manifests are
included under `calibration/paper`. Tiny-ImageNet remains marked pending until
the separately distributed dataset is available; no threshold is estimated or
copied from a different dataset.

## Quick functional evaluation

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m rain.cli.demo_shuffle --clients 8 --dimension 32 --chunk-size 16 --seed 7
python -m rain.cli.demo_round --clients 4 --dimension 31 --chunk-size 16 --seed 7
python -m rain.cli.integrity_demo --clients 8 --dimension 32 --seed 7
python -m rain.cli.audit_artifact --root . --strict-open-science
```

The final command also checks for Git metadata, user-specific home paths,
private IP addresses, email addresses, required open-science files, and the
anonymous artifact URL. Before an anonymous URL is assigned, omit
`--strict-open-science` and inspect the reported pending field.

## External comparisons

Camel and FLGuard are not vendored. Their repositories and exact commits are
recorded in `paper/external_baselines.json`. Install and verify them with:

```bash
python scripts/install_external_baselines.py --only camel --only flguard
python scripts/install_external_baselines.py --verify-only --only camel --only flguard
```

The provenance wrapper records the commit, command, timestamps, return code,
and unmodified upstream log. Measurements from one implementation are never
relabeled as another method.

## Results and reporting policy

Formal utility results use the prespecified seeds in
`paper/experiment_matrix.json` and are reported as mean and standard deviation.
No seed is selected by test-set performance. Plotting may change presentation
only; any smoothing must be declared and raw observations must remain
available. A result group is complete only when its raw logs, resolved configs,
environment record, split metadata, and checksums are listed in
`paper/results_manifest.json`.

At the current source snapshot, incomplete result groups remain explicitly
marked incomplete. They must be added from genuine executions before any paper
claim depending on them is finalized.

## Availability lifecycle

During double-blind review, the URL in `paper/open_science.json` must resolve
without authentication or tracking and remain available through shepherd
approval. After acceptance, the same materials and completed result records
will be published at a stable, non-anonymous archival location.
