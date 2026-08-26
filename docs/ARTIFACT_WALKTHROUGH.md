# Artifact walkthrough

1. Run protocol tests and both demos from README. Inspect byte counts and the
   oracle checks.
2. Read `THREAT_MODEL.md`, then inspect `party.py`, `shuffle.py`, and
   `secure_engine.py`. Party private registries are never exported together.
3. Compare `secure_rain_aggregate` with `rain_aggregate_oracle`; only tests call
   the latter.
4. Run the accountant and inspect every order in its JSON output.
5. Create an independent calibration file, then replace `protocol.tau` in a
   training config with `protocol.calibration`.
6. Run the FEMNIST, CIFAR-10, and Tiny-ImageNet quick configurations and
   inspect their split/reference manifests, raw JSONL, privacy report, and
   checkpoint. Resume into the same output directory.
7. Run the 62-class CNN and the 10-/200-class ResNet-18 model tests before expensive
   full configurations.
8. Use the chunk benchmark to verify direction invariance and compare logical
   communication, wall time, and peak Python memory.
9. Run the artifact audit. Treat `implementation_complete` as the in-tree RAIN
   status and `external_reproduction_ready` as the independent third-party
   comparison-installation status.
10. Generate a robustness sweep and inspect `manifest.json`; run at least one
    round of every aggregation before launching the formal matrix. Follow
    `BASELINES.md` to install exact Camel/FLGuard commits and retain upstream
    logs and run manifests.

CI intentionally covers the dependency-light protocol core. Dataset downloads
and GPU runs remain reviewer-invoked because hosted-runner availability varies.
