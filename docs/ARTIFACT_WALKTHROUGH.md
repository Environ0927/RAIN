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
6. Run MNIST quick and inspect its split/reference manifests, raw JSONL,
   privacy report, and checkpoint. Resume into the same output directory.
7. Run FashionNet and ResNet model tests before expensive full configurations.
8. Use the chunk benchmark to verify direction invariance and compare logical
   communication, wall time, and peak Python memory.

CI intentionally covers the dependency-light protocol core. Dataset downloads
and GPU runs remain reviewer-invoked because hosted-runner availability varies.
