# Known limitations

- Shuffle masks, daBits, and Beaver/GMW triples come from explicit ideal
  offline correlation functionalities. An OT/PRG implementation and its wire
  cost are not included. This is the principal cryptographic implementation gap.
- The simulator is single-process and sequential. It does not model sockets,
  latency, scheduling, traffic analysis, or real distributed throughput.
- The GMW ripple-carry comparison is deliberately auditable rather than fast.
  It now uses a bound-derived bit width instead of a fixed 64-bit circuit, but
  dense 21M-24M-parameter rounds remain communication- and memory-intensive.
- `tracemalloc` measures Python-managed allocations and can undercount native
  NumPy/PyTorch/CUDA memory.
- The RDP implementation is the analytical shuffled-Gaussian multinomial upper
  bound, conservatively capped by local Gaussian RDP. It is not claimed to be a
  tighter accountant than the cited paper and uses replacement sensitivity 2C.
- The legacy baseline collection has optional scientific dependencies and was
  preserved rather than comprehensively redesigned. The new `rain` path has no
  dependency on its plaintext RAIN implementation.
- The built-in image backdoor is a deterministic bottom-right 3x3 trigger with
  target class 0. Alternative paper-specific triggers require an explicit
  experiment configuration and should not be compared as identical attacks.
- CIFAR-100 downloads automatically. Tiny-ImageNet licensing/distribution is
  not bundled, so reviewers must provide the extracted dataset directory.
- FEMNIST is not downloaded automatically. Reviewers must provide standard
  LEAF train/test JSON shards; the first-run conversion requires additional
  disk space for the processed uint8 cache.
- Large full configs intentionally reference calibration files that are not
  fabricated by the repository. They must be produced from independent benign
  calibration data before reporting full results.
