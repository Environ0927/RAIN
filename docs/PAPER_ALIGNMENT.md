# Alignment with the current paper PDF

The August 26, 2026 PDF describes two semi-honest, non-colluding servers,
Byzantine clients, secret-shared shuffle/aggregation, and the RAIN-MAC
aggregation-input integrity layer in Section 5.4 and Appendix C. The artifact
therefore authenticates client shares with fresh tickets, compares canonical
accepted-batch digests, authenticates the ordered server transcript, and aborts
on any failed check before an aggregate is released.

The following implementation details should still be stated explicitly when
the paper is synchronized with the released artifact:

- the implementation is a single-process protocol-level simulator;
- computation timings exclude network delay and are not distributed deployment
  measurements;
- communication is the deterministic packed logical-message size and excludes
  TCP/IP framing;
- arithmetic uses `Z/(2^64)`, ideal correlated preprocessing, a tie-to-+1 rule,
  and omitted positive weight normalization;
- the accountant uses replacement sensitivity `2C` and a conservative
  pre-sign shuffled-Gaussian RDP upper bound;
- offline OT/PRG wire cost is not measured because correlated setup remains an
  ideal functionality.
- MAC keys are deterministically derived from the run seed in this simulator;
  a deployment must provision independent high-entropy pairwise/session keys.
- RAIN-MAC protects the aggregation-input boundary and transcript only. It does
  not prove semantic correctness inside MPC or a prescribed shuffle permutation.

## Locked evaluation datasets

The released experiment scope is now locked to FEMNIST, CIFAR-10, and
Tiny-ImageNet. Earlier MNIST, Fashion-MNIST, and CIFAR-100 code remains only as
auxiliary compatibility material and must not be mixed into the primary result
tables without an explicit paper revision. The primary models are:

- FEMNIST with the 62-class CNN and CIFAR-10/Tiny-ImageNet with small-image
  ResNet-18 (10-class and 200-class heads, respectively);
- benign utility plus Krum Attack, Min-Max, Scaling, Attack-DPFL, RAA, and
  WOAA robustness. RSCA remains available only for reproducing an older draft;
- at least three seeds, with mean and standard deviation;
- exact parameter dimension, clients, chunk size, C-P/P-P logical bytes,
  offline/online simulator time, and peak memory;
- a protocol-only scalability table separate from CUDA learning time;
- the bound-aware comparison optimization (`bit_length(d)` and
  `bit_length(N*T)+1`) as an implementation optimization that preserves the
  Algorithm 3 output;
- explicit Tiny-ImageNet acquisition/preprocessing and independent calibration
  partition details.
