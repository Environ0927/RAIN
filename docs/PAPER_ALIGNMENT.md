# Alignment with the current paper PDF

The August 22, 2026 PDF already describes two semi-honest, non-colluding
servers, Byzantine clients, secret-shared shuffle/aggregation, and no
cryptographic integrity layer. Earlier revision notes about deleting MAC,
ticket, digest, transcript, and integrity-proof material are therefore closed.

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

## Large-scale extension to add to Section 6

The current Section 6 dataset scope is MNIST, Fashion-MNIST, and CIFAR-10, and
Table 2 reports efficiency only for MNIST/Fashion-MNIST. The new experiments
should therefore be presented as an additional large-scale subsection rather
than silently replacing those results:

- CIFAR-100 with small-image ResNet-34 and Tiny-ImageNet with small-image
  ResNet-50;
- benign utility plus Scaling, Attack-DPFL, RAA, and RSCA robustness;
- at least three seeds, with mean and standard deviation;
- exact parameter dimension, clients, chunk size, C-P/P-P logical bytes,
  offline/online simulator time, and peak memory;
- a protocol-only scalability table separate from CUDA learning time;
- the bound-aware comparison optimization (`bit_length(d)` and
  `bit_length(N*T)+1`) as an implementation optimization that preserves the
  Algorithm 3 output;
- explicit Tiny-ImageNet acquisition/preprocessing and independent calibration
  partition details.
