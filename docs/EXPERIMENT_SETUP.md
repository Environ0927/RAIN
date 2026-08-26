# Reproducible experiment configuration

This file is the authoritative companion to the paper's experimental setup.
Every reported run must archive its resolved JSON configuration, environment,
data split, calibration record, checkpoint, and raw per-round JSONL output.

## Dataset and model configuration

| Dataset | Model | Parameters | Clients | Reference | Calibration | Validation | Partition |
|---|---:|---:|---:|---:|---:|---:|---|
| FEMNIST | 62-class CNN | 3,246,270 | 100 writers | 124 | 3,200 | 10,000 | natural writer-disjoint |
| CIFAR-10 | CIFAR ResNet-18 | 11,173,962 | 20 | 500 | 2,000 | 5,000 | Dirichlet alpha=0.5 |
| Tiny-ImageNet | small-image ResNet-18 | 11,271,432 | 20 | 400 | 1,000 | 5,000 | Dirichlet alpha=0.5 |

The FEMNIST CNN has 3x3 convolutions with 32 and 64 channels, one 2x2
max-pooling layer, a 256-unit fully connected layer, and a 62-class head.
Both ResNet-18 instances use a 3x3 stride-one stem, omit the ImageNet
max-pooling stem, and use adaptive global average pooling. CIFAR-10 inputs are
32x32 and Tiny-ImageNet inputs are 64x64; only the classification head differs.

## Optimization

| Dataset | Optimizer/update rule | Batch | Local work | Learning-rate schedule |
|---|---|---:|---:|---|
| FEMNIST | minibatch gradient on each selected writer; server SGD, no momentum or weight decay | 32 | 1 local minibatch step per round | FedAvg 0.1 to 0.001; RAIN 2e-4 to 1e-5; SignSGD 1e-4 to 1e-5, cosine |
| CIFAR-10 | client SGD with momentum 0.9, Nesterov, weight decay 5e-4; server applies the aggregated delta | 64 | 5 local minibatch steps per client per round | client LR 0.05; FedAvg server step 1.0; RAIN/SignSGD 2e-4 to 1e-5, cosine |
| Tiny-ImageNet | client SGD with momentum 0.9, Nesterov, weight decay 5e-4; server applies the aggregated delta | 16 | 2 local minibatch steps per client per round | client LR 0.025; FedAvg server step 1.0, RAIN 2e-4, SignSGD/FLOD 5e-5, constant |

`local_steps` is used instead of a dataset pass, because natural FEMNIST writers
and Dirichlet clients have different sample counts. Therefore no ambiguous
"local epoch" conversion is applied: the table reports the exact number of
optimizer steps performed by every selected client in every communication round.

Figure 4 runs 1,000 rounds. CIFAR-10 uses batch size 64, five local SGD steps,
client learning rate 0.05, momentum 0.9, Nesterov momentum, and weight decay
5e-4. FedAvg applies the averaged model delta with server step 1.0. RAIN and
SignSGD use server step 2e-4 with cosine decay to 1e-5. FEMNIST uses batch size
32 and one local gradient step; the server learning rates are 0.1 for FedAvg,
2e-4 for RAIN, and 1e-4 for SignSGD, with cosine minima 1e-3, 1e-5, and 1e-5.

Validation is evaluated every 10 rounds, the full test set at round 1,000, and
checkpoints are saved every 25 rounds. CUDA automatic mixed precision is used.
The Figure 4 seed-1 configurations are in `configs/paper/`.

Robustness, privacy-utility, and reference-mismatch experiments run 200 rounds.
They use batches/local steps of 64/2 for CIFAR-10, 32/2 for FEMNIST, and 16/2
for Tiny-ImageNet. Their method-specific learning rates are tuned only on the
reserved validation split, frozen before attacks are run, and recorded in each
resolved run configuration; for example, the 200-round Tiny-ImageNet RAIN
default is 0.001 with cosine decay to 5e-5. These tuning runs are not mixed
with the 1,000-round convergence configuration in the table above.

## Local randomization and calibration

RAIN clips every client update to L2 norm C=1. The paper's local randomization
index is mapped to the implementation noise multiplier by

    sigma(epsilon0) = kappa * Delta2 / epsilon0,
    kappa = 1, Delta2 = 2C, noise_multiplier = sigma/C = 2/epsilon0.

Thus Figure 4 uses epsilon0=10 and noise multiplier 0.2. Epsilon0 is a
randomization index; the formal end-to-end `(epsilon, delta)` result comes from
the shuffled-Gaussian RDP accountant with delta=1e-5.

RAIN calibrates tau separately for every dataset, epsilon0, partition, and seed
from an independent benign calibration split. Tau is the higher-interpolated
95th percentile of normalized Hamming mismatches. Calibration data are
disjoint from client, reference, validation, and test data.

## Repetition and reporting

Formal results use seeds 1, 2, and 3 and report mean and standard deviation.
No seed may be selected by test performance. All methods in one condition use
the same partition, initialization, participant sequence, minibatch sequence,
noise seed policy, and malicious-client indices. Report accuracy, balanced
accuracy, loss, Scaling ASR, cumulative privacy, logical communication,
offline/online time, integrity overhead, and peak memory as applicable.

GPU learning runs use the plaintext implementation of the stated aggregation
function. Secure-protocol equivalence and performance are measured separately;
GPU training time is never presented as MPC runtime.
