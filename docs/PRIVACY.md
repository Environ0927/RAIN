# Privacy accounting

For integer Renyi order alpha, the artifact evaluates the analytical
multinomial shuffled-Gaussian RDP upper bound in log space. The multinomial sum
is the degree-alpha coefficient of a polynomial raised to the number of
participating clients; truncated log-domain exponentiation avoids enumerating
weak compositions. Each order is capped by ordinary local Gaussian RDP,
composed additively over rounds, and converted with
`epsilon = rho + log(1/delta)/(alpha-1)`. The report retains every tested order
and identifies the minimizing one.

The default replacement adjacency sensitivity is `2C`, while Gaussian standard
deviation is `noise_multiplier*C`. The pre-sign Gaussian release is a
conservative bound because sign encoding and secure aggregation are
post-processing. Root/reference examples are separate from clients and are not
included in client adjacency. The implementation follows the analytical bound
from [Shuffle Gaussian Mechanism for Differential Privacy](https://arxiv.org/abs/2206.09569)
and does not claim a tighter accountant.

Tests verify deterministic output, parameter validation, decreasing epsilon as
noise or the shuffled population increases, and increasing loss under more
rounds.
