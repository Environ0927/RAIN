# Threat model

RAIN uses two servers, S0 and S1. For confidentiality, both follow the protocol
but may inspect their local view; at most one is passively corrupted and they do
not collude. Clients may be Byzantine and submit arbitrary updates. Client
updates are clipped, randomized, signed, and XOR-shared before reaching the
servers.

One server sees only its local shares, randomness, permutation, and received
masked messages. The two `PartyState` objects and primitive registries are
separate. Cross-party values use the deterministic transport. The coordinator
holds handles and message lists, not both private registries. Only the final
aggregate direction is reconstructed.

If S0 and S1 collude, shuffle anonymity is lost; only local Gaussian
randomization remains. RAIN-MAC additionally models limited active message
tampering at the aggregation-input boundary. It authenticates an honest
client's share in its session/round/batch/server context, rejects duplicate
tickets and inconsistent accepted batches, and authenticates the ordered
server-to-server transcript. Any failure aborts the round before release.

Semantic active server deviation inside MPC, malicious-secure MPC, traffic
analysis, denial of service, and compromise of MAC keys are out of scope.
Clients are not anonymous against external metadata in this in-process model.

RAIN-MAC is not a verifiable shuffle or a proof of internal MPC correctness.
Version, chunk, direction, shape, dtype, length, and bit-domain checks remain
ordinary engineering validation; the cryptographic claim is limited to MAC
unforgeability and hash-based batch/transcript consistency.
