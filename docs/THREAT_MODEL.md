# Threat model

RAIN uses two servers, S0 and S1. Both follow the protocol but may inspect their
local view. At most one is passively corrupted and they do not collude. Clients
may be Byzantine and submit arbitrary updates. Client updates are clipped,
randomized, signed, and XOR-shared before reaching the servers.

One server sees only its local shares, randomness, permutation, and received
masked messages. The two `PartyState` objects and primitive registries are
separate. Cross-party values use the deterministic transport. The coordinator
holds handles and message lists, not both private registries. Only the final
aggregate direction is reconstructed.

If S0 and S1 collude, shuffle anonymity is lost; only local Gaussian
randomization remains. Active server deviation, malicious-secure MPC, traffic
analysis, denial of service, and active network attacks are out of scope.
Clients are not anonymous against external metadata in this in-process model.

There are deliberately no MACs, tickets, batch digests, transcript
authentication, blind-MAC checks, verifiable shuffle, or integrity proofs.
Version, round, batch, chunk, direction, shape, dtype, length, and bit-domain
checks are ordinary engineering validation and provide no malicious security.
