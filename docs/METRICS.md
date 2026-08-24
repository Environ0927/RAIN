# Metrics

`LogicalTransport` serializes and deserializes every logical cross-party array.
Bytes are measured from the actual wire buffer and include the RAIN header and
payload. Bit arrays are little-endian packed. Counts exclude TCP/IP framing,
network delay, files, model broadcast, and messages that do not exist in the
threat model (MAC/ticket/digest/transcript data).

The round record maps paper labels as follows:

| Paper label | Artifact field |
|---|---|
| C-Comp | `client_comp_seconds` |
| S0-Comp / S1-Comp | `s0_comp_seconds` / `s1_comp_seconds` |
| S-Comp-Sum | `server_comp_sum_seconds` |
| S-Comp-Critical | `server_comp_critical_seconds` |
| Offline-Comp | `offline_comp_seconds` |
| Online-Comp | `online_comp_seconds` |
| C-P Comm | `client_to_server_bytes` |
| P-P Comm | `server_to_server_bytes` |
| peak memory | `peak_memory_bytes` (Python allocations via `tracemalloc`) |

`rounds.jsonl` also contains loss, accuracy, balanced accuracy, ASR (null unless
the backdoor evaluation is active), epsilon, delta, and wall time. S0/S1 time
measures each party's local state-machine calls; sum and critical-path values
remain simulator estimates. Offline
ideal-correlation bytes are zero because no concrete OT/PRG wire protocol is
implemented. Do not report these timings as real network performance.
