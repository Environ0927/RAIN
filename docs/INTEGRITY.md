# RAIN-MAC integrity checks

RAIN-MAC implements the lightweight aggregation-input boundary in paper
Section 5.4 and the experiment in Appendix C. A clean round proceeds only when:

1. every client share verifies in the expected session, round, batch, client,
   and server-role context;
2. tickets are fresh and unique within the batch;
3. both servers accept the same sorted ticket set and batch digest;
4. every server-to-server message verifies at the expected sequence number and
   previous transcript digest; and
5. the final transcript digests match.

Run the deterministic tamper suite with:

```bash
python -m rain.cli.integrity_demo --clients 8 --dimension 32 --seed 7
```

The JSON result covers modified shares, forged tags, duplicate tickets,
cross-round replay, transcript payload modification, and transcript replay.
`all_tampering_detected` must be `true`.

Secure RAIN configs enable this layer by default. Set `protocol.integrity` to
`false` only for a clearly labelled overhead ablation. Plaintext convergence
baselines cannot enable RAIN-MAC because they do not execute the two-server
protocol.

This is not a malicious-secure 2PC or verifiable shuffle. A malicious server
that owns the transcript key can authenticate an internally incorrect payload;
that semantic deviation remains outside the paper's integrity claim.
