"""Framework-independent orchestration of one complete RAIN round."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import asdict, dataclass

import numpy as np

from rain.client.randomizer import ClientRandomizer
from rain.client.sharing import share_client_batch
from rain.integrity import IntegrityContext, RainMacSession
from rain.protocol.aggregation import secure_rain_aggregate
from rain.protocol.boolean_sharing import reconstruct_bits, validate_bits
from rain.protocol.preprocessing import ShufflePreprocessingCache
from rain.protocol.shuffle import run_secret_shared_shuffle
from rain.transport import LogicalTransport
from rain.types import ProtocolPhase


@dataclass(frozen=True, slots=True)
class RoundMetrics:
    client_comp_seconds: float
    s0_comp_seconds: float
    s1_comp_seconds: float
    server_comp_sum_seconds: float
    server_comp_critical_seconds: float
    offline_comp_seconds: float
    online_comp_seconds: float
    client_to_server_bytes: int
    server_to_server_bytes: int
    offline_bytes: int
    online_bytes: int
    message_count: int
    peak_memory_bytes: int
    integrity_comp_seconds: float
    integrity_bytes: int
    integrity_message_count: int
    integrity_verified: bool
    batch_digest: str
    transcript_digest: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RoundResult:
    direction_bits: np.ndarray
    metrics: RoundMetrics
    threshold_count: int


class RainRoundCoordinator:
    """Runs client randomization, Algorithm 2, and Algorithm 3 without an oracle."""

    def __init__(
        self,
        *,
        clip_norm: float,
        noise_multiplier: float,
        tau: float,
        chunk_size: int = 4096,
        seed: int = 0,
        integrity: bool = True,
        session_id: str = "rain",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.clip_norm = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.tau = float(tau)
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        self.integrity = bool(integrity)
        if not session_id:
            raise ValueError("session_id must not be empty")
        self.session_id = str(session_id)
        self.preprocessing_cache = ShufflePreprocessingCache()

    def run_round(
        self,
        client_updates: np.ndarray,
        reference_bits: np.ndarray,
        *,
        round_id: int,
        batch_id: int = 0,
    ) -> RoundResult:
        updates = np.asarray(client_updates, dtype=np.float32)
        if updates.ndim != 2 or min(updates.shape) <= 0:
            raise ValueError("client_updates must be a non-empty matrix")
        reference = validate_bits(np.asarray(reference_bits, dtype=np.uint8), name="reference_bits")
        if reference.shape != (updates.shape[1],):
            raise ValueError("reference_bits dimension does not match client updates")
        if round_id < 0 or batch_id < 0:
            raise ValueError("round_id and batch_id must be non-negative")

        tracemalloc.start()
        round_seed = np.random.SeedSequence([self.seed, round_id, batch_id])
        seeds = [int(x) for x in round_seed.generate_state(8, dtype=np.uint64)]
        randomizer = ClientRandomizer(
            clip_norm=self.clip_norm,
            noise_multiplier=self.noise_multiplier,
            seed=seeds[0],
        )
        randomized = randomizer.randomize_batch(updates)
        sharing_started = time.perf_counter()
        share0, share1 = share_client_batch(randomized.bits, seed=seeds[1])
        client_seconds = randomized.elapsed_seconds + time.perf_counter() - sharing_started

        integrity_seconds = 0.0
        batch_digest = ""
        mac_session = None
        if self.integrity:
            integrity_started = time.perf_counter()
            mac_session = RainMacSession(
                context=IntegrityContext(
                    session_id=self.session_id, round_id=round_id, batch_id=batch_id
                ),
                seed=seeds[6],
            )
            authenticated0, authenticated1 = mac_session.authenticate_client_shares(
                share0, share1, seed=seeds[7]
            )
            verified = mac_session.verify_client_batches(authenticated0, authenticated1)
            share0, share1 = verified.server0_shares, verified.server1_shares
            batch_digest = verified.digest.hex()
            integrity_seconds += time.perf_counter() - integrity_started

        self.preprocessing_cache.prepare(
            batch_size=updates.shape[0],
            dimension=updates.shape[1],
            round_id=round_id,
            batch_id=batch_id,
            server0_seed=seeds[2],
            server1_seed=seeds[3],
        )
        preprocessing = self.preprocessing_cache.consume(round_id=round_id, batch_id=batch_id)
        transport = LogicalTransport(
            transcript=mac_session.transcript if mac_session is not None else None
        )
        if mac_session is not None:
            transport.record_integrity_control(
                wire_bytes=verified.client_auth_bytes,
                phase=ProtocolPhase.CLIENT,
                messages=2 * updates.shape[0],
            )
            transport.record_integrity_control(
                wire_bytes=verified.batch_exchange_bytes,
                phase=ProtocolPhase.ONLINE,
                messages=2,
            )
        shuffled = run_secret_shared_shuffle(
            server0=preprocessing.server0,
            server1=preprocessing.server1,
            client_share0=share0,
            client_share1=share1,
            transport=transport,
            chunk_size=self.chunk_size,
            client_seconds=client_seconds,
            preprocessing_seconds=preprocessing.elapsed_seconds,
        )
        aggregate = secure_rain_aggregate(
            shuffled_share0=shuffled.server0_share,
            shuffled_share1=shuffled.server1_share,
            reference_bits=reference,
            tau=self.tau,
            round_id=round_id,
            batch_id=batch_id,
            seed=seeds[4],
            transport=transport,
            chunk_size=self.chunk_size,
        )
        # This is the sole reconstruction in the formal round path.
        direction = reconstruct_bits(
            aggregate.server0_direction_share, aggregate.server1_direction_share
        ).reshape(-1)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        communication = transport.metrics
        transcript_digest = ""
        if mac_session is not None:
            integrity_started = time.perf_counter()
            transcript_digest = mac_session.transcript.final_digest.hex()
            integrity_seconds += time.perf_counter() - integrity_started
        s0 = shuffled.computation.s0_seconds + aggregate.server0_seconds
        s1 = shuffled.computation.s1_seconds + aggregate.server1_seconds
        return RoundResult(
            direction_bits=direction,
            threshold_count=aggregate.threshold_count,
            metrics=RoundMetrics(
                client_comp_seconds=client_seconds,
                s0_comp_seconds=s0,
                s1_comp_seconds=s1,
                server_comp_sum_seconds=s0 + s1,
                server_comp_critical_seconds=max(s0, s1),
                offline_comp_seconds=shuffled.computation.offline_seconds + aggregate.offline_seconds,
                online_comp_seconds=(shuffled.computation.online_seconds + aggregate.elapsed_seconds
                                     - aggregate.offline_seconds),
                client_to_server_bytes=communication.client_to_server_bytes,
                server_to_server_bytes=communication.server_to_server_bytes,
                offline_bytes=communication.offline_bytes,
                online_bytes=communication.online_bytes,
                message_count=communication.message_count,
                peak_memory_bytes=peak,
                integrity_comp_seconds=integrity_seconds,
                integrity_bytes=communication.integrity_bytes,
                integrity_message_count=communication.integrity_message_count,
                integrity_verified=mac_session is not None,
                batch_digest=batch_digest,
                transcript_digest=transcript_digest,
            ),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "clip_norm": self.clip_norm,
            "noise_multiplier": self.noise_multiplier,
            "tau": self.tau,
            "chunk_size": self.chunk_size,
            "seed": self.seed,
            "integrity": self.integrity,
            "session_id": self.session_id,
        }
