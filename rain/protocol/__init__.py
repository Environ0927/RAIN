"""Two-server protocol primitives for RAIN."""

from .boolean_sharing import reconstruct_bits, share_bits, validate_bits
from .arithmetic_sharing import decode_signed, encode_signed, reconstruct_arithmetic, share_arithmetic
from .party import Server0, Server0State, Server1, Server1State
from .preprocessing import IdealShufflePreprocessor, PreprocessedParties, ShufflePreprocessingCache
from .shuffle import ShuffleResult, run_secret_shared_shuffle
from .secure_engine import PrimitiveSession
from .aggregation import SecureAggregationResult, secure_rain_aggregate

__all__ = [
    "Server0",
    "Server0State",
    "Server1",
    "Server1State",
    "IdealShufflePreprocessor",
    "PreprocessedParties",
    "ShufflePreprocessingCache",
    "ShuffleResult",
    "reconstruct_bits",
    "run_secret_shared_shuffle",
    "share_bits",
    "validate_bits",
    "PrimitiveSession",
    "decode_signed",
    "encode_signed",
    "reconstruct_arithmetic",
    "share_arithmetic",
    "SecureAggregationResult",
    "secure_rain_aggregate",
]
