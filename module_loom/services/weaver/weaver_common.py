"""
DocLoom Weave Common Constants, Lookup Tables, and Low-Level Shard Functions.
Provides shared utilities, popcount lookup tables, and hashing functions across Weaver mixins.
"""
import hashlib
from functools import lru_cache
from typing import Dict, Any
import numpy as np

# Legacy loose files consolidated into universe.loom (moved to legacy_backup/ after migration)
LEGACY_STATE_FILES = [
    "coordinates.bin", "velocities.bin", "physics_tensors.bin", "brain_embeddings.bin",
    "working_memory.bin", "causal_links.bin", "metadata.bin", "replay.bin", "hdc_signatures.bin",
]

# Dead container segments that age out on universe.loom rebuilds
DEAD_CONTAINER_SEGMENTS = frozenset({
    "velocities.bin", "brain_embeddings.bin", "working_memory.bin",
    "metadata.bin", "replay.bin",
})

# Byte-wise popcount lookup table — used for Hamming distance on packed 128-bit leader signatures.
_BIT_COUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_distances(matrix: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Row-wise Hamming distance between a (L,16) packed-byte matrix and one signature."""
    if len(matrix) == 0:
        return np.empty(0, dtype=np.int64)
    return _BIT_COUNT_LUT[np.bitwise_xor(matrix, sig)].sum(axis=1).astype(np.int64)


@lru_cache(maxsize=1_048_576)
def hash_shard_id(shard_id: str) -> int:
    """Memoized SHA-256 64-bit integer hash of shard_id."""
    return int.from_bytes(hashlib.sha256(shard_id.encode("utf-8")).digest()[:8], byteorder="big")


def seed_phase(vector: np.ndarray) -> float:
    """
    Deterministic initial Kuramoto phase derived from the vector coordinates:
    theta_0 = arctan2( sum(v[:half]), sum(v[half:]) ) mod 2pi.
    """
    half = len(vector) // 2
    if half == 0:
        return 0.0
    a = float(np.sum(vector[:half]))
    b = float(np.sum(vector[half:]))
    return float((np.arctan2(a, b) + 2 * np.pi) % (2 * np.pi))


def np_msgpack_default(obj):
    """msgpack fallback for numpy scalars/arrays that slip into cognitive state."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Cannot serialize type {type(obj)} into cortex state")


def default_cognitive_fields() -> Dict[str, float]:
    """Default cognitive-physics fields carried by each ram_ledger entry."""
    return {
        "energy": 1.0,
        "entropy": 0.0,
        "stability": 1.0,
        "resonance": 0.0,
        "momentum": 0.0,
        "decay": 0.05,
        "attention": 0.0,
    }
