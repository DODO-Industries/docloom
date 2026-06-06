import hashlib
import numpy as np
from typing import Union

class UniverseSeedCore:
    """
    ============================================================================
    DOCLOOM — UNIVERSE SEED CORE (The Spacetime Metric)
    ============================================================================
    Fulfills the core seed-driven metric where a 64-bit integer key dictates
    the laws of geometry. Automatically expands into deterministic baseline
    coordinates for crystal regions (macro), cluster buckets (meso), and atomic 
    shards (micro) without disk reads.
    ============================================================================
    """
    def __init__(self, seed: int, dimension: int = 128):
        self.seed = seed & 0xFFFFFFFFFFFFFFFF  # Enforce 64-bit unsigned integer
        self.dimension = dimension

    def _get_rng_for_subseed(self, context: str) -> np.random.Generator:
        """
        Derives a stable, deterministic subseed from the root seed and context string.
        """
        hasher = hashlib.sha256()
        hasher.update(self.seed.to_bytes(8, byteorder="big"))
        hasher.update(context.encode("utf-8"))
        digest = hasher.digest()
        subseed = int.from_bytes(digest[:8], byteorder="big")
        return np.random.default_rng(subseed)

    def get_macro_offset(self, crystal_id: Union[str, int]) -> np.ndarray:
        """
        Macro-Level: Establishes default spatial positioning for a .loom crystal region.
        """
        rng = self._get_rng_for_subseed(f"macro_{crystal_id}")
        return rng.normal(0.0, 1.0, size=self.dimension).astype(np.float32)

    def get_meso_coordinates(self, bucket_idx: Union[str, int]) -> np.ndarray:
        """
        Meso-Level: Dictates baseline conceptual coordinates for cluster buckets.
        Returns a deterministic orthogonal baseline coordinate vector.
        """
        rng = self._get_rng_for_subseed(f"meso_{bucket_idx}")
        # Bipolar/orthogonal basis representation {-1, 1}
        return rng.choice([-1.0, 1.0], size=self.dimension).astype(np.float32)

    def get_micro_socket(self, shard_id: str) -> np.ndarray:
        """
        Micro-Level: Projects the "resting-state" target socket coordinate for an incoming atomic shard.
        """
        rng = self._get_rng_for_subseed(f"micro_{shard_id}")
        return rng.normal(0.0, 0.1, size=self.dimension).astype(np.float32)
