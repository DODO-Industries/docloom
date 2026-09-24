import os
import sys
import time
import hashlib
from typing import Any, Dict, List, Optional

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from module_loom.services.embedding.embedding_manager import get_embedding_model


def _align_vector(vec: np.ndarray, target_dim: int) -> np.ndarray:
    """Same alignment used by the testing/visualizer routes: truncate/pad the raw
    embedding to the coordinator's working dimension, then L2-normalize."""
    v = np.asarray(vec, dtype=np.float32)
    if len(v) > target_dim:
        v = v[:target_dim]
    elif len(v) < target_dim:
        v = np.pad(v, (0, target_dim - len(v)), constant_values=0.0)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v


class LoomMemory:
    """Bridge between the embedding model and the live WeaveBrainCoordinator —
    the read/write surface the AI pipeline talks to. Reuses the same coordinator
    singleton as the visualizer/testing routes so one process, one set of open
    file handles on the brain directory (WeaveBrainCoordinator is single-writer)."""

    def __init__(self, coordinator=None):
        if coordinator is None:
            from module_loom.routes.visualizer_routes import get_coordinator
            coordinator = get_coordinator()
        self.coordinator = coordinator
        self.model = get_embedding_model()
        import threading
        self._lock = threading.Lock()

    def embed(self, text: str) -> np.ndarray:
        raw = np.asarray(self.model.encode([text])[0], dtype=np.float32)
        return _align_vector(raw, self.coordinator.dimension)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        raw = np.asarray(self.model.encode(texts), dtype=np.float32)
        return [_align_vector(v, self.coordinator.dimension) for v in raw]

    def recall(self, text: str, top_k: int = 5, learn: bool = False) -> List[Dict[str, Any]]:
        vec = self.embed(text)
        with self._lock:
            return self.coordinator.recall(vec, top_k=top_k, learn=learn) or []

    def recall_distilled(self, text: str, candidate_pool: int = 16, top_k: int = 8,
                         learn: bool = False, candidate_k: Optional[int] = None,
                         session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        pool = candidate_k if candidate_k is not None else candidate_pool
        vec = self.embed(text)
        with self._lock:
            if hasattr(self.coordinator, "recall_multi_pass"):
                return self.coordinator.recall_multi_pass(
                    vec, candidate_pool=pool, top_k=top_k, learn=learn, session_id=session_id
                )
            return self.coordinator.recall(vec, top_k=top_k, learn=learn) or []



    def remember(self, text: str, metadata: Optional[Dict[str, Any]] = None, mass: float = 1.0,
                 vector: Optional[np.ndarray] = None) -> str:
        vec = vector if vector is not None else self.embed(text)
        shard_id = (metadata or {}).get("shard_id") or hashlib.sha256(
            f"{text}|{time.time_ns()}".encode("utf-8")
        ).hexdigest()[:24]
        with self._lock:
            self.coordinator.ingest_shard(shard_id, vec, text, mass=mass, metadata=metadata)
        return shard_id

    def get_shard_physics(self, shard_id: str) -> Dict[str, float]:
        """
        Per-shard physics state from the live ram_ledger (activation, energy,
        phase, momentum, stability, hits) — the "how alive is this memory
        right now" signal that plain-text RAG throws away. Used to build the
        richer memory-token input (roadmap Stage 1) and the training-data log.
        Returns {} if the shard isn't in the active ledger (e.g. cold/evicted).
        """
        with self._lock:
            entry = self.coordinator.ram_ledger.get(shard_id)
            if not entry:
                return {}
            return {
                "activation": float(entry.get("activation", 0.0)),
                "energy": float(entry.get("energy", 0.0)),
                "phase_angle": float(entry.get("phase_angle", 0.0)),
                "momentum": float(entry.get("momentum", 0.0)),
                "stability": float(entry.get("stability", 0.0)),
                "resonance": float(entry.get("resonance", 0.0)),
                "attention": float(entry.get("attention", 0.0)),
                "hits": int(entry.get("hits", 0)),
            }

    def stats(self) -> Dict[str, Any]:
        return {
            "crystals": len(self.coordinator.atlas.crystals),
            "dimension": self.coordinator.dimension,
            "storage_dir": self.coordinator.storage_dir,
        }
