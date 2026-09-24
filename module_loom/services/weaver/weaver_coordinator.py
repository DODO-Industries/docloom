"""
============================================================================
DOCLOOM — WEAVE BRAIN COORDINATOR (The Runtime Orchestrator)
============================================================================
Coordinates ingestion and recall loops across the seed core, atlas router,
physical substrate mmap layers, physics engines, and temporal memory loops.
Modularized with WeavePersistenceMixin, WeaveIngestMixin, WeaveRecallMixin,
and WeaveCognitiveMixin for high maintainability and clean separation of concerns.
============================================================================
"""
import os
import sys
import time
import struct
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import networkx as nx

from module_loom.config.tuning_config import tuning_manager
from module_loom.services.weaver.seed_core import UniverseSeedCore
from module_loom.services.weaver.atlas_router import GlobalAtlasRouter
from module_loom.services.weaver.substrate_layout import LoomStore
from module_loom.services.weaver.storage_physics import LatentFieldPhysicsEngine
from module_loom.services.weaver.memory_fluidity import DynamicMemoryFluidity
from module_loom.services.weaver.universe_container import UniverseContainer
from module_loom.services.cortex.latent_field_cognition.predictive_processing import CausalGraphs
from module_loom.services.cortex.latent_field_cognition.attractor_basin_compilation import (
    AssemblyCompilation, CognitiveAssembly,
)

# Re-export common symbols for full backward compatibility
from module_loom.services.weaver.weaver_common import (
    LEGACY_STATE_FILES, DEAD_CONTAINER_SEGMENTS, _BIT_COUNT_LUT,
    hamming_distances as _hamming_distances,
    hash_shard_id,
    seed_phase as _seed_phase,
    np_msgpack_default as _np_msgpack_default,
    default_cognitive_fields as _default_cognitive_fields,
)
from module_loom.services.weaver.weaver_persistence import WeavePersistenceMixin
from module_loom.services.weaver.weaver_ingest import WeaveIngestMixin
from module_loom.services.weaver.weaver_recall import WeaveRecallMixin
from module_loom.services.weaver.weaver_cognitive import WeaveCognitiveMixin


class WeaveBrainCoordinator(
    WeavePersistenceMixin,
    WeaveIngestMixin,
    WeaveRecallMixin,
    WeaveCognitiveMixin
):
    """
    Unified Runtime Orchestrator integrating persistence, ingestion,
    recall, and cognitive dynamics across the living memory substrate.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        storage_dir: str = "",
        dimension: int = 128,
        lambda_base: float = 0.05
    ):
        from module_loom.config.env_config import BRAIN_STORAGE_DIR, EMBEDDING_DIMENSION
        if dimension == 128:
            dimension = EMBEDDING_DIMENSION
        self.dimension = dimension
        if not storage_dir:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            configured = os.path.join(project_root, BRAIN_STORAGE_DIR)
            preferred = os.path.join(project_root, "assets", ".brain_data")
            storage_dir = configured if os.path.isdir(configured) else (preferred if os.path.isdir(preferred) else os.path.join(project_root, ".brain_data"))
        self.storage_dir = os.path.abspath(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)

        self.atlas_path = os.path.join(self.storage_dir, "atlas.capnp")
        self.metric_path = os.path.join(self.storage_dir, "universe.metric")
        self.universe_path = os.path.join(self.storage_dir, "universe.loom")
        self.max_shards_per_crystal = tuning_manager.get_int("MAX_CRYSTAL_SIZE", 500000)
        self.universe: Optional[UniverseContainer] = None
        self._atlas_dirty = False

        # Mechanism 4: auditable tombstone log
        self._tombstone_path = os.path.join(self.storage_dir, "tombstones.jsonl")
        self.tombstones: List[Dict[str, Any]] = []
        if os.path.exists(self._tombstone_path):
            try:
                with open(self._tombstone_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.tombstones.append(json.loads(line))
                self.tombstones = self.tombstones[-2000:]
            except Exception:
                pass

        if UniverseContainer.exists(self.universe_path):
            self.universe = UniverseContainer(self.universe_path)
            self.seed = self.universe.seed
            self.dimension = self.universe.dimension
            self.atlas = GlobalAtlasRouter(seed=self.seed)
            atlas_blob = self.universe.get_segment("atlas")
            if atlas_blob:
                self.atlas.from_bytes(atlas_blob)
            self._normalize_crystal_paths()
        else:
            metric_loaded = False
            if os.path.exists(self.metric_path):
                try:
                    with open(self.metric_path, "rb") as f:
                        metric_bytes = f.read(14)
                    if len(metric_bytes) == 14:
                        magic, dim, metric_seed = struct.unpack("<4sHQ", metric_bytes)
                        if magic == b"LOOM":
                            self.seed = metric_seed
                            self.dimension = dim
                            metric_loaded = True
                except Exception:
                    pass

            if not metric_loaded:
                if os.path.exists(self.atlas_path):
                    self.atlas = GlobalAtlasRouter(atlas_path=self.atlas_path)
                    self.seed = self.atlas.seed
                else:
                    if seed is None:
                        from module_loom.config.env_config import MASTER_SEED
                        self.seed = MASTER_SEED
                    else:
                        self.seed = seed & 0xFFFFFFFFFFFFFFFF

            self.atlas = GlobalAtlasRouter(atlas_path=self.atlas_path, seed=self.seed)
            self._normalize_crystal_paths()
            self._consolidate_universe()

        self.seed_core = UniverseSeedCore(seed=self.seed, dimension=self.dimension)
        self.physics = LatentFieldPhysicsEngine(dimension=self.dimension)
        if lambda_base == 0.05:
            lambda_base_val = tuning_manager.get_float("MEMORY_DECAY_RATE", lambda_base)
        else:
            lambda_base_val = lambda_base
        self.fluidity = DynamicMemoryFluidity(lambda_base=lambda_base_val)

        self._stores: Dict[str, LoomStore] = {}
        self.ram_ledger: Dict[str, Dict[str, Any]] = {}

        # Shared cognition state
        self.causal = CausalGraphs()
        self.causal_graph: "nx.DiGraph" = self.causal.causal_matrix
        self.compiler = AssemblyCompilation()
        self.active_assemblies: List["CognitiveAssembly"] = []
        self.latent_field: Optional[np.ndarray] = None
        self.working_latent: Optional[np.ndarray] = None
        self.latent_velocity: Optional[np.ndarray] = None
        self.cognitive_entropy: float = 0.5
        self.cognitive_coherence: float = 0.5
        self.cognitive_energy_budget: float = 1.0
        self.cognitive_tick: int = 0
        self._last_cognitive_coherence: float = 0.5
        self._last_ingested_sid: Optional[str] = None
        self._last_adequacy_score: float = 1.0
        self._last_hop_triggered: bool = False
        self.working_set_pairs: Dict[Tuple[str, str], int] = {}
        self._last_full_save_ts: float = 0.0
        self._closed: bool = False

        self.superseded_by_crystal: Dict[str, Dict[str, str]] = {}
        self._superseded_idx_cache: Dict[str, np.ndarray] = {}
        self._last_message_ts: float = time.time()
        self._shards_since_last_sleep: int = 0
        self._last_sleep_ts: float = 0.0

        self._recall_traces: List[Tuple[List[str], List[float], float]] = []
        self.sessions: Dict[str, Dict[str, Any]] = {}

        self._load_persistence_layers()
        try:
            self.load_cortex_state()
        except Exception:
            pass

    def _get_store(self, path: str) -> LoomStore:
        """Cached, reusable LoomStore handle per crystal file (O(1) mmap lookup)."""
        store = self._stores.get(path)
        if store is None:
            store = LoomStore(path, vector_dim=self.dimension, seed=self.seed, writable=True)
            self._stores[path] = store
        return store

    def _normalize_crystal_paths(self) -> None:
        """Normalizes crystal paths stored in atlas to be relative to current storage_dir."""
        normalized = {}
        for old_path, info in self.atlas.crystals.items():
            base_name = os.path.basename(old_path)
            new_path = os.path.join(self.storage_dir, base_name)
            normalized[new_path] = info
        self.atlas.crystals = normalized

    def close(self) -> None:
        """Safe shutdown: flushes pending traces, closes stores, and saves cortex state."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self.consolidate_traces()
        except Exception:
            pass
        for store in list(self._stores.values()):
            try:
                store.close()
            except Exception:
                pass
        self._stores = {}
        self.checkpoint()
        try:
            self.save_cortex_state()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            if getattr(self, "_closed", False):
                return
            if sys is None or getattr(sys, "meta_path", None) is None:
                return
            self.close()
        except Exception:
            pass
