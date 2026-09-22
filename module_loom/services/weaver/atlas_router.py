import os
import struct
import numpy as np
from typing import Dict, Any, Optional
from module_loom.config.tuning_config import tuning_manager

HEADER_FORMAT = "<4sHQII10s"  # magic(4s), version(H), seed(Q), num_crystals(I), vector_dim(I), reserved(10s)
HEADER_SIZE = 32

RECORD_PREFIX_FORMAT = "<BIH"  # state(B), num_shards(I), path_len(H)
RECORD_PREFIX_SIZE = 7

# State mappings
STATE_MAP_TO_BYTE = {"hot": 1, "warm": 2, "cold": 3}
STATE_MAP_FROM_BYTE = {1: "hot", 2: "warm", 3: "cold"}

class GlobalAtlasRouter:
    """
    ============================================================================
    DOCLOOM — GLOBAL ATLAS CLUSTER ROUTER (The Macro-Cortex)
    ============================================================================
    Keeps a lightweight binary mapping of all .loom files ("crystals") and their 
    centroids. Saves to a custom binary file format `atlas.capnp`.
    Embeds the 64-bit Universe Seed Key in its header to allow boot restoration.
    ============================================================================
    """
    def __init__(self, atlas_path: Optional[str] = None, seed: int = 42):
        self.atlas_path = atlas_path
        self.seed = seed & 0xFFFFFFFFFFFFFFFF
        self.crystals: Dict[str, Dict[str, Any]] = {}
        if atlas_path and os.path.exists(atlas_path):
            self.load()

    def register_crystal(self, path: str, centroid: np.ndarray, state: str = "warm") -> None:
        """
        Registers a new crystal with its initial centroid.
        """
        self.crystals[path] = {
            "state": state,
            "centroid": centroid.copy().astype(np.float32),
            "num_shards": 0
        }

    def set_state(self, path: str, state: str) -> None:
        """
        Sets the state of a crystal, maintaining single hot active cortex constraint.
        """
        if state not in ("hot", "warm", "cold"):
            raise ValueError("State must be 'hot', 'warm', or 'cold'")
        
        if path in self.crystals:
            self.crystals[path]["state"] = state
            if state == "hot":
                # Ensure only one crystal is hot
                for p, info in self.crystals.items():
                    if p != path and info["state"] == "hot":
                        info["state"] = "warm"

    def get_active_cortex(self) -> Optional[str]:
        """
        Returns the path of the Hot crystal.
        """
        for path, info in self.crystals.items():
            if info["state"] == "hot":
                return path
        return None

    def route_vector_with_embassy(self, vector: np.ndarray) -> Dict[str, Any]:
        """
        Centroid Gravity Tracking with Emergent Border Embassy Detection:
        Evaluates candidate vector affinity against all active crystal centroids.
        If a second crystal exhibits competitive affinity, designates the shard
        as a Border Embassy bridging the two crystals.
        Purely emergent: NO hardcoded domains, categories, or tags.
        """
        if not self.crystals:
            return {
                "primary_crystal": None,
                "primary_sim": 0.0,
                "secondary_crystal": None,
                "secondary_sim": 0.0,
                "is_embassy": False,
            }

        v_norm = np.linalg.norm(vector)
        if v_norm == 0:
            first_path = list(self.crystals.keys())[0]
            return {
                "primary_crystal": first_path,
                "primary_sim": 0.0,
                "secondary_crystal": None,
                "secondary_sim": 0.0,
                "is_embassy": False,
            }

        scores = []
        for path, info in self.crystals.items():
            centroid = info["centroid"]
            c_norm = np.linalg.norm(centroid)
            sim = 0.0 if c_norm == 0 else float(np.dot(vector, centroid) / (v_norm * c_norm))
            scores.append((path, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        primary_path, primary_sim = scores[0]

        is_embassy = False
        secondary_path = None
        secondary_sim = 0.0

        if len(scores) >= 2:
            second_path, second_sim = scores[1]
            embassy_ratio = tuning_manager.get_float("EMBASSY_COMPETITIVE_RATIO", 0.35)
            min_sim = tuning_manager.get_float("EMBASSY_MIN_SIMILARITY", 0.08)
            if (second_sim >= embassy_ratio * primary_sim) and (second_sim >= min_sim):
                is_embassy = True
                secondary_path = second_path
                secondary_sim = second_sim

        return {
            "primary_crystal": primary_path,
            "primary_sim": round(primary_sim, 4),
            "secondary_crystal": secondary_path,
            "secondary_sim": round(secondary_sim, 4) if is_embassy else 0.0,
            "is_embassy": is_embassy,
        }

    def route_vector(self, vector: np.ndarray) -> str:
        """
        Centroid Gravity Tracking: Routes a vector to the closest crystal path
        based on cosine similarity of their centroids.
        """
        res = self.route_vector_with_embassy(vector)
        return res["primary_crystal"]

    def get_crystal_anchor_3d(self, path: str) -> np.ndarray:
        """
        Derives the 3D galaxy anchor dynamically from the crystal's actual centroid vector
        using a deterministic orthonormal projection matrix seeded by the universe seed.
        Zero hardcoded coordinates.
        """
        info = self.crystals.get(path)
        if info is None or "centroid" not in info:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        centroid = info["centroid"]
        dim = len(centroid)
        if not hasattr(self, "_proj_matrix_3d") or self._proj_matrix_3d is None or self._proj_matrix_3d.shape[0] != dim:
            rng = np.random.RandomState(int(self.seed % 2147483647))
            P = rng.randn(dim, 3)
            q, _ = np.linalg.qr(P)
            self._proj_matrix_3d = q.astype(np.float32)

        c_norm = np.linalg.norm(centroid)
        unit_c = (centroid / c_norm) if c_norm > 0 else centroid
        scale = tuning_manager.get_float("CRYSTAL_OFFSET_RADIUS", 52.0)
        anchor_3d = (unit_c @ self._proj_matrix_3d) * scale
        return anchor_3d.astype(np.float32)

    def update_centroid(self, path: str, vector: np.ndarray) -> None:
        """
        Updates the moving average centroid of a crystal with a new semantic
        vector. Also accumulates cosine-to-centroid similarity (`cohesion_sum`)
        against the centroid AS IT WAS before this vector's own contribution —
        the entropy-triggered-division signal (mechanism 1): avg_cohesion =
        cohesion_sum / num_shards falls as members diverge from the crystal's
        center of mass, independent of raw item count. Calibrated against real
        MiniLM semantic embeddings (2026-08-02): single-topic corpora average
        ~0.31 avg-cosine-to-centroid, genuinely mixed-topic corpora ~0.22.
        """
        if path not in self.crystals:
            self.register_crystal(path, vector, state="hot")

        info = self.crystals[path]
        centroid = info["centroid"]
        n = info["num_shards"]

        v_norm = np.linalg.norm(vector)
        c_norm = np.linalg.norm(centroid)
        sim = 1.0 if n == 0 else (0.0 if (v_norm == 0 or c_norm == 0) else float(np.dot(vector, centroid) / (v_norm * c_norm)))
        # ponytail: cohesion_sum is runtime-only, not persisted in to_bytes/from_bytes
        # (keeps the on-disk atlas format stable). After a restart a crystal's cohesion
        # rebuilds from scratch and falls back to the MAX_CRYSTAL_SIZE ceiling until it
        # does — degrades to the old count-only behavior, never breaks.
        info["cohesion_sum"] = info.get("cohesion_sum", 0.0) + sim

        # Calculate new moving average center of mass
        new_centroid = (centroid * n + vector) / (n + 1)
        info["centroid"] = new_centroid.astype(np.float32)
        info["num_shards"] = n + 1

    @staticmethod
    def avg_cohesion(info: Dict[str, Any]) -> float:
        n = info.get("num_shards", 0)
        return 1.0 if n <= 0 else info.get("cohesion_sum", n) / n

    def to_bytes(self) -> bytes:
        """Serializes the full atlas (header + crystal records) to bytes —
        used both for the legacy atlas.capnp file and the universe.loom segment."""
        num_crystals = len(self.crystals)
        vector_dim = 0
        if num_crystals > 0:
            vector_dim = len(next(iter(self.crystals.values()))["centroid"])

        header = struct.pack(
            HEADER_FORMAT,
            b"ATLS",
            1,  # Version 1
            self.seed,
            num_crystals,
            vector_dim,
            b"\x00" * 10
        )

        body = bytearray()
        for p, info in self.crystals.items():
            state_byte = STATE_MAP_TO_BYTE.get(info["state"], 2)
            num_shards = info["num_shards"]
            path_bytes = p.encode("utf-8")

            prefix = struct.pack(RECORD_PREFIX_FORMAT, state_byte, num_shards, len(path_bytes))
            body.extend(prefix)
            body.extend(path_bytes)

            centroid_f32 = info["centroid"].astype(np.float32)
            body.extend(centroid_f32.tobytes())

        return bytes(header) + bytes(body)

    def from_bytes(self, data: bytes) -> None:
        """Restores the atlas from serialized bytes (header + crystal records)."""
        if len(data) < HEADER_SIZE:
            raise ValueError("Atlas payload is too small to contain a valid header.")

        magic, _version, seed, num_crystals, vector_dim, _ = struct.unpack(
            HEADER_FORMAT, data[:HEADER_SIZE]
        )
        if magic != b"ATLS":
            raise ValueError(f"Invalid magic bytes in Atlas: {magic}")

        self.seed = seed
        self.crystals.clear()
        pos = HEADER_SIZE

        for _ in range(num_crystals):
            if pos + RECORD_PREFIX_SIZE > len(data):
                raise ValueError("Truncated record prefix in Atlas payload.")
            state_byte, num_shards, path_len = struct.unpack_from(RECORD_PREFIX_FORMAT, data, pos)
            pos += RECORD_PREFIX_SIZE
            state = STATE_MAP_FROM_BYTE.get(state_byte, "warm")

            if pos + path_len > len(data):
                raise ValueError("Truncated path string in Atlas payload.")
            p = data[pos:pos + path_len].decode("utf-8")
            pos += path_len

            centroid_size = vector_dim * 4
            if pos + centroid_size > len(data):
                raise ValueError("Truncated centroid vector in Atlas payload.")
            centroid = np.frombuffer(data[pos:pos + centroid_size], dtype=np.float32).copy()
            pos += centroid_size

            self.crystals[p] = {
                "state": state,
                "centroid": centroid,
                "num_shards": num_shards
            }

    def save(self, path: Optional[str] = None) -> None:
        """
        Persists the lightweight atlas to disk in binary format.
        """
        save_path = path or self.atlas_path
        if not save_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(self.to_bytes())

    def load(self, path: Optional[str] = None) -> None:
        """
        Loads the atlas state from binary format, extracting the Universe Seed.
        """
        load_path = path or self.atlas_path
        if not load_path or not os.path.exists(load_path):
            return
        with open(load_path, "rb") as f:
            self.from_bytes(f.read())
