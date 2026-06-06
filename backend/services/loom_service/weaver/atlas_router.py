import os
import struct
import numpy as np
from typing import Dict, Any, Optional

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

    def route_vector(self, vector: np.ndarray) -> str:
        """
        Centroid Gravity Tracking: Routes a vector to the closest crystal path
        based on cosine similarity of their centroids.
        """
        if not self.crystals:
            raise ValueError("No crystals registered in the atlas.")

        v_norm = np.linalg.norm(vector)
        if v_norm == 0:
            return list(self.crystals.keys())[0]

        best_path = None
        best_sim = -2.0

        for path, info in self.crystals.items():
            centroid = info["centroid"]
            c_norm = np.linalg.norm(centroid)
            if c_norm == 0:
                sim = 0.0
            else:
                sim = float(np.dot(vector, centroid) / (v_norm * c_norm))

            if sim > best_sim:
                best_sim = sim
                best_path = path

        return best_path

    def update_centroid(self, path: str, vector: np.ndarray) -> None:
        """
        Updates the moving average centroid of a crystal with a new vector.
        """
        if path not in self.crystals:
            self.register_crystal(path, vector, state="hot")

        info = self.crystals[path]
        centroid = info["centroid"]
        n = info["num_shards"]

        # Calculate new moving average center of mass
        new_centroid = (centroid * n + vector) / (n + 1)
        info["centroid"] = new_centroid.astype(np.float32)
        info["num_shards"] = n + 1

    def save(self, path: Optional[str] = None) -> None:
        """
        Persists the lightweight atlas to disk in binary format.
        """
        save_path = path or self.atlas_path
        if not save_path:
            return

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

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(header)
            f.write(body)

    def load(self, path: Optional[str] = None) -> None:
        """
        Loads the atlas state from binary format, extracting the Universe Seed.
        """
        load_path = path or self.atlas_path
        if not load_path or not os.path.exists(load_path):
            return

        with open(load_path, "rb") as f:
            header_bytes = f.read(HEADER_SIZE)
            if len(header_bytes) < HEADER_SIZE:
                raise ValueError("Atlas file is too small to contain a valid header.")

            magic, _version, seed, num_crystals, vector_dim, _ = struct.unpack(
                HEADER_FORMAT,
                header_bytes
            )

            if magic != b"ATLS":
                raise ValueError(f"Invalid magic bytes in Atlas: {magic}")

            self.seed = seed
            self.crystals.clear()

            for _ in range(num_crystals):
                prefix_bytes = f.read(RECORD_PREFIX_SIZE)
                if len(prefix_bytes) < RECORD_PREFIX_SIZE:
                    raise ValueError("Truncated record prefix in Atlas file.")

                state_byte, num_shards, path_len = struct.unpack(RECORD_PREFIX_FORMAT, prefix_bytes)
                state = STATE_MAP_FROM_BYTE.get(state_byte, "warm")

                path_bytes = f.read(path_len)
                if len(path_bytes) < path_len:
                    raise ValueError("Truncated path string in Atlas file.")
                p = path_bytes.decode("utf-8")

                centroid_size = vector_dim * 4
                centroid_bytes = f.read(centroid_size)
                if len(centroid_bytes) < centroid_size:
                    raise ValueError("Truncated centroid vector in Atlas file.")
                centroid = np.frombuffer(centroid_bytes, dtype=np.float32).copy()

                self.crystals[p] = {
                    "state": state,
                    "centroid": centroid,
                    "num_shards": num_shards
                }
