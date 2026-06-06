import hashlib
import numpy as np
from typing import Dict, List, Tuple, Any

class CorticalRouter:
    """
    Handles Sparse Cortical Routing and locality bucketing using Hamming distance.
    Reduces complexity from O(N^2) to O(N * K).
    """

    def __init__(self, neighborhood_size: int = 32):
        self.neighborhood_size = neighborhood_size
        self.neighborhood_index: Dict[int, List[str]] = {}
        self.shard_bucket: Dict[str, int] = {}
        self.bucket_leaders: List[np.ndarray] = []

    def clear(self) -> None:
        self.neighborhood_index.clear()
        self.shard_bucket.clear()
        self.bucket_leaders.clear()

    def add_shard(self, shard_id: str, field_vec: np.ndarray) -> None:
        # Enforce slicing to first 128 dimensions and binarize to bipolar {-1, 1}
        sig = np.sign(field_vec[:128]).astype(np.int8)
        # Convert any 0s to 1s to guarantee pure bipolar {-1, 1}
        sig[sig == 0] = 1

        sig_bin = ((sig + 1) // 2).astype(np.uint8)

        best_bucket_idx = -1
        best_dist = 129

        for idx, leader in enumerate(self.bucket_leaders):
            leader_bin = ((leader + 1) // 2).astype(np.uint8)
            dist = np.bitwise_xor(sig_bin, leader_bin).sum()
            if dist < best_dist:
                best_dist = dist
                best_bucket_idx = idx

        # Threshold of 32 bits difference out of 128 (25% difference)
        if best_dist <= 32:
            bucket_id = best_bucket_idx
        else:
            self.bucket_leaders.append(sig)
            bucket_id = len(self.bucket_leaders) - 1
            self.neighborhood_index[bucket_id] = []

        self.shard_bucket[shard_id] = bucket_id
        self.neighborhood_index[bucket_id].append(shard_id)

    def remove_shard(self, shard_id: str) -> None:
        bucket_id = self.shard_bucket.pop(shard_id, None)
        if bucket_id is not None and bucket_id in self.neighborhood_index:
            try:
                self.neighborhood_index[bucket_id].remove(shard_id)
            except ValueError:
                pass

    def get_local_neighbors(self, shard_id: str, all_shards: List[str]) -> List[str]:
        bucket_id = self.shard_bucket.get(shard_id)
        if bucket_id is None:
            return [s for s in all_shards if s != shard_id]

        candidates = [
            s for s in self.neighborhood_index.get(bucket_id, [])
            if s != shard_id
        ]

        if len(candidates) < self.neighborhood_size:
            this_leader = self.bucket_leaders[bucket_id]
            this_leader_bin = ((this_leader + 1) // 2).astype(np.uint8)

            other_buckets = []
            for idx, leader in enumerate(self.bucket_leaders):
                if idx == bucket_id:
                    continue
                leader_bin = ((leader + 1) // 2).astype(np.uint8)
                dist = np.bitwise_xor(this_leader_bin, leader_bin).sum()
                other_buckets.append((idx, dist))

            other_buckets.sort(key=lambda x: x[1])

            for bid, _ in other_buckets:
                for s in self.neighborhood_index.get(bid, []):
                    if s != shard_id and s not in candidates:
                        candidates.append(s)
                if len(candidates) >= self.neighborhood_size:
                    break

        return candidates[:self.neighborhood_size]

    def get_local_neighbors_by_sig(self, query_sig: np.ndarray, all_shards: List[str]) -> List[str]:
        # Enforce shape/bipolar on query_sig
        sig = np.sign(query_sig[:128]).astype(np.int8)
        sig[sig == 0] = 1
        sig_bin = ((sig + 1) // 2).astype(np.uint8)

        bucket_dists = []
        for idx, leader in enumerate(self.bucket_leaders):
            leader_bin = ((leader + 1) // 2).astype(np.uint8)
            dist = np.bitwise_xor(sig_bin, leader_bin).sum()
            bucket_dists.append((idx, dist))

        bucket_dists.sort(key=lambda x: x[1])

        candidates = []
        for bid, _ in bucket_dists:
            for s in self.neighborhood_index.get(bid, []):
                if s not in candidates:
                    candidates.append(s)
            if len(candidates) >= self.neighborhood_size:
                break

        return candidates[:self.neighborhood_size]

    def rebucket_all(
        self,
        field_nodes: Dict[str, Any],
        resonance_memory: Dict[Tuple[str, str], Any],
        alpha: float = 0.8,
        beta: float = 0.2
    ) -> None:
        self.clear()
        for sid, node in field_nodes.items():
            fv = node["field_vec"]
            vec_sig = np.sign(fv[:128]).astype(np.float32)

            learned_sig = np.zeros(128, dtype=np.float32)
            neighbor_count = 0
            for (a, b), mem in resonance_memory.items():
                if sid not in (a, b) or mem.get("stability", 0.0) < 0.01:
                    continue
                other = b if a == sid else a
                if other in field_nodes:
                    learned_sig += np.sign(
                        field_nodes[other]["field_vec"][:128]
                    ).astype(np.float32) * mem.get("stability", 0.0)
                    neighbor_count += 1

            if neighbor_count > 0:
                learned_sig /= neighbor_count

            effective_sig_val = alpha * vec_sig + beta * learned_sig
            effective_sig = np.sign(effective_sig_val).astype(np.int8)
            effective_sig[effective_sig == 0] = 1

            sig_bin = ((effective_sig + 1) // 2).astype(np.uint8)

            best_bucket_idx = -1
            best_dist = 129
            for idx, leader in enumerate(self.bucket_leaders):
                leader_bin = ((leader + 1) // 2).astype(np.uint8)
                dist = np.bitwise_xor(sig_bin, leader_bin).sum()
                if dist < best_dist:
                    best_dist = dist
                    best_bucket_idx = idx

            if best_dist <= 32:
                bucket_id = best_bucket_idx
            else:
                self.bucket_leaders.append(effective_sig)
                bucket_id = len(self.bucket_leaders) - 1
                self.neighborhood_index[bucket_id] = []

            self.shard_bucket[sid] = bucket_id
            self.neighborhood_index[bucket_id].append(sid)
