import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("MorphogenesisEngine")

class MorphogenesisEngine:
    """
    ============================================================================
    DOCLOOM — MORPHOGENESIS ENGINE
    ============================================================================
    Growth framework that dynamically groups individual shards into high-level 
    Meta-Shards and Macro-Shards using physical Semantic Gravity attraction 
    and HDC Bipolar Vector Bundling.
    ============================================================================
    """

    def __init__(
        self,
        weaver: Any,
        resonance_engine: Optional[Any] = None,
        group_size: int = 20,
        min_attraction: float = 1e-6
    ):
        self.weaver = weaver
        self.resonance_engine = resonance_engine
        self.group_size = group_size
        self.min_attraction = min_attraction

    def grow(self) -> None:
        log_service(logger, "Running Morphogenesis growth cycle...", "info")

        # 1. Gather active data shards
        data_shards = [
            nid for nid, node in self.weaver.nodes.items()
            if node["t"] not in ["root", "meta_shard", "macro_shard"]
        ]
        if not data_shards:
            log_service(logger, "No data shards found to cluster.", "warning")
            return

        log_service(logger, f"Morphogenesis analyzing {len(data_shards)} base shards.", "info")

        # 2. Clean up existing meta_shard and macro_shard nodes and their contains edges
        old_nodes_to_remove = [
            nid for nid, node in self.weaver.nodes.items()
            if node["t"] in ["meta_shard", "macro_shard"]
        ]
        for nid in old_nodes_to_remove:
            self.weaver.nodes.pop(nid, None)
            if self.resonance_engine and nid in self.resonance_engine.field_nodes:
                self.resonance_engine.deregister_shard(nid)

        self.weaver.edges = [
            e for e in self.weaver.edges
            if e["r"] != "contains" and e["f"] not in old_nodes_to_remove and e["t"] not in old_nodes_to_remove
        ]

        # Ensure all data shards have np.ndarray format for emb
        for nid in data_shards:
            node = self.weaver.nodes[nid]
            emb = node["m"].get("emb")
            if emb is not None and not isinstance(emb, np.ndarray):
                node["m"]["emb"] = np.array(emb, dtype=np.int8)

        # 3. Compute pairwise attraction between all data shards
        n = len(data_shards)
        attraction_cache: Dict[Tuple[str, str], float] = {}
        
        for i in range(n):
            id_i = data_shards[i]
            node_i = self.weaver.nodes[id_i]
            v_i = node_i["m"].get("emb")
            mass_i = node_i["m"].get("mass", 1.0)
            if v_i is None:
                continue

            for j in range(i + 1, n):
                id_j = data_shards[j]
                node_j = self.weaver.nodes[id_j]
                v_j = node_j["m"].get("emb")
                mass_j = node_j["m"].get("mass", 1.0)
                if v_j is None:
                    continue

                # Hamming Distance: count of differing bipolar coordinates
                hamming_dist = float(np.count_nonzero(v_i != v_j))
                hamming_dist = max(1.0, hamming_dist)

                # Stability weight lookup from Resonance Engine if available
                stability = 0.1
                if self.resonance_engine:
                    stability_forward = self.resonance_engine.resonance_memory.get((id_i, id_j), {}).get("stability", 0.0)
                    stability_reverse = self.resonance_engine.resonance_memory.get((id_j, id_i), {}).get("stability", 0.0)
                    stability = max(0.1, stability_forward, stability_reverse)

                # Attraction formula
                attr = (mass_i * mass_j) / (hamming_dist ** 2) * stability
                key = tuple(sorted((id_i, id_j)))
                attraction_cache[key] = attr

        # 4. Agglomerative clustering based on average attraction
        clusters: List[List[str]] = [[nid] for nid in data_shards]
        target_clusters = max(1, n // self.group_size)

        while len(clusters) > target_clusters:
            # Find best merge pair
            best_pair = None
            max_attr = -1.0

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Compute average linkage attraction
                    total_attr = 0.0
                    count = 0
                    for id_a in clusters[i]:
                        for id_b in clusters[j]:
                            key = tuple(sorted((id_a, id_b)))
                            total_attr += attraction_cache.get(key, 0.0)
                            count += 1
                    
                    avg_attr = total_attr / count if count > 0 else 0.0
                    if avg_attr > max_attr:
                        max_attr = avg_attr
                        best_pair = (i, j)

            if best_pair is None or max_attr < self.min_attraction:
                break

            # Merge clusters
            idx_i, idx_j = best_pair
            clusters[idx_i].extend(clusters[idx_j])
            clusters.pop(idx_j)

        # 5. Form Meta-Shards & bundle HDC vectors
        meta_shard_nodes = []
        for idx, cluster in enumerate(clusters):
            # Sum vectors and take sign for HDC Superposition (Bundling)
            cluster_vectors = [self.weaver.nodes[nid]["m"]["emb"] for nid in cluster]
            summed = np.sum(cluster_vectors, axis=0)
            meta_vec = np.sign(summed).astype(np.int8)
            meta_vec[meta_vec == 0] = 1 # Force tie-breaker zero states to 1

            # Store Meta-Shard node in weaver
            meta_id = self.weaver._add_node(
                f"Meta-Shard {idx}",
                "meta_shard",
                {
                    "cluster_id": idx,
                    "emb": meta_vec.tolist(),
                    "mass": sum(self.weaver.nodes[nid]["m"].get("mass", 1.0) for nid in cluster)
                }
            )
            meta_shard_nodes.append(meta_id)

            # Connect Meta-Shard to member nodes
            for nid in cluster:
                self.weaver.connect(meta_id, nid, "contains")
                self.weaver.nodes[nid]["m"]["parent_meta_shard"] = meta_id

            # Register in Resonance Engine if active
            if self.resonance_engine:
                self.resonance_engine.register_shard(meta_id, meta_vec, meta={"type": "meta_shard"})

        log_service(logger, f"Formed {len(meta_shard_nodes)} Meta-Shards from base shards.", "info")

        # 6. Form Macro-Shards from Meta-Shards
        if len(meta_shard_nodes) <= 2:
            # Connect directly to Document Root
            for aid in meta_shard_nodes:
                self.weaver.connect(self.weaver.doc_root_id, aid, "contains")
            return

        # Perform clustering on Meta-Shards to build Macro-Shards
        macro_clusters: List[List[str]] = [[aid] for aid in meta_shard_nodes]
        target_macro_shards = max(1, len(meta_shard_nodes) // 10)

        # Pairwise attraction between Meta-Shards
        meta_attraction: Dict[Tuple[str, str], float] = {}
        for i in range(len(meta_shard_nodes)):
            aid_i = meta_shard_nodes[i]
            node_i = self.weaver.nodes[aid_i]
            v_i = np.array(node_i["m"]["emb"])
            mass_i = node_i["m"]["mass"]

            for j in range(i + 1, len(meta_shard_nodes)):
                aid_j = meta_shard_nodes[j]
                node_j = self.weaver.nodes[aid_j]
                v_j = np.array(node_j["m"]["emb"])
                mass_j = node_j["m"]["mass"]

                hamming_dist = float(np.count_nonzero(v_i != v_j))
                hamming_dist = max(1.0, hamming_dist)

                # Average stability between member elements of Meta-Shard i and Meta-Shard j
                stability = 0.1
                if self.resonance_engine:
                    members_i = [nid for nid, n in self.weaver.nodes.items() if n["m"].get("parent_meta_shard") == aid_i]
                    members_j = [nid for nid, n in self.weaver.nodes.items() if n["m"].get("parent_meta_shard") == aid_j]
                    total_s = 0.0
                    count = 0
                    for m_i in members_i:
                        for m_j in members_j:
                            s_fw = self.resonance_engine.resonance_memory.get((m_i, m_j), {}).get("stability", 0.0)
                            s_rv = self.resonance_engine.resonance_memory.get((m_j, m_i), {}).get("stability", 0.0)
                            total_s += max(s_fw, s_rv)
                            count += 1
                    if count > 0:
                        stability = max(0.1, total_s / count)

                attr = (mass_i * mass_j) / (hamming_dist ** 2) * stability
                meta_attraction[tuple(sorted((aid_i, aid_j)))] = attr

        while len(macro_clusters) > target_macro_shards:
            best_pair = None
            max_attr = -1.0

            for i in range(len(macro_clusters)):
                for j in range(i + 1, len(macro_clusters)):
                    total_attr = 0.0
                    count = 0
                    for aid_a in macro_clusters[i]:
                        for aid_b in macro_clusters[j]:
                            key = tuple(sorted((aid_a, aid_b)))
                            total_attr += meta_attraction.get(key, 0.0)
                            count += 1
                    avg_attr = total_attr / count if count > 0 else 0.0
                    if avg_attr > max_attr:
                        max_attr = avg_attr
                        best_pair = (i, j)

            if best_pair is None or max_attr < self.min_attraction:
                break

            idx_i, idx_j = best_pair
            macro_clusters[idx_i].extend(macro_clusters[idx_j])
            macro_clusters.pop(idx_j)

        # Create Macro-Shard nodes and connect
        for c_idx, cluster in enumerate(macro_clusters):
            cluster_vectors = [np.array(self.weaver.nodes[aid]["m"]["emb"]) for aid in cluster]
            summed = np.sum(cluster_vectors, axis=0)
            macro_vec = np.sign(summed).astype(np.int8)
            macro_vec[macro_vec == 0] = 1

            macro_id = self.weaver._add_node(
                f"Macro-Shard {c_idx}",
                "macro_shard",
                {
                    "cluster_id": c_idx,
                    "emb": macro_vec.tolist(),
                    "mass": sum(self.weaver.nodes[aid]["m"]["mass"] for aid in cluster)
                }
            )

            # Connect Document Root to Macro-Shard
            self.weaver.connect(self.weaver.doc_root_id, macro_id, "contains")

            # Connect Macro-Shard to constituent Meta-Shards
            for aid in cluster:
                self.weaver.connect(macro_id, aid, "contains")
                self.weaver.nodes[aid]["m"]["parent_macro_shard"] = macro_id

            # Register in Resonance Engine if active
            if self.resonance_engine:
                self.resonance_engine.register_shard(macro_id, macro_vec, meta={"type": "macro_shard"})

        log_service(logger, f"Formed {len(macro_clusters)} Macro-Shards from Meta-Shards.", "info")
