import msgpack
import statistics
import hashlib
import re
import os
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from backend.config.envConfig import setup_logger, log_service
from backend.test.legacy_loom.transformer import LoomTransformer

logger = setup_logger("LoomWeaver")

class LoomWeaver:
    def __init__(self, node_limit=1000):
        self.nodes = {}
        self.edges = []
        self.concept_bridge = {} # Concept Hash -> Node IDs
        self.transformer = LoomTransformer()
        self.doc_root_id = None
        self.constellation_ids = []
        self.node_limit = node_limit
        self._init_graph()

    def _init_graph(self):
        # Deterministic Root ID
        self.doc_root_id = self._add_node("Document Root", "root", {"fixed": True})

    def _generate_stable_id(self, content, node_type, metadata=None):
        """
        Implements Content-Addressable Hashing (Blake2b) for deduplication.
        Hash includes content, geometry (bbox), and font_size for Slot 0 mapping.
        """
        m = hashlib.blake2b(digest_size=12)
        m.update(str(content).encode('utf-8'))
        m.update(str(node_type).encode('utf-8'))
        
        if metadata:
            # Hash structural and spatial properties for deterministic uniqueness
            for key in ['bbox', 'font_size', 'num', 'col', 'cluster_id']:
                if key in metadata:
                    m.update(str(metadata[key]).encode('utf-8'))
        
        return f"{node_type}_{m.hexdigest()}"

    def _compress_bbox(self, bbox):
        if not bbox or not isinstance(bbox, list) or len(bbox) < 4:
            return None
        return [int(round(x)) for x in bbox]

    def weave(self, all_pages, output_path, incremental=False):
        """
        Transforms flat extraction data into a hierarchical, semantically 
        linked graph (DocLoom Architecture).
        """
        if incremental:
            log_service(logger, "Running in INCREMENTAL Mode...", "info")
            # We don't wipe self.nodes/edges if already loaded via load_and_expand
        log_service(logger, "Initializing Advanced Loom Weaving...", "info")
        
        # 1. Structural Weaving (Hierarchy Building)
        all_items = [item for page in all_pages for item in page.get("content", [])]
        font_sizes = [item.get("font_size") for item in all_items if item.get("font_size")]
        
        if not font_sizes:
            threshold = 14
        else:
            mode_size = Counter(font_sizes).most_common(1)[0][0]
            std_dev = statistics.stdev(font_sizes) if len(font_sizes) > 1 else 0
            threshold = mode_size + (1.5 * std_dev)

        # threshold = mode_size + (1.5 * std_dev)
        # current_hierarchy_stack = [self.doc_root_id]
        current_hierarchy_stack = [self.doc_root_id]

        # 1. Cognitive Weaving (Direct Conceptual Ingestion)
        # SECTION 10: Elimination of Pages - Pages exist only as metadata
        all_shards = []
        for page in all_pages:
            pnum = page.get("page_number")
            for item in page.get("content", []):
                # SECTION 3: Shard (Atomic Knowledge Unit)
                # Decompose paragraphs into atomic conceptual shards
                if item.get("type") in ["paragraph", "heading"]:
                    is_h = item.get("type") == "heading"
                    sub_texts = self.transformer.decompose(item.get("text", ""), is_heading=is_h)
                    for sub in sub_texts:
                        all_shards.append({
                            "type": "shard_heading" if item.get("type") == "heading" else "shard",
                            "text": sub,
                            "origin_page": pnum,
                            "bbox": item.get("bbox"),
                            "font_size": item.get("font_size", 0)
                        })
                else:
                    # Tables and images remain as specialized shards
                    all_shards.append({
                        "type": item.get("type"),
                        "text": item.get("text", f"[{item.get('type')}]"),
                        "origin_page": pnum,
                        "bbox": item.get("bbox"),
                        "font_size": item.get("font_size", 0),
                        "data": item.get("data"),
                        "base64_data": item.get("base64_data")
                    })
        
        # SECTION 11: Structural Inheritance (The "RAG-Killer")
        # Link headings to the content that follows them until the next heading of same/higher level
        last_heading_id = None
        current_shards = []

        # Hierarchy is now CONCEPTUAL: Root connects to Atlases (Clusters), which connect to Shards
        for item in all_shards:
            meta = {
                "bbox": self._compress_bbox(item.get("bbox")),
                "page": item.get("origin_page"),
                "weight": 1.0,
                "activation_score": 0.0,
                "truth_score": 0.1, # Base existence truth
                "font_size": item.get("font_size")
            }
            if "base64_data" in item: meta["binary"] = item["base64_data"]
            if "data" in item: meta["table"] = item["data"]
            
            node_id = self._add_node(item["text"], item["type"], meta)
            
            # Form Structural Edges
            if item["type"] == "shard_heading":
                last_heading_id = node_id
            elif last_heading_id:
                # Link every paragraph to its parent heading
                self.connect(last_heading_id, node_id, "structural_child", score=1.0)
            
            # Form standard hierarchy (shard -> root fallback)
            # We no longer connect to root here; _form_atlases will handle hierarchical mapping.

        # 2. Semantic Decomposition & Enrichment
        log_service(logger, "Performing Semantic Enrichment...", "info")
        node_ids = list(self.nodes.keys())
        all_texts = [self.nodes[nid]["c"] for nid in node_ids]
        all_embeddings = self.transformer.get_embeddings(all_texts)
        
        for i, nid in enumerate(node_ids):
            content = self.nodes[nid]["c"]
            self.nodes[nid]["m"]["emb"] = all_embeddings[i].tolist()
            if self.nodes[nid]["t"] in ["paragraph", "heading", "shard", "shard_heading"]:
                concepts = self.transformer.extract_concepts(content)
                self.nodes[nid]["m"]["concepts"] = concepts
                self.nodes[nid]["m"]["entities"] = self.transformer.extract_entities(content)
                
                # SECTION 9: Building the Concept Bridge
                for concept in concepts:
                    c_hash = hashlib.md5(concept.lower().strip().encode('utf-8')).hexdigest()[:16]
                    if c_hash not in self.concept_bridge:
                        self.concept_bridge[c_hash] = []
                    # Unified format: list of dicts for ranking
                    self.concept_bridge[c_hash].append({"n": nid, "s": 1.0})

        # 3. Sparse Semantic Linking (Top-K Edges)
        log_service(logger, "Calculating Semantic Path Growth...", "info")
        self._score_and_link_nodes(node_ids, all_embeddings)

        # 4. Atlas Clustering
        log_service(logger, "Forming Knowledge Atlases...", "info")
        atlas_nodes = self._form_atlases(node_ids, all_embeddings)

        # 5. Constellation Formation (Meta-level abstraction)
        log_service(logger, "Forming Knowledge Constellations...", "info")
        self._form_constellations(atlas_nodes)

        # 6. Multi-Shard Serialization
        self._save_shards(output_path)

    def _score_and_link_nodes(self, node_ids, embeddings, top_k=5, tau=0.6, search_k=20):
        """
        Implements the Edge Scoring Function.
        FIX: Use ANN (NearestNeighbors) to avoid O(N^2) complexity.
        """
        num_nodes = len(node_ids)
        if num_nodes < 2: return

        # 1. ANN Search for candidates
        nn = NearestNeighbors(n_neighbors=min(num_nodes, search_k), metric='cosine')
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)

        for i in range(num_nodes):
            scores = []
            node_a = self.nodes[node_ids[i]]
            
            # 2. Score only ANN candidates
            for idx in indices[i]:
                j = int(idx)
                if i == j: continue
                
                node_b = self.nodes[node_ids[j]]
                
                # Math implementation
                alpha, beta, gamma, delta = 0.2, 0.4, 0.2, 0.2
                
                jaccard = self.transformer.calculate_jaccard(
                    node_a["m"].get("concepts", []), 
                    node_b["m"].get("concepts", [])
                )
                cosine = 1.0 - distances[i][list(indices[i]).index(j)]
                struct = 1.0 if node_a["m"].get("page") == node_b["m"].get("page") else 0.0

                entity_overlap = self.transformer.calculate_jaccard(
                    node_a["m"].get("entities", []),
                    node_b["m"].get("entities", [])
                )
                
                score = (alpha * jaccard) + (beta * cosine) + (gamma * struct) + (delta * entity_overlap)
                
                if score > tau:
                    scores.append((node_ids[j], score, cosine))
            
            # 3. Degree Enforcement (Top-K)
            scores.sort(key=lambda x: x[1], reverse=True)
            for target_id, score, cosine in scores[:top_k]:
                node_b = self.nodes[target_id]
                
                # TRUTH RESOLUTION (Step 4 Fix: Cumulative support)
                support = score * 0.5 # Reduced multiplier for stability
                current_truth = node_b["m"].get("truth_score", 0.1)
                node_b["m"]["truth_score"] = float(current_truth + support)

                # Relationship Detection
                rel_type = "contextual_association"
                content_b = node_b["c"].lower()
                
                if any(c in content_b for c in ["because", "due to", "consequently", "therefore"]):
                    rel_type = "causality"
                elif any(d in content_b for d in ["requires", "depends", "essential for", "prerequisite"]):
                    rel_type = "dependency"
                elif cosine > 0.85:
                    rel_type = "similarity"
                
                self.connect(node_ids[i], target_id, rel_type, score)
                
                # FIX: Weighted Concept Bridge (Step 3: Optimized)
                for concept in node_b["m"].get("concepts", []):
                    c_hash = hashlib.md5(concept.lower().strip().encode('utf-8')).hexdigest()[:16]
                    if c_hash not in self.concept_bridge: self.concept_bridge[c_hash] = []
                    
                    # Store with score for ranking
                    self.concept_bridge[c_hash].append({"n": target_id, "s": float(score)})
                    
                    # Keep top 10 per concept for sanity & ranking
                    self.concept_bridge[c_hash].sort(key=lambda x: x["s"], reverse=True)
                    self.concept_bridge[c_hash] = self.concept_bridge[c_hash][:10]
                    
                    # Normalization (optional, since score is already normalized by tau)

    def _form_atlases(self, node_ids, embeddings):
        """Clusters CONTENT nodes into Atlases."""
        SEMANTIC_TYPES = {"shard", "shard_heading", "image", "table"}
        content_pairs = [
            (nid, emb)
            for nid, emb in zip(node_ids, embeddings)
            if self.nodes[nid]["t"] in SEMANTIC_TYPES
        ]
        if len(content_pairs) < 5:
            return []

        c_ids  = [p[0] for p in content_pairs]
        c_embs = [p[1] for p in content_pairs]

        num_clusters = max(1, len(c_ids) // 20)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(c_embs)

        atlas_nodes = []
        atlas_id_map = {}
        for i, cluster_id in enumerate(clusters):
            cluster_id = int(cluster_id)
            if cluster_id not in atlas_id_map:
                new_id = self._add_node(f"Atlas Cluster {cluster_id}", "atlas", {
                    "cluster_id": cluster_id,
                    "centroid":   kmeans.cluster_centers_[cluster_id].tolist()
                })
                atlas_id_map[cluster_id] = new_id
                atlas_nodes.append(new_id)

            atlas_id = atlas_id_map[cluster_id]
            self.connect(atlas_id, c_ids[i], "contains")
            # Set parent atlas in node metadata for edge partitioning
            self.nodes[c_ids[i]]["m"]["parent_atlas"] = atlas_id
        
        return atlas_nodes

    def _form_constellations(self, atlas_ids):
        """Groups Atlases into Constellations (Multi-level hierarchy)."""
        if not atlas_ids or len(atlas_ids) < 3:
            # If few atlases, connect directly to root
            for aid in atlas_ids:
                self.connect(self.doc_root_id, aid, "contains")
            return

        centroids = [self.nodes[aid]["m"]["centroid"] for aid in atlas_ids]
        num_const = max(1, len(atlas_ids) // 10)
        
        kmeans = KMeans(n_clusters=num_const, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(centroids)

        const_id_map = {}
        for i, cluster_id in enumerate(clusters):
            cluster_id = int(cluster_id)
            if cluster_id not in const_id_map:
                new_id = self._add_node(f"Constellation {cluster_id}", "constellation", {
                    "cluster_id": cluster_id,
                    "centroid": kmeans.cluster_centers_[cluster_id].tolist()
                })
                const_id_map[cluster_id] = new_id
                self.constellation_ids.append(new_id)
                # Connect root -> constellation
                self.connect(self.doc_root_id, new_id, "contains")

            const_id = const_id_map[cluster_id]
            self.connect(const_id, atlas_ids[i], "contains")


    def _save_shards(self, base_path):
        """
        Creates a master 'atlas.loom' architecture with partitioned edges.
        Step 1: same atlas -> shard, different atlas -> master
        """
        base_dir = os.path.dirname(base_path)
        base_name = os.path.basename(base_path).replace(".loom", "")
        
        # 1. Partition Data Nodes vs Structural Nodes
        data_nodes = {nid: n for nid, n in self.nodes.items() if n["t"] not in ["root", "atlas", "constellation"]}
        struct_nodes = {nid: n for nid, n in self.nodes.items() if n["t"] in ["root", "atlas", "constellation"]}
        
        node_ids = list(data_nodes.keys())
        total_shards = (len(node_ids) // self.node_limit) + 1
        
        log_service(logger, f"Architecture Expansion: Creating {total_shards} child shards + master atlas hub.", "info")
        
        # Map nodes to shards
        node_shard_map = {}
        for s in range(total_shards):
            shard_nid_subset = node_ids[s * self.node_limit : (s + 1) * self.node_limit]
            for nid in shard_nid_subset:
                node_shard_map[nid] = s

        # 2. Partition Edges: Local vs Cross-Atlas (Step 1 Fix)
        # same atlas -> shard, different atlas -> master
        shard_edges = {s: [] for s in range(total_shards)}
        master_edges = []
        
        for edge in self.edges:
            f_id, t_id = edge["f"], edge["t"]
            f_node = self.nodes.get(f_id)
            t_node = self.nodes.get(t_id)
            
            # If both are in the same shard, it's local
            if f_id in node_shard_map and t_id in node_shard_map and node_shard_map[f_id] == node_shard_map[t_id]:
                shard_edges[node_shard_map[f_id]].append(edge)
            else:
                # Cross-shard, Cross-atlas, or Structural edges go to master
                master_edges.append(edge)

        # 3. Save Child Shards (Data + Local Edges)
        for s in range(total_shards):
            shard_nid_subset = node_ids[s * self.node_limit : (s + 1) * self.node_limit]
            shard_data = {nid: data_nodes[nid] for nid in shard_nid_subset}
            
            shard_path = os.path.join(base_dir, f"{base_name}_shard_{s}.loom")
            with open(shard_path, "wb") as f:
                # Shards now contain their own local edges (Step 1)
                bundle = {
                    "v": "1.3", 
                    "n": shard_data,
                    "e": shard_edges[s] 
                }
                f.write(msgpack.packb(bundle, use_bin_type=True))
        
        # 4. Save Master Atlas
        atlas_bundle = {
            "v": "1.3",
            "type": "atlas_master",
            "map": node_shard_map,
            "bridge": self.concept_bridge,
            "g": {
                "n": struct_nodes, 
                "e": master_edges
            }
        }
        
        atlas_path = os.path.join(base_dir, f"atlas_{base_name}.loom")
        with open(atlas_path, "wb") as f:
            f.write(msgpack.packb(atlas_bundle, use_bin_type=True))
            
        log_service(logger, f"Master Atlas woven: {atlas_path}", "info")

    def load_and_expand(self, shard_paths):
        """Loads existing shards into memory to allow appending new knowledge."""
        log_service(logger, f"Expanding architecture: Loading {len(shard_paths)} existing shards...", "info")
        for path in shard_paths:
            with open(path, "rb") as f:
                bundle = msgpack.unpackb(f.read(), raw=False)
                # Merge nodes and edges, skipping root if already exists
                incoming_nodes = bundle.get("g", {}).get("n", {})
                for nid, node in incoming_nodes.items():
                    if node["t"] == "root" and self.doc_root_id:
                        continue 
                    self.nodes[nid] = node
                
                self.edges.extend(bundle.get("g", {}).get("e", []))
        
        # Re-identify root
        for nid, node in self.nodes.items():
            if node["t"] == "root":
                self.doc_root_id = nid
                break

    def _add_node(self, content, node_type, metadata=None):
        """Creates a node with a deterministic ID based on content and context."""
        node_id = self._generate_stable_id(content, node_type, metadata)
        
        # Deduplication check: If node exists, merge metadata instead of overwriting
        if node_id in self.nodes:
            self.nodes[node_id]["m"].update(metadata or {})
            return node_id

        self.nodes[node_id] = {
            "t": node_type,
            "c": content[:5000], 
            "m": metadata or {}
        }
        return node_id

    def connect(self, source_id, target_id, rel_type, score=1.0):
        self.edges.append({"f": source_id, "t": target_id, "r": rel_type, "s": float(score)})
