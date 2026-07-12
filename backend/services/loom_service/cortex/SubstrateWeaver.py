import msgpack
import statistics
import hashlib
import re
import os
import math
import time
import numpy as np
from collections import Counter
from backend.config.envConfig import setup_logger, log_service
from backend.services.LLM_service.embedding.transformer import EmbeddingTransformer
from backend.services.loom_service.cortex.HyperVectorCreation import HyperVectorEngine
from backend.services.loom_service.cortex.latent_field_cognition.dynamic_field_substrate import DynamicFieldSubstrateEngine
from backend.services.LLM_service.embedding.embedding_manager import get_embeddings

logger = setup_logger("SubstrateWeaver")

class SubstrateWeaver:
    def __init__(self, node_limit=1000, storage_dir=None):
        # storage_dir is accepted (WeaveBrainCoordinator.orchestrate_weave
        # passes it) but unused here — SubstrateWeaver only computes the
        # in-memory graph; the coordinator does all physical persistence.
        self.nodes = {}
        self.edges = []
        self.concept_bridge = {}
        self.transformer = EmbeddingTransformer()
        self.hdc_engine = HyperVectorEngine(dimension=8000)
        self.doc_root_id = None
        self.macro_shard_ids = []
        self.node_limit = node_limit
        self.engine = DynamicFieldSubstrateEngine(max_nodes=10000, tau_collapse=5.0)
        self._init_graph()

    def _init_graph(self):
        self.doc_root_id = self._add_node("Substrate Root", "root", {"fixed": True})

    def _generate_stable_id(self, content, node_type, metadata=None):
        m = hashlib.blake2b(digest_size=12)
        m.update(str(content).encode('utf-8'))
        m.update(str(node_type).encode('utf-8'))
        return f"{node_type}_{m.hexdigest()}"

    def weave(self, data, output_path, incremental=False):
        if incremental:
            log_service(logger, "Running in INCREMENTAL Mode...", "info")
        log_service(logger, "Initializing Advanced Loom Weaving...", "info")
        
        all_shards = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    all_shards.append({"text": item, "type": "shard"})
                elif isinstance(item, dict):
                    txt = item.get("text") or item.get("content")
                    if txt:
                        all_shards.append({
                            "text": txt,
                            "type": item.get("type") or "shard",
                            "data": item.get("data"),
                            "binary": item.get("base64_data")
                        })
        
        for item in all_shards:
            meta = {
                "weight": 1.0,
                "activation_score": 0.0,
                "truth_score": 0.1
            }
            if "binary" in item and item["binary"]: meta["binary"] = item["binary"]
            if "data" in item and item["data"]: meta["table"] = item["data"]
            
            self._add_node(item["text"], item["type"], meta)
            
        log_service(logger, "Performing Semantic Enrichment...", "info")
        node_ids = list(self.nodes.keys())
        all_texts = [self.nodes[nid]["c"] for nid in node_ids]
        all_embeddings = get_embeddings(all_texts)
        
        for i, nid in enumerate(node_ids):
            content = self.nodes[nid]["c"]
            # Convert continuous float embedding to 8000-D bipolar {-1, 1} vector
            bipolar_vec = self.hdc_engine.get_semantic_basis_vector(content, all_embeddings[i])
            self.nodes[nid]["m"]["emb"] = np.array(bipolar_vec, dtype=np.int8).tolist()
            
            # Physics Calculation using vector length and embedding values
            emb_vec = all_embeddings[i]
            word_count = len(content.split())
            importance_factor = float(np.sum(np.abs(emb_vec)))
            mass = float(math.log(word_count + 1) * importance_factor)
            mass = max(0.1, mass)
            energy = float(np.mean(np.abs(emb_vec)))
            # Enforce deterministic phase calculation using embedding vector split
            phase = float((np.arctan2(np.sum(emb_vec[:len(emb_vec)//2]), np.sum(emb_vec[len(emb_vec)//2:])) + 2 * np.pi) % (2 * np.pi))

            self.nodes[nid]["m"]["mass"] = mass
            self.nodes[nid]["m"]["energy"] = energy
            self.nodes[nid]["m"]["prev_energy"] = 0.0
            self.nodes[nid]["m"]["phase"] = phase

            if self.nodes[nid]["t"] == "shard":
                concepts = self.transformer.extract_concepts(content)
                self.nodes[nid]["m"]["concepts"] = concepts
                self.nodes[nid]["m"]["entities"] = self.transformer.extract_entities(content)
                
                for concept in concepts:
                    c_hash = hashlib.md5(concept.lower().strip().encode('utf-8')).hexdigest()[:16]
                    if c_hash not in self.concept_bridge:
                        self.concept_bridge[c_hash] = []
                    self.concept_bridge[c_hash].append({"n": nid, "s": 1.0})

        # 1. Ingest baseline shards into dynamic substrate
        log_service(logger, "Ingesting shards into Dynamic Field Substrate...", "info")
        shard_ids = [nid for nid in node_ids if self.nodes[nid]["t"] not in ["root", "meta_shard", "macro_shard"]]
        for nid in shard_ids:
            orig_idx = node_ids.index(nid)
            self.engine.ingest(nid, all_embeddings[orig_idx])

        # 2. Stitch transitions chronologically between successive shards
        log_service(logger, "Stitching chronological transitions...", "info")
        for idx_sh in range(len(shard_ids) - 1):
            self.engine.record_transition(shard_ids[idx_sh], shard_ids[idx_sh + 1])

        # 3. Run physics simulation and density-based collapse synchronously
        log_service(logger, "Running real-time physics & gravitational collapse updates...", "info")
        for _ in range(10):  # Simulate 10 ticks (equivalent to ~1.0s at dt=0.1)
            self.engine.run_physics_step()

        # 4. Integrate spawned macro nodes and edges
        n = self.engine.current_node_count
        concept_ids = self.engine.concept_ids
        for i in range(n):
            node_id = concept_ids[i]
            if node_id not in self.nodes:
                label = self.engine.probe_node_label(i)
                mass = float(self.engine.physics_map[i, 0])
                phase = float(self.engine.physics_map[i, 1])
                activation = float(self.engine.physics_map[i, 2])
                coords = self.engine.coords_map[i].tolist()
                
                self.nodes[node_id] = {
                    "t": "macro_shard",
                    "c": label,
                    "m": {
                        "emb": self.engine.hdc_map[i].tolist(),
                        "mass": mass,
                        "phase": phase,
                        "activation": activation,
                        "coords": coords
                    }
                }
                self.connect(self.doc_root_id, node_id, "contains")

        # Map containment and causal entanglements
        for i in range(n):
            node_id = concept_ids[i]
            links = self.engine.causal_map[i]
            for link in links:
                target_idx = int(link[0])
                if target_idx != -1 and target_idx < n:
                    target_id = concept_ids[target_idx]
                    strength = float(link[1])
                    
                    if target_id.startswith("macro_") and not node_id.startswith("macro_"):
                        self.connect(target_id, node_id, "contains")
                        if "parent_macro_shard" not in self.nodes[node_id]["m"]:
                            self.nodes[node_id]["m"]["parent_macro_shard"] = target_id
                    elif node_id.startswith("macro_") and target_id.startswith("macro_"):
                        self.connect(node_id, target_id, "entanglement", score=strength)

        log_service(logger, "Cortex Weaving completed successfully. Returning graph in-memory.", "info")
        
        vector_map = {nid: all_embeddings[i] for i, nid in enumerate(node_ids)}
        
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "concept_bridge": self.concept_bridge,
            "all_embeddings": vector_map
        }

    def load_and_expand(self, shard_paths):
        log_service(logger, f"Expanding architecture: Loading {len(shard_paths)} existing shards...", "info")
        for path in shard_paths:
            with open(path, "rb") as f:
                bundle = msgpack.unpackb(f.read(), raw=False)
                incoming_nodes = bundle.get("g", {}).get("n", {})
                for nid, node in incoming_nodes.items():
                    if node["t"] == "root" and self.doc_root_id:
                        continue 
                    # Deserialization Cast: ensure list turns back into numpy int8 array
                    if "m" in node and "emb" in node["m"] and node["m"]["emb"] is not None:
                        node["m"]["emb"] = np.array(node["m"]["emb"], dtype=np.int8)
                    self.nodes[nid] = node
                
                self.edges.extend(bundle.get("g", {}).get("e", []))
        
        for nid, node in self.nodes.items():
            if node["t"] == "root":
                self.doc_root_id = nid
                break

    def _add_node(self, content, node_type, metadata=None):
        node_id = self._generate_stable_id(content, node_type, metadata)
        
        if node_id in self.nodes:
            if metadata:
                self.nodes[node_id]["m"].update(metadata)
            return node_id

        meta = metadata or {}
        
        # Initialize default baseline physical properties
        if "mass" not in meta:
            meta["mass"] = 1.0
        if "energy" not in meta:
            meta["energy"] = 0.0
        if "prev_energy" not in meta:
            meta["prev_energy"] = 0.0
        if "phase" not in meta:
            meta["phase"] = float(np.random.uniform(0.0, 2.0 * np.pi))

        self.nodes[node_id] = {
            "t": node_type,
            "c": content[:5000], 
            "m": meta
        }
        return node_id

    def connect(self, source_id, target_id, rel_type, score=1.0):
        self.edges.append({"f": source_id, "t": target_id, "r": rel_type, "s": float(score)})
