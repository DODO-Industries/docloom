import msgpack
import statistics
import hashlib
import re
import os
import math
import numpy as np
from collections import Counter
from backend.config.envConfig import setup_logger, log_service
from backend.services.loom_service.cortex.embedding.transformer import EmbeddingTransformer
from backend.services.loom_service.cortex.HyperVectorCreation import HyperVectorEngine
from backend.services.loom_service.cortex.Morphogenesis import MorphogenesisEngine

logger = setup_logger("SubstrateWeaver")

class SubstrateWeaver:
    def __init__(self, node_limit=1000):
        self.nodes = {}
        self.edges = []
        self.concept_bridge = {}
        self.transformer = EmbeddingTransformer()
        self.hdc_engine = HyperVectorEngine(dimension=8000)
        self.doc_root_id = None
        self.macro_shard_ids = []
        self.node_limit = node_limit
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
        all_embeddings = self.transformer.get_embeddings(all_texts)
        
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

        log_service(logger, "Running Morphogenesis Evolution...", "info")
        morph = MorphogenesisEngine(self)
        morph.grow()

        self._save_shards(output_path)


    def _save_shards(self, base_path):
        base_dir = os.path.dirname(base_path)
        base_name = os.path.basename(base_path).replace(".loom", "")
        
        data_nodes = {nid: n for nid, n in self.nodes.items() if n["t"] not in ["root", "meta_shard", "macro_shard"]}
        struct_nodes = {nid: n for nid, n in self.nodes.items() if n["t"] in ["root", "meta_shard", "macro_shard"]}
        
        node_ids = list(data_nodes.keys())
        total_shards = (len(node_ids) // self.node_limit) + 1
        
        log_service(logger, f"Architecture Expansion: Creating {total_shards} child shards + master substrate hub.", "info")
        
        node_shard_map = {}
        for s in range(total_shards):
            shard_nid_subset = node_ids[s * self.node_limit : (s + 1) * self.node_limit]
            for nid in shard_nid_subset:
                node_shard_map[nid] = s

        shard_edges = {s: [] for s in range(total_shards)}
        master_edges = []
        
        for edge in self.edges:
            f_id, t_id = edge["f"], edge["t"]
            
            if f_id in node_shard_map and t_id in node_shard_map and node_shard_map[f_id] == node_shard_map[t_id]:
                shard_edges[node_shard_map[f_id]].append(edge)
            else:
                master_edges.append(edge)

        for s in range(total_shards):
            shard_nid_subset = node_ids[s * self.node_limit : (s + 1) * self.node_limit]
            shard_data = {nid: data_nodes[nid] for nid in shard_nid_subset}
            
            shard_path = os.path.join(base_dir, f"{base_name}_shard_{s}.loom")
            with open(shard_path, "wb") as f:
                bundle = {
                    "v": "1.3", 
                    "n": shard_data,
                    "e": shard_edges[s] 
                }
                f.write(msgpack.packb(bundle, use_bin_type=True))
        
        substrate_bundle = {
            "v": "1.3",
            "type": "substrate_master",
            "map": node_shard_map,
            "bridge": self.concept_bridge,
            "g": {
                "n": struct_nodes, 
                "e": master_edges
            }
        }
        
        substrate_path = os.path.join(base_dir, f"substrate_{base_name}.loom")
        with open(substrate_path, "wb") as f:
            f.write(msgpack.packb(substrate_bundle, use_bin_type=True))
            
        log_service(logger, f"Master Substrate woven: {substrate_path}", "info")

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
