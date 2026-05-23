import os
import numpy as np
import hashlib
from backend.test.legacy_loom.transformer import LoomTransformer
from backend.test.legacy_loom.navigator import LoomNavigator
from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("LoomServerService")

class LoomServerService:
    def __init__(self):
        log_service(logger, "Initializing LoomServerService (Loading Transformer)...", "info")
        self.transformer = LoomTransformer()
        log_service(logger, "LoomServerService ready.", "info")

    def process_text(self, text, extract_entities=True, extract_concepts=True, generate_embeddings=True):
        """Processes raw text into atomic semantic units with metadata."""
        units = self.transformer.decompose(text)
        results = []
        
        for unit in units:
            data = {"text": unit}
            if extract_entities:
                data["entities"] = self.transformer.extract_entities(unit)
            if extract_concepts:
                data["concepts"] = self.transformer.extract_concepts(unit)
            results.append(data)
            
        if generate_embeddings and results:
            texts = [r["text"] for r in results]
            embeddings = self.transformer.get_embeddings(texts)
            for i, emb in enumerate(embeddings):
                results[i]["embedding"] = emb.tolist()
                
        return results

    def get_query_embedding(self, query):
        """Generates an embedding for a search query."""
        embedding = self.transformer.get_embeddings([query])[0]
        return embedding.tolist()

    def activate_loom(self, viewer, query, depth=3, width=2):
        """
        SECTION 8: Hybrid Cognitive Retrieval (Neural Activation)
        Orchestrates Entry Point detection and Graph Activation.
        Step 2/4: Partial Loading & Activation Cache aware.
        """
        if isinstance(viewer, dict):
            # Fallback for old callers
            from backend.test.legacy_loom.viewer import LoomViewer
            v_obj = LoomViewer()
            v_obj.data = viewer
            viewer = v_obj

        log_service(logger, f"Activating Loom with query: '{query[:30]}...'", "info")
        
        loom_data = viewer.data
        nodes = loom_data.get("g", {}).get("n", {}) # Define nodes here
        navigator = LoomNavigator(loom_data, node_resolver=viewer.get_node)
        query_emb = np.array(self.get_query_embedding(query))
        
        # 1. Entry Point Detection (Hybrid: Concept Bridge + Hub Jumper)
        # SECTION 9: Concept-Aware Entry
        query_words = [w.lower().strip("?!.,") for w in query.split() if len(w) > 3]
        concept_hits = []
        for word in query_words:
            c_hash = hashlib.md5(word.encode('utf-8')).hexdigest()[:16]
            if c_hash in loom_data.get("bridge", {}):
                concept_hits.extend(navigator.concept_jump(c_hash))
        
        # Sort concept hits by score and pick best
        concept_hits.sort(key=lambda x: x["score"], reverse=True)
        best_node_id = concept_hits[0]["id"] if concept_hits else None
        
        if best_node_id:
            log_service(logger, f"Jumped via Concept Bridge to: {best_node_id}", "info")
            best_hub_id = best_node_id # We start activation from the exact node
            hub_sim = concept_hits[0]["score"]
        else:
            # Fallback to Hub Centroid Jump
            hub_centroids = {
                nid: node["m"]["centroid"] 
                for nid, node in nodes.items() 
                if node["t"] in ["constellation", "atlas"] and "centroid" in node["m"]
            }
            best_hub_id, hub_sim = navigator.heuristic_jump(query_emb, hub_centroids)
        
        if not best_hub_id:
            return {"success": False, "reason": "No entry point hub detected"}
            
        log_service(logger, f"Entry point established: {best_hub_id} (sim={hub_sim:.2f})", "info")

        # 2. Local Activation
        activation_path = navigator.beam_search(best_hub_id, beam_width=width, max_depth=depth)
        
        # 3. Context Assembly (Synthesis)
        context_units = []
        for nid, d, score in activation_path:
            node = viewer.get_node(nid) # Use viewer to handle lazy-load
            if node:
                context_units.append({
                    "id": nid,
                    "content": node["c"],
                    "score": score,
                    "depth": d,
                    "type": node["t"]
                })
        
        return {
            "success": True,
            "entry_point": best_hub_id,
            "similarity": hub_sim,
            "activation_path": context_units
        }
