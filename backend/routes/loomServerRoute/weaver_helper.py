import os
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from backend.services.loom_service.cortex.embedding.transformer import EmbeddingTransformer
from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("RealtimeWeaver")

class RealtimeWeaver:
    def __init__(self):
        log_service(logger, "Initializing RealtimeWeaver...", "info")
        self.transformer = EmbeddingTransformer()

    def weave_into_state(self, all_pages, state, save_callback=None):
        """
        Decomposes document extraction data and directly ingests it into
        the running GlobalCognitiveState instance in real-time.
        """
        log_service(logger, "Initializing Real-time Field Ingestion...", "info")
        
        # 1. Structural Decomposition & Sharding
        all_shards = []
        for page in all_pages:
            pnum = page.get("page_number")
            for item in page.get("content", []):
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
                    all_shards.append({
                        "type": item.get("type"),
                        "text": item.get("text", f"[{item.get('type')}]"),
                        "origin_page": pnum,
                        "bbox": item.get("bbox"),
                        "font_size": item.get("font_size", 0),
                        "data": item.get("data"),
                        "base64_data": item.get("base64_data")
                    })

        if not all_shards:
            log_service(logger, "No shards extracted to weave.", "warning")
            return []

        # 2. Embedding & Semantic Decomposition
        log_service(logger, f"Generating Embeddings for {len(all_shards)} shards...", "info")
        shards_texts = [s["text"] for s in all_shards]
        embeddings = self.transformer.get_embeddings(shards_texts)
        
        # Ingest the shards directly into the global state
        for idx, shard in enumerate(all_shards):
            text = shard["text"]
            emb = np.array(embeddings[idx], dtype=np.float32)
            
            # Register embedding in the semantic field engine
            state.semantic_field.concept_embeddings[text] = emb
            
            # Initialize coordinates and velocities
            if text not in state.concept_coords:
                state.concept_coords[text] = emb.copy()
                state.concept_velocities[text] = np.zeros_like(emb)
                state.physics_tensors[text] = {
                    "energy": 1.0,
                    "entropy": 0.0,
                    "stability": 1.0,
                    "activation": 0.9,
                    "resonance": 0.0,
                    "momentum": 0.0,
                    "decay": 0.05,
                    "attention": 0.0
                }
            
            # Add to working memory graph
            try:
                state.memory.graph.add_node(
                    text,
                    activation=0.9,
                    stability=1.0,
                    temporal_depth=0
                )
            except Exception as e:
                logger.debug(f"Failed to add node to working memory graph: {e}")

        # 3. Establish Causal / Association Edges
        # Connect sequential/consecutive nodes to establish narrative flow
        log_service(logger, "Weaving sequential narrative trajectory links...", "info")
        for i in range(len(all_shards) - 1):
            source = all_shards[i]["text"]
            target = all_shards[i + 1]["text"]
            if source != target:
                try:
                    state.causal.causal_matrix.add_edge(
                        source, target, weight=0.8, frequency=5, last_seen=time.time()
                    )
                except Exception as e:
                    logger.debug(f"Failed to connect sequential edge: {e}")

        # Heading structural inheritance
        last_heading = None
        for shard in all_shards:
            if shard["type"] == "shard_heading":
                last_heading = shard["text"]
            elif last_heading:
                text = shard["text"]
                if last_heading != text:
                    try:
                        state.causal.causal_matrix.add_edge(
                            last_heading, text, weight=0.85, frequency=5, last_seen=time.time()
                        )
                    except Exception as e:
                        logger.debug(f"Failed to connect heading edge: {e}")

        # 4. Sparse Semantic Path Connections (Top-K similarity edges)
        log_service(logger, "Weaving cross-node semantic paths...", "info")
        num_nodes = len(all_shards)
        if num_nodes >= 2:
            search_k = min(num_nodes, 10)
            nn = NearestNeighbors(n_neighbors=search_k, mesh_sim=None, metric='cosine') if hasattr(NearestNeighbors, 'mesh_sim') else NearestNeighbors(n_neighbors=search_k, metric='cosine')
            nn.fit(embeddings)
            distances, indices = nn.kneighbors(embeddings)
            
            for i in range(num_nodes):
                node_a_text = all_shards[i]["text"]
                for idx_idx, neighbor_idx in enumerate(indices[i]):
                    j = int(neighbor_idx)
                    if i == j:
                        continue
                    node_b_text = all_shards[j]["text"]
                    cosine_sim = 1.0 - distances[i][idx_idx]
                    
                    if cosine_sim > 0.6:
                        # Add semantic similarity edge to the causal matrix
                        try:
                            state.causal.causal_matrix.add_edge(
                                node_a_text, node_b_text, weight=float(cosine_sim), frequency=3, last_seen=time.time()
                            )
                        except Exception as e:
                            logger.debug(f"Failed to connect semantic edge: {e}")

        # 5. Trigger an initial activation tick in the cognitive state to settle physics
        first_concept = all_shards[0]["text"]
        try:
            state.process_tick({first_concept: 0.95})
        except Exception as e:
            log_service(logger, f"Failed to run initial tick: {e}", "warning")

        # 6. Save the state back to .brain_data
        if save_callback:
            save_callback()
        
        log_service(logger, f"Weaver successfully ingested {len(all_shards)} thought cells into real-time field state.", "info")
        return all_shards
