import os
import sys
import time
import json
import numpy as np
import networkx as nx
from typing import Any, Optional, Dict, List

# Resolve Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

RESULTS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "backend", "results"))

from backend.routes.loomServerRoute.state_manager import (
    get_cognition_state,
    set_cognition_state,
    get_shard_groups,
    set_shard_groups,
    load_brain_file,
    init_cognition_engine,
    save_brain_file,
    auto_cluster_shard_groups,
    project_concepts_to_3d
)

from backend.services.loom_service.cognition.attractor_dynamics import get_embedding_model, safe_normalize

def query_cognition_memory_algorithm(query: str, top_k: int = 5) -> dict:
    state = get_cognition_state()
    if state is None:
        if not load_brain_file():
            init_cognition_engine()
        save_brain_file()
        state = get_cognition_state()
        
    q = query.strip()
    if not q:
        raise ValueError("Query cannot be empty")
        
    # Embed the query using the shared model (imported from attractor_dynamics)
    model = get_embedding_model()
    q_vec = safe_normalize(np.array(model.encode(q), dtype=np.float32))
    
    # 1. Compute cosine similarity with all dynamic concept coordinates
    sims = {}
    for concept, vec in state.concept_coords.items():
        sims[concept] = float(np.dot(q_vec, vec))
        
    if not sims:
        return {
            "success": True,
            "query": q,
            "query_coords": [0.0, 0.0, 0.0],
            "matches": []
        }

    # 2. Wave Propagation & Field Excitation Search
    node_energies = {c: max(0.0, sim) for c, sim in sims.items()}
    
    # Select initial seed concepts (top 3 highest initial energy)
    sorted_seeds = sorted(node_energies.items(), key=lambda x: x[1], reverse=True)
    seed_count = min(3, len(sorted_seeds))
    seeds = [c for c, eng in sorted_seeds[:seed_count]]
    
    G = state.causal.causal_matrix
    
    # Accumulator for total energy received by each node during propagation
    accumulated_resonance = {c: 0.0 for c in sims}
    paths = {c: [c] for c in seeds}
    
    # 5 steps of wave propagation
    for step in range(5):
        next_energies = {c: 0.0 for c in sims}
        for u, energy in node_energies.items():
            if energy < 0.05:
                continue
            
            accumulated_resonance[u] += energy
            
            # Propagate to causal neighbors
            if u in G:
                edges = G.edges(u, data=True)
                for _, v, data in edges:
                    if v not in sims:
                        continue
                    weight = data.get("weight", 0.5)
                    # Energy transfer is proportional to causal edge weight and propagation coefficient
                    transfer = energy * weight * 0.45
                    next_energies[v] += transfer
                    
                    # Record path flow
                    if u in paths and (v not in paths or len(paths[v]) > len(paths[u]) + 1):
                        paths[v] = paths[u] + [v]
                        
        # Self-excitation in stable attractors (they sustain energy better)
        for c in sims:
            stability = state.physics_tensors.get(c, {}).get("stability", 0.5)
            attractor_resonance = node_energies[c] * stability * 0.15
            next_energies[c] += attractor_resonance
            
        # Apply global decay/leakage to suppress noise
        node_energies = {c: next_energies[c] * 0.75 for c in sims}
        
    # 3. Finalize resonance scores
    results = []
    for concept in sims.keys():
        sim = sims[concept]
        resonance = accumulated_resonance.get(concept, 0.0)
        # Blend direct semantic similarity with propagation resonance
        score = sim * 0.35 + resonance * 0.65
        path = paths.get(concept, [])
        source_seed = path[0] if path else None
        
        results.append({
            "concept": concept,
            "similarity": sim,
            "score": score,
            "path": path,
            "source_seed": source_seed
        })
        
    # Sort by final graph-decayed score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    top_matches = results[:top_k]
    
    # Excite the brain state: recall boosts activation/energy and reinforces traversed causal paths
    for match in top_matches:
        concept = match["concept"]
        score = match["score"]
        
        # Boost activation in physics tensors
        if concept in state.physics_tensors:
            tensors = state.physics_tensors[concept]
            tensors["activation"] = min(1.0, tensors.get("activation", 0.5) + score * 0.20)
            tensors["energy"] = min(1.0, tensors.get("energy", 0.5) + score * 0.15)
            tensors["stability"] = min(1.0, tensors.get("stability", 0.5) + 0.05)
            
        # Boost in working memory if present
        if state.memory.graph.has_node(concept):
            state.memory.graph.nodes[concept]["activation"] = min(
                1.0, state.memory.graph.nodes[concept]["activation"] + score * 0.20
            )
            
        # Strengthen causal paths traversed during propagation
        path = match["path"]
        if len(path) > 1:
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                if state.causal.causal_matrix.has_edge(u, v):
                    edge_data = state.causal.causal_matrix[u][v]
                    edge_data["weight"] = min(1.0, edge_data.get("weight", 0.5) + 0.06)
                    edge_data["frequency"] = edge_data.get("frequency", 1) + 1
                    
    save_brain_file()
    
    # Project query vector to 3D space using saved PCA parameters
    q_proj_3d = [0.0, 0.0, 0.0]
    pca_mean = getattr(state, "_pca_mean", None)
    pca_vectors = getattr(state, "_pca_vectors", None)
    pca_scale = getattr(state, "_pca_scale", None)
    
    if pca_mean is not None and pca_vectors is not None and pca_scale is not None:
        centered_q = q_vec - pca_mean
        q_p = np.dot(centered_q, pca_vectors)
        q_proj_3d = [
            float(q_p[0]) * pca_scale,
            float(q_p[1]) * pca_scale,
            float(q_p[2]) * pca_scale
        ]
        
    return {
        "success": True,
        "query": q,
        "query_coords": q_proj_3d,
        "matches": top_matches
    }

def get_serialized_cognition_state_helper() -> dict:
    state = get_cognition_state()
    shard_groups = get_shard_groups()
    if state is None:
        init_cognition_engine()
        save_brain_file()
        state = get_cognition_state()
        
    # Project concepts to 3D via PCA
    coords, lf_proj, wl_proj = project_concepts_to_3d(state.concept_coords)
    
    # Gather active assemblies
    assemblies = []
    for assy in state.active_assemblies:
        assemblies.append({
            "id": assy.assembly_id,
            "dominant": assy.dominant_concept,
            "activations": assy.node_activations,
            "confidence": assy.confidence,
            "stability": assy.stability,
            "basin_strength": assy.meta_data.get("basin_strength", 0.0)
        })
        
    # Gather working memory graph
    wm_nodes = []
    wm_edges = []
    for node, data in state.memory.graph.nodes(data=True):
        wm_nodes.append({
            "id": node,
            "activation": float(data.get("activation", 0.0)),
            "stability": float(data.get("stability", 0.0)),
            "temporal_depth": int(data.get("temporal_depth", 0))
        })
    for u, v, data in state.memory.graph.edges(data=True):
        wm_edges.append({
            "from": u,
            "to": v,
            "weight": float(data.get("weight", 0.0))
        })
        
    # Gather meta shards
    meta_shards = []
    for meta_id, shard in state.meta_shards.meta_store.items():
        meta_shards.append({
            "id": meta_id,
            "label": shard.emergent_label,
            "children": shard.children,
            "stability": float(shard.stability),
            "abstraction_level": int(shard.abstraction_level)
        })
        
    # Gather causal paths
    causal_links = []
    for u, v, data in state.causal.causal_matrix.edges(data=True):
        causal_links.append({
            "from": u,
            "to": v,
            "weight": float(data.get("weight", 0.0)),
            "frequency": int(data.get("frequency", 1))
        })
        
    return {
        "success": True,
        "tick": state.tick,
        "metrics": {
            "entropy": float(state.entropy),
            "coherence": float(state.coherence),
            "surprise": float(state.surprise),
            "energy": float(state.energy_budget),
            "pressure": float(state.pressure),
            "mode": str(state.reflection.cognitive_mode)
        },
        "concepts_3d": coords,
        "latent_field_3d": lf_proj,
        "working_latent_3d": wl_proj,
        "active_assemblies": assemblies,
        "working_memory": {
            "nodes": wm_nodes,
            "edges": wm_edges
        },
        "attention_focus": list(state.attention_focus),
        "meta_shards": meta_shards,
        "causal_links": causal_links,
        "shard_groups": shard_groups,
        "concept_thoughts": getattr(state, '_concept_thoughts', {}),
        "physics_tensors": state.physics_tensors,
        "pipeline": {
            "status": "idle",
            "reason": None,
            "decision": {}
        }
    }

def clean_text(x):
    if x is None: return None
    if not isinstance(x, str): x = str(x)
    x = x.strip()
    return x if x else None

def extract_strings(obj):
    results = []
    if isinstance(obj, str):
        txt = clean_text(obj)
        if txt: results.append(txt)
    elif isinstance(obj, list):
        for item in obj: results.extend(extract_strings(item))
    elif isinstance(obj, dict):
        preferred_keys = ["text", "thought", "thoughts", "content", "sentence", "idea", "concept", "description", "message", "value"]
        for key in preferred_keys:
            if key in obj: results.extend(extract_strings(obj[key]))
        for k, v in obj.items():
            if k.lower() in ["id", "uuid", "timestamp", "created_at", "updated_at"]: continue
            results.extend(extract_strings(v))
    return results

def inject_single_shard_algorithm(shard: Any, prev_concept: Optional[str]) -> dict:
    state = get_cognition_state()
    shard_groups = get_shard_groups()
    
    if state is None:
        if not load_brain_file():
            init_cognition_engine()
            state = get_cognition_state()
            
    if not hasattr(state, '_concept_thoughts') or state._concept_thoughts is None:
        state._concept_thoughts = {}
        
    all_concepts = []
    seen_concepts = set(state.semantic_field.concept_embeddings.keys())
    
    def register_concept(concept, activation=0.9, stability=0.9, temporal_depth=0):
        concept = clean_text(concept)
        if not concept: return
        if concept in seen_concepts: return
        seen_concepts.add(concept)
        try: state.semantic_field.register_concept(concept)
        except: pass
        try:
            state.memory.graph.add_node(
                concept, activation=activation, stability=stability, temporal_depth=temporal_depth
            )
        except: pass
        all_concepts.append(concept)
        
    def connect(a, b, weight=0.7, frequency=3):
        a = clean_text(a); b = clean_text(b)
        if not a or not b or a == b: return
        try:
            state.causal.causal_matrix.add_edge(
                a, b, weight=weight, frequency=frequency, last_seen=time.time()
            )
        except: pass

    def process_cluster(cluster_name, thoughts):
        if not thoughts: return
        cleaned = []
        for t in thoughts:
            t = clean_text(t)
            if not t: continue
            cleaned.append(t)
            register_concept(t)
        if cluster_name:
            state._concept_thoughts[cluster_name] = cleaned
        for i in range(len(cleaned) - 1):
            connect(cleaned[i], cleaned[i + 1], weight=0.85, frequency=5)

    # Decode shard formats
    if isinstance(shard, str):
        register_concept(shard)
    elif isinstance(shard, dict):
        if "label" in shard and "thoughts" in shard:
            cluster_name = clean_text(shard.get("label"))
            thoughts = extract_strings(shard.get("thoughts", []))
            process_cluster(cluster_name, thoughts)
        elif "text" in shard and "thoughts" in shard:
            concept = clean_text(shard.get("text"))
            thoughts = extract_strings(shard.get("thoughts", []))
            if concept:
                register_concept(concept)
                state._concept_thoughts[concept] = thoughts
            for t in thoughts: register_concept(t)
            for t in thoughts: connect(concept, t, weight=0.9, frequency=5)
            for i in range(len(thoughts) - 1):
                connect(thoughts[i], thoughts[i + 1], weight=0.8, frequency=4)
        else:
            extracted = extract_strings(shard)
            for text in extracted: register_concept(text)
            for i in range(len(extracted) - 1):
                connect(extracted[i], extracted[i + 1], weight=0.55, frequency=2)
    elif isinstance(shard, list):
        for item in shard:
            if isinstance(item, str):
                register_concept(item)
            else:
                extracted = extract_strings(item)
                for text in extracted: register_concept(text)
                for i in range(len(extracted) - 1):
                    connect(extracted[i], extracted[i + 1], weight=0.55, frequency=2)
                    
    # Sequential trace connection across shard packages
    if prev_concept and all_concepts:
        connect(prev_concept, all_concepts[0], weight=0.35, frequency=1)
        
    # Re-cluster total concept pool
    total_concepts_list = list(state.semantic_field.concept_embeddings.keys())
    concept_count = len(total_concepts_list)
    if concept_count > 0:
        cluster_count = min(12, max(2, concept_count // 15))
        try:
            new_groups = auto_cluster_shard_groups(n_clusters=cluster_count)
            set_shard_groups(new_groups)
        except:
            set_shard_groups({})
    else:
        set_shard_groups({})
        
    # Tick the engine to trigger sync force
    if all_concepts:
        try: state.process_tick({all_concepts[0]: 0.95})
        except: pass
        
    save_brain_file()
    serialized_state = get_serialized_cognition_state_helper()
    
    last_c = all_concepts[-1] if all_concepts else prev_concept
    return {
        "success": True,
        "added": len(all_concepts),
        "last_concept": last_c,
        "state": serialized_state
    }

def run_cognition_stress_test_algorithm() -> dict:
    state = get_cognition_state()
    shard_groups = get_shard_groups()
    if state is None:
        if not load_brain_file():
            init_cognition_engine()
            state = get_cognition_state()

    if not hasattr(state, '_concept_thoughts') or state._concept_thoughts is None:
        state._concept_thoughts = {}

    test_path = os.path.join(RESULTS_DIR, "test_shards_100.json")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    with open(test_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if raw is None:
        raise ValueError("JSON file is empty")

    all_concepts = []
    seen_concepts = set(state.semantic_field.concept_embeddings.keys())

    def register_concept(concept, activation=0.9, stability=0.9, temporal_depth=0):
        concept = clean_text(concept)
        if not concept: return
        if concept in seen_concepts: return
        seen_concepts.add(concept)
        try: state.semantic_field.register_concept(concept)
        except Exception: pass
        try:
            state.memory.graph.add_node(
                concept, activation=activation, stability=stability, temporal_depth=temporal_depth
            )
        except Exception: pass
        all_concepts.append(concept)

    def connect(a, b, weight=0.7, frequency=3):
        a = clean_text(a); b = clean_text(b)
        if not a or not b or a == b: return
        try:
            state.causal.causal_matrix.add_edge(
                a, b, weight=weight, frequency=frequency, last_seen=time.time()
            )
        except Exception: pass

    def process_cluster(cluster_name, thoughts):
        if not thoughts: return
        cleaned = []
        for t in thoughts:
            t = clean_text(t)
            if not t: continue
            cleaned.append(t)
            register_concept(t)
        if cluster_name:
            state._concept_thoughts[cluster_name] = cleaned
        for i in range(len(cleaned) - 1):
            connect(cleaned[i], cleaned[i + 1], weight=0.85, frequency=5)

    # Decode formats
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                register_concept(item)
            elif isinstance(item, dict):
                if "label" in item and "thoughts" in item:
                    cluster_name = clean_text(item.get("label"))
                    thoughts = extract_strings(item.get("thoughts", []))
                    process_cluster(cluster_name, thoughts)
                elif "text" in item and "thoughts" in item:
                    concept = clean_text(item.get("text"))
                    thoughts = extract_strings(item.get("thoughts", []))
                    if concept:
                        register_concept(concept)
                        state._concept_thoughts[concept] = thoughts
                    for t in thoughts: register_concept(t)
                    for t in thoughts: connect(concept, t, weight=0.9, frequency=5)
                    for i in range(len(thoughts) - 1):
                        connect(thoughts[i], thoughts[i + 1], weight=0.8, frequency=4)
                else:
                    extracted = extract_strings(item)
                    for text in extracted: register_concept(text)
                    for i in range(len(extracted) - 1):
                        connect(extracted[i], extracted[i + 1], weight=0.55, frequency=2)
    elif isinstance(raw, dict):
        extracted = extract_strings(raw)
        for text in extracted: register_concept(text)
        for i in range(len(extracted) - 1):
            connect(extracted[i], extracted[i + 1], weight=0.5, frequency=2)
    else:
        extracted = extract_strings(raw)
        for text in extracted: register_concept(text)

    # Global causal network transition
    for i in range(len(all_concepts) - 1):
        connect(all_concepts[i], all_concepts[i + 1], weight=0.35, frequency=1)

    # Re-clustering
    total_concepts_list = list(state.semantic_field.concept_embeddings.keys())
    concept_count = len(total_concepts_list)
    if concept_count > 0:
        cluster_count = min(12, max(2, concept_count // 15))
        try:
            new_groups = auto_cluster_shard_groups(n_clusters=cluster_count)
            set_shard_groups(new_groups)
        except Exception:
            set_shard_groups({})
    else:
        set_shard_groups({})

    # Inter-cluster links
    try:
        current_groups = get_shard_groups()
        cluster_labels = list(current_groups.keys())
        for i in range(len(cluster_labels) - 1):
            c1 = current_groups[cluster_labels[i]]
            c2 = current_groups[cluster_labels[i + 1]]
            if c1 and c2:
                connect(c1[0], c2[0], weight=0.25, frequency=1)
    except Exception:
        pass

    # Initial activation
    if all_concepts:
        try: state.process_tick({all_concepts[0]: 0.95})
        except Exception: pass

    try: save_brain_file()
    except Exception: pass

    serialized_state = get_serialized_cognition_state_helper()
    current_groups = get_shard_groups()
    return {
        "success": True,
        "loaded_concepts": len(all_concepts),
        "clusters": len(current_groups),
        "sample_concepts": all_concepts[:10],
        "state": serialized_state
    }
