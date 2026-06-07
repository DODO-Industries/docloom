import numpy as np
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

def cosine_similarity_vec(v1, v2):
    """Native numpy cosine similarity."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return dot / (norm1 * norm2)

from backend.services.loom_service.cortex.embedding.embedding_manager import get_embedding_model

def safe_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-9 else vec

@dataclass
class CognitiveAssembly:
    """The resulting semantic thought unit."""
    assembly_id: str
    dominant_concept: str
    node_activations: Dict[str, float]
    confidence: float
    stability: float
    phase_coherence: float
    meta_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetaShard:
    """Higher-level conceptual node."""
    meta_id: str
    children: List[str]
    field_vec: np.ndarray
    abstraction_level: int
    stability: float
    emergent_label: str

class SemanticFieldEngine:
    """Transforms symbolic nodes into meaning geometry."""
    def __init__(self):
        self.concept_embeddings: Dict[str, np.ndarray] = {}
        model = get_embedding_model()
        self.embedding_dim = len(model.encode("dimension_probe"))
        
    def register_concept(self, nid: str, vector: Optional[np.ndarray] = None):
        if vector is not None:
            self.concept_embeddings[nid] = safe_normalize(vector)
        elif nid not in self.concept_embeddings:
            # Uses caching implicitly via `not in`
            model = get_embedding_model()
            vec = model.encode(nid)
            self.concept_embeddings[nid] = safe_normalize(vec)

    def get_semantic_distance(self, nid1: str, nid2: str) -> float:
        if nid1 not in self.concept_embeddings or nid2 not in self.concept_embeddings: return 1.0
        v1 = self.concept_embeddings[nid1].reshape(1, -1)
        v2 = self.concept_embeddings[nid2].reshape(1, -1)
        return float(1.0 - cosine_similarity_vec(v1.flatten(), v2.flatten()))

class AssemblyCompilation:
    """Converts resonance field into crystallized assemblies."""
    def __init__(self):
        self.attractor_memory = {}

    def compile_assembly(self, 
                         latent_field: np.ndarray,
                         projected_activations: Dict[str, float], 
                         entropy: float, 
                         coherence: float,
                         pressure: float,
                         semantic_field: SemanticFieldEngine) -> Optional[CognitiveAssembly]:
        """
        CRYSTALLIZATION: Transforms resonance into discrete emergent attractors.
        Uses Multi-Factor Attractor Physics.
        """
        if not projected_activations: return None
        
        # 1. Base Semantic Energy
        energy = coherence * (1.0 - entropy)
        adaptive_threshold = 0.2 + (entropy * 0.3) + (pressure * 0.1)
        if energy < adaptive_threshold: return None # Dynamic threshold
        
        # Clustering: Emergent assembly is the set of active nodes resonating together
        sorted_nodes = sorted(projected_activations.items(), key=lambda x: x[1], reverse=True)
        if not sorted_nodes: return None
        
        core_nodes = {k: v for k, v in sorted_nodes if v > sorted_nodes[0][1] * 0.8} # Competitive suppression
        
        dominant = sorted_nodes[0][0]
        confidence = float(np.mean(list(core_nodes.values())))
        
        # 4. Attractor Basin Updates
        if dominant not in self.attractor_memory:
            self.attractor_memory[dominant] = {"frequency": 0, "avg_confidence": 0.0, "basin_strength": 0.0}
        
        mem = self.attractor_memory[dominant]
        mem["frequency"] += 1
        mem["avg_confidence"] = (mem["avg_confidence"] * (mem["frequency"] - 1) + confidence) / mem["frequency"]
        mem["basin_strength"] = min(1.0, mem["basin_strength"] + 0.15)
        
        # Strengthen repeated patterns
        confidence += (mem["basin_strength"] * 0.1)
        
        return CognitiveAssembly(
            assembly_id=f"assy_{uuid.uuid4().hex[:6]}",
            dominant_concept=dominant,
            node_activations=core_nodes,
            confidence=float(confidence),
            stability=1.0 - entropy,
            phase_coherence=coherence,
            meta_data={"basin_strength": mem["basin_strength"]}
        )

    def compute_temporal_persistence(self, active_node_ids: List[str], temporal_history: List[Dict[str, float]]) -> float:
        if not temporal_history: return 0.0
        persistences = []
        for nid in active_node_ids:
            active_frames = sum(1 for frame in temporal_history if nid in frame and frame[nid] > 0.1)
            persistences.append(active_frames / len(temporal_history))
        return float(np.mean(persistences))

    def compute_phase_alignment(self, phases: Dict[str, float]) -> float:
        if not phases: return 0.0
        phi = np.array(list(phases.values()))
        return float(np.abs(np.mean(np.exp(1j * phi))))

    def compute_stability(self, history: List[Dict[str, float]]) -> float:
        if len(history) < 2: return 0.0
        h1, h2 = history[-2], history[-1]
        keys = set(h1.keys()).union(set(h2.keys()))
        v1 = np.array([h1.get(k, 0.0) for k in keys])
        v2 = np.array([h2.get(k, 0.0) for k in keys])
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0: return 0.0
        return float(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 0, 1))

class MetaShards:
    """Recursive concept abstraction engine."""
    def __init__(self, max_shards=1000):
        self.meta_store: Dict[str, MetaShard] = {}
        self.max_shards = max_shards

    def create_meta_shard(self, child_ids: List[str], child_vecs: List[np.ndarray], activations: List[float], semantic_field=None) -> MetaShard:
        weights = np.array(activations)
        weights /= (np.sum(weights) + 1e-9)
        meta_vec = np.sum([w * v for w, v in zip(weights, child_vecs)], axis=0)
        mnorm = np.linalg.norm(meta_vec)
        if mnorm > 0: meta_vec /= mnorm
        stability = float(1.0 / (1.0 + np.mean(np.var(child_vecs, axis=0))))
        
        # Real Abstraction: Emergent concept bounding based on nearest intersections
        emergent_label = "Uncharted_Semantic_Region"
        if semantic_field:
            sims = []
            for concept, vec in semantic_field.concept_embeddings.items():
                if concept not in child_ids:
                    sims.append((concept, np.dot(meta_vec, vec)))
            sims.sort(key=lambda x: x[1], reverse=True)
            if len(sims) >= 2:
                # Truncate strings to prevent huge labels
                c1 = sims[0][0][:20].strip()
                c2 = sims[1][0][:20].strip()
                emergent_label = f"Intersection({c1} AND {c2})"
            elif len(sims) == 1:
                emergent_label = f"Variant({sims[0][0][:20]})"
                    
        shard = MetaShard(f"meta_{uuid.uuid4().hex[:8]}", child_ids, meta_vec, 1, stability, f"Abstraction[{emergent_label}]")
        self.meta_store[shard.meta_id] = shard
        
        # MetaShard explosion protection
        if len(self.meta_store) > self.max_shards:
            oldest = next(iter(self.meta_store))
            del self.meta_store[oldest]
            
        return shard
