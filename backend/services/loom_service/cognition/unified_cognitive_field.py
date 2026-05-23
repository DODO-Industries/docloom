import numpy as np
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .attractor_dynamics import CognitiveAssembly, SemanticFieldEngine, MetaShards, AssemblyCompilation, safe_normalize
from .cognitive_workspace import AttentionDynamics, WorkingMemory
from .predictive_world_model import PredictiveProcessingEngine, CausalGraphs
from .metacognition import Reflection

@dataclass
class LoomShard:
    """PHASE 2 TARGET: Grounded cognitive nodes replacing symbolic strings."""
    shard_id: str
    embedding: np.ndarray
    text: str
    source: str
    timestamp: float
    semantic_links: List[str] = field(default_factory=list)
    causal_links: List[str] = field(default_factory=list)
    confidence: float = 1.0

@dataclass
class IntentNode:
    """Represents a top-down goal or intentional constraint."""
    intent_id: str
    description: str
    target_field: np.ndarray 
    intensity: float = 1.0     
    active: bool = True

class IntentField:
    """PURPOSE: Top-down goal steering."""
    def __init__(self):
        self.intents: Dict[str, IntentNode] = {}
        self.global_bias: Optional[np.ndarray] = None
        
    def add_intent(self, description: str, target_field: np.ndarray, intensity: float = 1.0) -> str:
        intent_id = f"intent_{uuid.uuid4().hex[:8]}"
        self.intents[intent_id] = IntentNode(intent_id, description, target_field, intensity)
        self._update_global_bias()
        return intent_id

    def _update_global_bias(self):
        active_intents = [i for i in self.intents.values() if i.active]
        if not active_intents:
            self.global_bias = None
            return
        bias = np.zeros_like(active_intents[0].target_field)
        for i in active_intents:
            bias += i.target_field * i.intensity
        norm = np.linalg.norm(bias)
        self.global_bias = bias / norm if norm > 0 else bias

    def apply_pressure(self, field_state: np.ndarray) -> np.ndarray:
        if self.global_bias is None: return field_state
        return field_state + self.global_bias

class CognitiveMetrics:
    """
    Continuous semantic field metrics.
    """
    @staticmethod
    def compute_coherence(active_vectors: List[np.ndarray]) -> float:
        """
        Computes Kuramoto-like phase synchronization (coherence) across active semantic vectors.
        """
        if not active_vectors or len(active_vectors) < 2:
            return 1.0
        
        # Calculate the centroid (mean phase of the semantic cluster)
        centroid = np.mean(active_vectors, axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0: return 0.0
        centroid /= norm
        
        # Kuramoto order parameter equivalent for high-dimensional latent vectors
        # r = 1/N \sum cos(\theta_i - \theta_{centroid})
        alignments = []
        for vec in active_vectors:
            v_norm = np.linalg.norm(vec)
            if v_norm > 0:
                alignments.append(np.dot(vec / v_norm, centroid))
            else:
                alignments.append(0.0)
                
        return float(np.clip(np.mean(alignments), 0.0, 1.0))

    @staticmethod
    def compute_entropy(latent_field: np.ndarray, prev_field: np.ndarray) -> float:
        # Entropy is field turbulence (change in continuous space)
        if prev_field is None or latent_field is None: return 0.0
        dist = np.linalg.norm(latent_field - prev_field)
        return float(np.clip(dist, 0.0, 1.0))

@dataclass
class GlobalCognitiveState:
    """
    THE CENTRAL NERVOUS SYSTEM OF DOCLOOM (CONTINUOUS VECTOR-SPACE)
    """
    semantic_field: SemanticFieldEngine = field(default_factory=SemanticFieldEngine)
    attention: AttentionDynamics = field(default_factory=AttentionDynamics)
    predictor: PredictiveProcessingEngine = field(default_factory=PredictiveProcessingEngine)
    reflection: Reflection = field(default_factory=Reflection)
    intent: IntentField = field(default_factory=IntentField)
    meta_shards: MetaShards = field(default_factory=MetaShards)
    compiler: AssemblyCompilation = field(default_factory=AssemblyCompilation)
    memory: WorkingMemory = field(default_factory=lambda: WorkingMemory(capacity=10))
    causal: CausalGraphs = field(default_factory=CausalGraphs)
    
    tick: int = 0
    active_field: Dict[str, float] = field(default_factory=dict) # Legacy backward compat
    
    # Continuous Multi-Scale Tensors
    latent_field: Optional[np.ndarray] = None
    working_latent: Optional[np.ndarray] = None
    episodic_tensors: List[np.ndarray] = field(default_factory=list)
    
    # Evolving Coordinate & Physics Tensor Systems
    concept_coords: Dict[str, np.ndarray] = field(default_factory=dict)
    concept_velocities: Dict[str, np.ndarray] = field(default_factory=dict)
    physics_tensors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Explicit Temporal Memory Layer Separation
    sensory_memory: List[str] = field(default_factory=list)
    
    temporal_history: List[Dict[str, float]] = field(default_factory=list)
    active_assemblies: List[CognitiveAssembly] = field(default_factory=list)
    attention_focus: List[str] = field(default_factory=list)
    
    entropy: float = 0.5
    coherence: float = 0.5
    pressure: float = 0.0
    surprise: float = 0.0
    energy_budget: float = 1.0
    
    def process_tick(self, resonance_input: Dict[str, float]) -> Optional[CognitiveAssembly]:
        # Sensory Memory Layer: Buffer incoming inputs
        self.sensory_memory = list(resonance_input.keys())
        
        # Process memory decay in Working Memory Layer
        self.memory.tick()
        
        # 1. Convert sparse input into Continuous Vector Field
        # Ensure semantic field is updated
        active_vecs = []
        for text in resonance_input:
            self.semantic_field.register_concept(text)
            
        # Synchronize dynamic concept physics states on concept registration
        for concept in self.semantic_field.concept_embeddings:
            if concept not in self.concept_coords:
                emb = self.semantic_field.concept_embeddings[concept]
                self.concept_coords[concept] = emb.copy()
                self.concept_velocities[concept] = np.zeros_like(emb)
                self.physics_tensors[concept] = {
                    "energy": 1.0,
                    "entropy": 0.0,
                    "stability": 1.0,
                    "activation": 0.9,
                    "resonance": 0.0,
                    "momentum": 0.0,
                    "decay": 0.05,
                    "attention": 0.0
                }
            
        field_dim = len(next(iter(self.semantic_field.concept_embeddings.values()))) if self.semantic_field.concept_embeddings else 128
        input_tensor = np.zeros(field_dim)
        
        for text, intensity in resonance_input.items():
            # Use dynamic evolving coordinates instead of static embeddings for field driving
            vec = self.concept_coords[text]
            active_vecs.append(vec)
            input_tensor += vec * intensity
            
        norm = np.linalg.norm(input_tensor)
        if norm > 0: input_tensor /= norm
        
        # Apply Intent Pressure (Vector Field Bias)
        if self.intent.global_bias is not None:
            input_tensor = input_tensor + (self.intent.global_bias * 0.5)
            inorm = np.linalg.norm(input_tensor)
            if inorm > 0: input_tensor /= inorm
            
        prev_latent = self.latent_field if self.latent_field is not None else np.zeros_like(input_tensor)
        
        # 2. Competitive Resonance Dynamics (Second-Order Oscillatory Field)
        if self.working_latent is None: self.working_latent = np.zeros_like(input_tensor)
        if not hasattr(self, 'latent_velocity') or self.latent_velocity is None: self.latent_velocity = np.zeros_like(input_tensor)
        
        # Kuramoto-like attractive force towards working memory (synchronization)
        synchronization_force = self.working_latent - prev_latent
        
        # Hopfield-like competitive inhibition (destructively interferes with non-aligned states)
        inhibition_tensor = self.working_latent * 0.15
        
        # Second-order physics: Mass-Spring-Damper system with semantic driving force
        self.latent_velocity = (0.8 * self.latent_velocity) + (input_tensor * 0.4) + (synchronization_force * 0.2) - inhibition_tensor
        
        # Field Position Update
        new_latent = prev_latent + self.latent_velocity
        lnorm = np.linalg.norm(new_latent)
        if lnorm > 0: self.latent_field = new_latent / lnorm
        else: self.latent_field = new_latent
        
        # Stable Attractor Manifold (Working Memory) follows via exponential moving average
        self.working_latent = 0.9 * self.working_latent + 0.1 * self.latent_field
        wnorm = np.linalg.norm(self.working_latent)
        if wnorm > 0: self.working_latent /= wnorm
        
        # 3. Continuous Field Physics Step: drift, attract, and evolve individual concept coordinates
        dt = 0.1
        if self.latent_field is not None and self.concept_coords:
            for c, pos in self.concept_coords.items():
                vel = self.concept_velocities.get(c, np.zeros_like(pos))
                tensors = self.physics_tensors.setdefault(c, {
                    "energy": 1.0, "entropy": 0.0, "stability": 1.0, "activation": 0.9,
                    "resonance": 0.0, "momentum": 0.0, "decay": 0.05, "attention": 0.0
                })
                
                # Attraction to consciousness focus (latent field)
                attention_boost = 1.8 if c in self.attention_focus else 1.0
                activation = tensors.get("activation", 0.5)
                f_latent = (self.latent_field - pos) * (0.25 * activation * attention_boost)
                
                # Causal spring attraction to connected neighbors
                f_causal = np.zeros_like(pos)
                if self.causal.causal_matrix.has_node(c):
                    edges = self.causal.causal_matrix.edges(c, data=True)
                    for _, target, data in edges:
                        weight = data.get("weight", 0.5)
                        if target in self.concept_coords:
                            f_causal += (self.concept_coords[target] - pos) * (weight * 0.15)
                
                # Semantic repulsion to prevent clumping
                f_repulsion = np.zeros_like(pos)
                for other, other_pos in self.concept_coords.items():
                    if other == c: continue
                    diff = pos - other_pos
                    dist = np.linalg.norm(diff)
                    if dist < 0.2:
                        f_repulsion += (diff / (dist + 1e-5)) * 0.02
                        
                f_total = f_latent + f_causal + f_repulsion
                
                # Momentum & Velocity Update
                vel = vel * 0.7 + f_total * dt
                self.concept_velocities[c] = vel
                
                # Update position & project on unit hypersphere
                self.concept_coords[c] = safe_normalize(pos + vel * dt)
                
                # Update physics state tensors for the node
                res = float(np.dot(self.concept_coords[c], self.latent_field))
                tensors["resonance"] = float(np.clip(res, 0.0, 1.0))
                
                decay_rate = tensors.get("decay", 0.05)
                input_intensity = resonance_input.get(c, 0.0)
                if input_intensity > 0:
                    tensors["activation"] = min(1.0, tensors["activation"] + input_intensity * 0.3)
                    tensors["energy"] = min(1.0, tensors["energy"] + 0.2)
                else:
                    tensors["activation"] = max(0.0, tensors["activation"] - decay_rate)
                    tensors["energy"] = max(0.0, tensors["energy"] - 0.02)
                    
                tensors["entropy"] = float(np.clip(np.linalg.norm(vel), 0.0, 1.0))
                if tensors["activation"] > 0.4:
                    tensors["stability"] = min(1.0, tensors["stability"] + 0.02)
                else:
                    tensors["stability"] = max(0.01, tensors["stability"] - 0.005)
                tensors["attention"] = 1.0 if c in self.attention_focus else 0.0
        
        # 4. Continuous Field Metrics
        self.coherence = CognitiveMetrics.compute_coherence(active_vecs)
        self.entropy = CognitiveMetrics.compute_entropy(self.latent_field, prev_latent)
        
        # 5. Thermodynamic Energy & Latent Prediction
        coherence_gain = max(0.0, self.coherence - getattr(self, '_last_coherence', self.coherence))
        self._last_coherence = self.coherence
        
        predicted_latent = self.predictor.generate_continuous_prediction(prev_latent)
        prediction_error = np.linalg.norm(self.latent_field - predicted_latent) if predicted_latent is not None else 0.5
        self.surprise = float(np.clip(prediction_error, 0.0, 1.0))
        
        self.predictor.update_model_continuous(prev_latent, self.latent_field)
        
        previous_nodes = list(self.active_field.keys())
        predicted_map = self.predictor.generate_prediction(previous_nodes)
        self.predictor.update_model(previous_nodes, list(resonance_input.keys()))
        
        cognitive_load = self.entropy * 0.02
        novelty_cost = self.surprise * 0.015
        recovery = coherence_gain * 0.15 + (self.coherence * 0.01)
        drain = cognitive_load + novelty_cost + (self.pressure * 0.005)
        
        self.energy_budget += (recovery - drain)
        self.energy_budget = float(np.clip(self.energy_budget, 0.0, 1.0))
        
        # 6. Extract Symbolic Projection for Legacy Components
        projected_activations = {}
        for text, vec in self.concept_coords.items():
            sim = np.dot(self.latent_field, vec)
            if sim > 0.4:
                projected_activations[text] = float(sim)
                
        self.active_field = projected_activations
        self.temporal_history.append(dict(self.active_field))
        if len(self.temporal_history) > 5:
            self.temporal_history.pop(0)
            
        self.attention_focus = self.attention.compute_attention(self.active_field, list(self.intent.intents.keys()), self.surprise)
        
        stats = {"entropy": self.entropy, "coherence": self.coherence, "surprise": self.surprise, "energy": self.energy_budget}
        refl_report = self.reflection.analyze_self(stats, self.latent_field)
        self.pressure = refl_report["reflection_pressure"]
        self.attention.adjust_parameters(self.pressure)
        
        # 7. Attractor Crystallization from Field
        ca = self.compiler.compile_assembly(
            latent_field=self.latent_field,
            projected_activations=projected_activations,
            entropy=self.entropy,
            coherence=self.coherence,
            pressure=self.pressure,
            semantic_field=self.semantic_field
        )
        
        if ca:
            self.active_assemblies.append(ca)
            self.memory.insert_assembly(ca)
            self.episodic_tensors.append(self.latent_field.copy()) # Episodic Memory Layer
            if len(self.episodic_tensors) > 1000:
                self.episodic_tensors.pop(0)
            
            if len(self.active_assemblies) > 1:
                prev = self.active_assemblies[-2].dominant_concept
                if prev != ca.dominant_concept:
                    self.causal.record_transition(prev, ca.dominant_concept)
                    
            if len(self.active_assemblies) >= 3:
                recent_ids = [a.dominant_concept for a in self.active_assemblies[-3:]]
                recent_vecs = [self.concept_coords[nid] for nid in recent_ids]
                activations = [a.confidence for a in self.active_assemblies[-3:]]
                self.meta_shards.create_meta_shard(recent_ids, recent_vecs, activations, self.semantic_field)
                self.active_assemblies = self.active_assemblies[-1:]
                
        # Trigger dynamic memory consolidation during recovery mode or low energy budget
        if self.energy_budget < 0.35 or self.reflection.cognitive_mode == 'recovery':
            self.consolidate_memories()
            
        self.tick += 1
        return ca

    def consolidate_memories(self):
        """
        Sleep consolidation phase: decay weak concepts, run episodic replay, compress abstractions.
        """
        # 1. Sensory Forgetting (decay weak thoughts)
        to_remove = []
        for c, tensors in list(self.physics_tensors.items()):
            if tensors["stability"] < 0.05 and tensors["activation"] < 0.1:
                to_remove.append(c)
                
        for c in to_remove:
            if c in self.concept_coords: del self.concept_coords[c]
            if c in self.concept_velocities: del self.concept_velocities[c]
            if c in self.physics_tensors: del self.physics_tensors[c]
            if c in self.semantic_field.concept_embeddings:
                del self.semantic_field.concept_embeddings[c]
            if self.memory.graph.has_node(c):
                self.memory.graph.remove_node(c)
            if self.causal.causal_matrix.has_node(c):
                self.causal.causal_matrix.remove_node(c)
                
        # 2. Episodic Replay (reactivate past memory traces)
        if self.episodic_tensors and self.concept_coords:
            import random
            past_trace = random.choice(self.episodic_tensors)
            for c, pos in self.concept_coords.items():
                alignment = float(np.dot(pos, past_trace))
                if alignment > 0.6:
                    self.physics_tensors[c]["activation"] = min(1.0, self.physics_tensors[c]["activation"] + 0.15)
                    self.physics_tensors[c]["stability"] = min(1.0, self.physics_tensors[c]["stability"] + 0.05)
                    
        # 3. Abstraction Compression (crystallize and compress working memory)
        self.memory.memory_compression(threshold=0.75, semantic_engine=self.semantic_field)
        
        # Recharge energy budget after a successful consolidation (sleep) cycle
        self.energy_budget = min(1.0, self.energy_budget + 0.35)
        
    def get_summary(self) -> Dict[str, Any]:
        return {
            "tick": self.tick, "entropy": self.entropy, "coherence": self.coherence,
            "pressure": self.pressure, "surprise": self.surprise, "energy": self.energy_budget
        }

    def save_state(self) -> Dict[str, Any]:
        """ISSUE 16: Basic serialization layer."""
        coords_serial = {k: v.tolist() for k, v in self.concept_coords.items()}
        vel_serial = {k: v.tolist() for k, v in self.concept_velocities.items()}
        return {
            "tick": self.tick,
            "entropy": self.entropy,
            "coherence": self.coherence,
            "pressure": self.pressure,
            "surprise": self.surprise,
            "energy_budget": self.energy_budget,
            "latent_field": self.latent_field.tolist() if self.latent_field is not None else None,
            "working_latent": self.working_latent.tolist() if self.working_latent is not None else None,
            "concept_coords": coords_serial,
            "concept_velocities": vel_serial,
            "physics_tensors": self.physics_tensors
        }
        
    def load_state(self, state_dict: Dict[str, Any]):
        self.tick = state_dict.get("tick", 0)
        self.entropy = state_dict.get("entropy", 0.5)
        self.coherence = state_dict.get("coherence", 0.5)
        self.pressure = state_dict.get("pressure", 0.0)
        self.surprise = state_dict.get("surprise", 0.0)
        self.energy_budget = state_dict.get("energy_budget", 1.0)
        
        l_field = state_dict.get("latent_field")
        if l_field is not None: self.latent_field = np.array(l_field)
        
        w_field = state_dict.get("working_latent")
        if w_field is not None: self.working_latent = np.array(w_field)
        
        coords_dict = state_dict.get("concept_coords", {})
        self.concept_coords = {k: np.array(v) for k, v in coords_dict.items()}
        
        vel_dict = state_dict.get("concept_velocities", {})
        self.concept_velocities = {k: np.array(v) for k, v in vel_dict.items()}
        
        self.physics_tensors = state_dict.get("physics_tensors", {})
