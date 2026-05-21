import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional
import time

class PredictiveProcessingEngine:
    """ACTIVE INFERENCE: Predicts and learns future states."""
    def __init__(self, learning_rate=0.01, max_context=2):
        self.internal_model: Dict[tuple, Dict[str, float]] = {} 
        self.prediction_history: List[float] = []
        self.learning_rate = learning_rate
        self.max_context = max_context
        self.context_window = []
        # ISSUE 7: Prediction operates in embedding space
        self.transition_matrix: Optional[np.ndarray] = None
        self.last_latent: Optional[np.ndarray] = None

    def update_model_continuous(self, prev_latent: np.ndarray, curr_latent: np.ndarray):
        if prev_latent is None or curr_latent is None: return
        dim = len(prev_latent)
        if self.transition_matrix is None:
            self.transition_matrix = np.eye(dim)
        
        # Simple Hebbian update for continuous trajectory prediction
        pred = self.transition_matrix @ prev_latent
        error = curr_latent - pred
        self.transition_matrix += self.learning_rate * np.outer(error, prev_latent)
        
        # Decay to prevent destabilization (Issue 8)
        self.transition_matrix *= 0.999
        
        # Normalize to prevent explosion
        norm = np.linalg.norm(self.transition_matrix)
        if norm > 10.0: self.transition_matrix *= (10.0 / norm)

    def generate_continuous_prediction(self, current_latent: np.ndarray) -> np.ndarray:
        if self.transition_matrix is None or current_latent is None: return current_latent
        pred = self.transition_matrix @ current_latent
        norm = np.linalg.norm(pred)
        return pred / norm if norm > 0 else pred
        
    def generate_prediction(self, current_active_nodes: List[str]) -> Dict[str, float]:
        predictions = {}
        context = tuple(self.context_window) if self.context_window else tuple(current_active_nodes[:self.max_context])
        
        if context in self.internal_model:
            for target, prob in self.internal_model[context].items():
                predictions[target] = predictions.get(target, 0.0) + prob
                
        # Fallback to markovian
        if not predictions:
            for nid in current_active_nodes:
                ctx = (nid,)
                if ctx in self.internal_model:
                    for target, prob in self.internal_model[ctx].items():
                        predictions[target] = predictions.get(target, 0.0) + prob
                        
        total = sum(predictions.values())
        return {k: v/total for k, v in predictions.items()} if total > 0 else {}

    def update_model(self, previous_nodes: List[str], current_nodes: List[str]):
        if previous_nodes:
            self.context_window.append(previous_nodes[0])
            if len(self.context_window) > self.max_context:
                self.context_window.pop(0)
                
        if len(self.context_window) == self.max_context:
            ctx = tuple(self.context_window)
            if ctx not in self.internal_model: self.internal_model[ctx] = {}
            for c in current_nodes:
                old = self.internal_model[ctx].get(c, 0.0)
                self.internal_model[ctx][c] = (1 - self.learning_rate) * old + self.learning_rate
                
        for p in previous_nodes:
            ctx = (p,)
            if ctx not in self.internal_model: self.internal_model[ctx] = {}
            for c in current_nodes:
                old = self.internal_model[ctx].get(c, 0.0)
                # FIX 1: Exponential moving probabilities to prevent catastrophic forgetting
                self.internal_model[ctx][c] = (1 - self.learning_rate) * old + self.learning_rate

    def compute_surprise(self, predicted: Dict[str, float], actual: List[str]) -> float:
        if not actual: return 0.0
        # FIX: Uncertainty-aware surprise (0.25 instead of 1.0) for new environments
        if not predicted: return 0.25
        surprise = float(1.0 - sum(predicted.get(nid, 0.0) for nid in actual))
        self.prediction_history.append(surprise)
        if len(self.prediction_history) > 100: self.prediction_history.pop(0)
        return surprise

class CausalGraphs:
    """CAUSAL ENGINE: Tracks A causes B transitions."""
    def __init__(self):
        self.causal_matrix = nx.DiGraph()
        self.transition_history = []

    def record_transition(self, node_a: str, node_b: str, weight: float = 1.0, prediction_error: float = 0.5):
        if node_a == node_b: return # Anti-hallucination: no self-loops
        
        # ISSUE 12: True Causality
        # Temporal adjacency ≠ causation. A causes B only if B is a predictable consequence of A.
        # We penalize the causal weight by the prediction error. If it's pure surprise, it's temporal noise.
        causal_strength = weight * (1.0 - prediction_error)
        if causal_strength < 0.1: return # Reject pure temporal noise
        
        now = time.time()
        if self.causal_matrix.has_edge(node_a, node_b):
            self.causal_matrix[node_a][node_b]['weight'] = (self.causal_matrix[node_a][node_b]['weight'] + causal_strength) / 2
            self.causal_matrix[node_a][node_b]['last_seen'] = now
            self.causal_matrix[node_a][node_b]['frequency'] += 1
        else:
            self.causal_matrix.add_edge(node_a, node_b, weight=causal_strength, last_seen=now, frequency=1)
        self.transition_history.append((node_a, node_b))
        self.decay_stale_edges()

    def decay_stale_edges(self, decay_rate=0.1, threshold=0.05):
        now = time.time()
        to_remove = []
        for u, v, data in self.causal_matrix.edges(data=True):
            age = max(0, now - data.get('last_seen', now))
            if age > 60: # Older than 60 seconds
                data['weight'] *= (1.0 - decay_rate)
            if data['weight'] < threshold:
                to_remove.append((u, v))
        self.causal_matrix.remove_edges_from(to_remove)
        self.prune_orphan_nodes()

    def prune_orphan_nodes(self):
        """ISSUE 7: Clean up disconnected nodes to prevent graph bloat."""
        orphans = [n for n, d in self.causal_matrix.degree() if d == 0]
        self.causal_matrix.remove_nodes_from(orphans)

    def detect_feedback_loops(self) -> List[List[str]]:
        try: return list(nx.simple_cycles(self.causal_matrix))
        except: return []

    def extract_causal_chain(self, start_node: str, max_length: int = 5) -> List[str]:
        if start_node not in self.causal_matrix: return []
        chain = []
        current = start_node
        for _ in range(max_length):
            neighbors = list(self.causal_matrix.neighbors(current))
            if not neighbors: break
            current = max(neighbors, key=lambda n: self.causal_matrix[current][n]['weight'])
            if current in chain or current == start_node: break # Stop on cycle
            chain.append(current)
        return chain

    def infer_hidden_causes(self, common_effect: str) -> List[str]:
        if common_effect not in self.causal_matrix: return []
        return [n for n in self.causal_matrix.predecessors(common_effect)]

    def simulate_future_path(self, start_node: str, depth: int = 3) -> List[List[str]]:
        """Counterfactual reasoning engine. Generates multiple future paths."""
        if start_node not in self.causal_matrix: return []
        paths = []
        def dfs(current_node, current_path, d):
            if d == 0:
                paths.append(current_path)
                return
            neighbors = list(self.causal_matrix.neighbors(current_node))
            if not neighbors:
                paths.append(current_path)
                return
            
            # Follow top branches
            neighbors.sort(key=lambda n: self.causal_matrix[current_node][n].get('weight', 0), reverse=True)
            for n in neighbors[:2]:
                if n not in current_path: # avoid cycles in simulation
                    dfs(n, current_path + [n], d - 1)
                    
        dfs(start_node, [start_node], depth)
        return paths
