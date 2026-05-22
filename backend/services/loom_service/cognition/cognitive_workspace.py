import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional
from .attractor_dynamics import CognitiveAssembly

class AttentionDynamics:
    """COGNITIVE SPOTLIGHT: Prioritizes concepts."""
    def __init__(self, decay_rate=0.8, gain_rate=0.2, inertia=0.3):
        self.salience_map: Dict[str, float] = {}
        self.decay_rate = decay_rate
        self.gain_rate = gain_rate
        self.inertia = inertia
        self.inhibition_level = 0.5
        
    def compute_attention(self, activations: Dict[str, float], goals: List[str], prediction_error: float) -> List[str]:
        new_salience = {}
        for nid, val in activations.items():
            prev_s = self.salience_map.get(nid, 0.0) * self.decay_rate
            score = (val * self.gain_rate + prev_s * self.inertia) * (1.0 + prediction_error) * (1.5 if nid in goals else 1.0)
            new_salience[nid] = score
        
        if new_salience:
            max_s = max(new_salience.values())
            for nid in new_salience:
                if new_salience[nid] < max_s * self.inhibition_level: new_salience[nid] *= 0.5
        
        self.salience_map = new_salience
        sorted_nodes = sorted(self.salience_map.items(), key=lambda x: x[1], reverse=True)
        return [n for n, s in sorted_nodes[:(3 if prediction_error < 0.3 else 6)]]

    def adjust_parameters(self, pressure: float):
        self.inhibition_level = np.clip(pressure, 0.2, 0.9)
        self.decay_rate = np.clip(1.0 - pressure, 0.5, 0.9)

class WorkingMemory:
    """ACTIVE WORKSPACE: Dynamic graph-based memory."""
    def __init__(self, capacity=7, decay_rate=0.1):
        self.graph = nx.DiGraph()
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.last_inserted_id: Optional[str] = None

    def insert_assembly(self, assembly: CognitiveAssembly):
        nid = assembly.dominant_concept
        if self.graph.has_node(nid):
            self.graph.nodes[nid]['activation'] = (self.graph.nodes[nid]['activation'] + assembly.confidence) / 2
            self.graph.nodes[nid]['temporal_depth'] = 0
        else:
            self.graph.add_node(nid, activation=assembly.confidence, stability=assembly.stability, temporal_depth=0)
            
        if self.last_inserted_id and self.last_inserted_id != nid:
            weight = (assembly.confidence + self.graph.nodes[self.last_inserted_id]['activation']) / 2
            self.graph.add_edge(self.last_inserted_id, nid, weight=weight)
        
        self.last_inserted_id = nid
        self._enforce_capacity()

    def tick(self):
        to_remove = []
        for nid, data in self.graph.nodes(data=True):
            importance = data.get("stability", 0.5)
            adaptive_decay = self.decay_rate * (1.0 - importance)
            data['activation'] *= (1.0 - adaptive_decay)
            data['temporal_depth'] += 1
            if data['activation'] < 0.1: to_remove.append(nid)
        for nid in to_remove: self.graph.remove_node(nid)

    def _enforce_capacity(self):
        if len(self.graph.nodes) > self.capacity:
            sorted_nodes = sorted(self.graph.nodes(data=True), key=lambda x: x[1]['activation'])
            self.graph.remove_node(sorted_nodes[0][0])

    def get_active_context(self) -> List[str]:
        return sorted(self.graph.nodes(), key=lambda n: self.graph.nodes[n]['activation'], reverse=True)

    def replay_active_memories(self, global_state=None):
        """Reactivates strongest nodes, boosts associated edges, feeds predictor again (internal thinking)."""
        if not self.graph.nodes: return
        active = self.get_active_context()
        if not active: return
        strongest = active[0]
        self.graph.nodes[strongest]['activation'] = min(1.0, self.graph.nodes[strongest]['activation'] * 1.2)
        
        # ISSUE 15: True Rehearsal (Re-run prediction / compression)
        if global_state and hasattr(global_state, 'causal'):
            for node in active[:2]:
                global_state.causal.simulate_future_path(node, depth=1)
                
        for neighbor in self.graph.neighbors(strongest):
            self.graph.nodes[neighbor]['activation'] = min(1.0, self.graph.nodes[neighbor]['activation'] * 1.1)
            self.graph[strongest][neighbor]['weight'] = min(1.0, self.graph[strongest][neighbor]['weight'] * 1.1)

    def memory_compression(self, threshold=0.8, semantic_engine=None):
        if len(self.graph.nodes) < 2: return
        to_merge = []
        nodes = list(self.graph.nodes(data=True))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n1, n2 = nodes[i][0], nodes[j][0]
                
                # 1. Structural Overlap
                structural = 0.0
                if self.graph.has_edge(n1, n2) and self.graph.has_edge(n2, n1):
                    structural = (self.graph[n1][n2]['weight'] + self.graph[n2][n1]['weight'])/2
                    
                # 2. Semantic Similarity
                semantic = 0.0
                if semantic_engine:
                    semantic = 1.0 - semantic_engine.get_semantic_distance(n1, n2)
                    
                # 3. Temporal Overlap
                temporal = 1.0 / (1.0 + abs(self.graph.nodes[n1].get('temporal_depth', 0) - self.graph.nodes[n2].get('temporal_depth', 0)))
                
                merge_score = (0.4 * structural) + (0.4 * semantic) + (0.2 * temporal)
                
                if merge_score > threshold: to_merge.append((n1, n2))
        for n1, n2 in to_merge:
            if self.graph.has_node(n1) and self.graph.has_node(n2):
                self.graph.nodes[n1]['activation'] = (self.graph.nodes[n1]['activation'] + self.graph.nodes[n2]['activation']) / 2
                self.graph.remove_node(n2)
                if self.last_inserted_id == n2: self.last_inserted_id = n1
