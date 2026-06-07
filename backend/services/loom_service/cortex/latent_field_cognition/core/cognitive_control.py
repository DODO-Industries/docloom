import numpy as np
from typing import Dict, Any, List, Optional
import time

from .state_machine import CognitiveStateMachine
from .consciousness_streaming import ConsciousnessStream
from .global_workspace import GlobalWorkspace
from .cognitive_scheduling import CognitiveScheduler
from .energy_regulation import EnergyRegulation
from .temporal_awareness import TemporalAwareness
from .cognitive_routing import CognitiveRouter

class CognitiveController:
    """
    The Brain Stem + Cognitive Cortex of DocLoom.
    Orchestrates tick cycles, reads perception inputs, directs planning/reasoning,
    evaluates safety constraints, dispatches execution tasks, and registers memory consolidations.
    """
    def __init__(self, global_state=None):
        self.state_machine = CognitiveStateMachine()
        self.stream = ConsciousnessStream()
        self.workspace = GlobalWorkspace()
        self.scheduler = CognitiveScheduler()
        self.energy_regulator = EnergyRegulation()
        self.temporal_awareness = TemporalAwareness()
        self.router = CognitiveRouter()
        
        self.global_state = global_state  # Reference to GlobalCognitiveState
        self.active_tick = 0
        
    def execute_cognitive_tick(self, perception_input: Dict[str, Any]) -> Dict[str, Any]:
        """Runs a complete cognitive cycle: Perceive -> Understand -> Simulate -> Plan -> Act -> Reflect -> Improve."""
        self.active_tick += 1
        
        # 1. Temporal & Ingestion awareness
        text_content = perception_input.get("text", "")
        self.temporal_awareness.record_tick(input_size=len(text_content.split()))
        
        # 2. Workspace ingestion (Publish input)
        self.workspace.publish(sender="perception", content=perception_input, salience=1.0)
        
        # 3. Routing Analysis
        route_decision = self.router.route(perception_input)
        self.workspace.publish(sender="router", content={"route": route_decision}, salience=0.8)
        
        # 4. State Machine & Energy updates
        metrics = {
            "entropy": getattr(self.global_state, "entropy", 0.5) if self.global_state else 0.5,
            "coherence": getattr(self.global_state, "coherence", 0.5) if self.global_state else 0.5,
            "surprise": getattr(self.global_state, "surprise", 0.0) if self.global_state else 0.0,
            "has_goals": 1.0 if "goal" in perception_input else 0.0,
            "plan_ready": 0.0,
            "execution_done": 0.0
        }
        
        reg_status = self.energy_regulator.regulate(metrics)
        metrics["energy"] = reg_status["energy"]
        metrics["pressure"] = reg_status["pressure"]
        
        cognitive_mode = self.state_machine.transition(metrics)
        
        # 5. Continuous Stream of Consciousness update
        dummy_vec = np.ones(384) / np.linalg.norm(np.ones(384))
        if self.global_state and getattr(self.global_state, "latent_field", None) is not None:
            latent_vec = self.global_state.latent_field
        else:
            latent_vec = dummy_vec
            
        dominant_concept = perception_input.get("dominant_concept", "ambient_observation")
        self.stream.add_thought(dominant_concept, latent_vec, metadata={"mode": cognitive_mode})
        
        # 6. Global Workspace Broadcast
        broadcast_packet = self.workspace.broadcast()
        
        # 7. Subsystem Coordination based on mode
        execution_results = {}
        planning_status = {}
        reasoning_insights = []
        
        if cognitive_mode == "planning" and self.global_state:
            # Trigger goal decomposer & planning
            goal = perception_input.get("goal")
            if goal:
                # We can communicate via global state modules
                planner = getattr(self.global_state, "planner", None)
                if planner:
                    planning_status = planner.create_and_solve_plan(goal)
                    metrics["plan_ready"] = 1.0
                    
        elif cognitive_mode == "execution" and self.global_state:
            # Trigger agentic execution
            executor = getattr(self.global_state, "executor", None)
            if executor:
                execution_results = executor.execute_next_plan_action()
                metrics["execution_done"] = 1.0
                
        elif cognitive_mode == "focused_reasoning" and self.global_state:
            # Trigger neuro-symbolic reasoning
            reasoner = getattr(self.global_state, "reasoner", None)
            if reasoner:
                reasoning_insights = reasoner.verify_workspace_consistency(self.workspace)
                
        return {
            "tick": self.active_tick,
            "mode": cognitive_mode,
            "energy_state": reg_status,
            "broadcast": broadcast_packet,
            "stream_coherence": self.stream.get_thematic_coherence(),
            "temporal": self.temporal_awareness.get_time_metrics(),
            "planning": planning_status,
            "execution": execution_results,
            "reasoning": reasoning_insights
        }

    def process_percept(self, perception_input: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper method for processing percept inputs, invoking execute_cognitive_tick."""
        return self.execute_cognitive_tick(perception_input)
