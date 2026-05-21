import numpy as np
from typing import Dict, Any

class CognitiveStateMachine:
    """
    Tracks and regulates the system's cognitive state mode using
    state transition rules and historical stability.
    
    Modes:
      - exploration: Searching for new links and concepts.
      - focused_reasoning: Running deep logic/symbolic reasoning.
      - planning: Building HTN plans and running rollouts.
      - execution: Using tools and running actions.
      - recovery: Cooling down under high cognitive pressure/entropy.
    """
    def __init__(self, inertia: float = 0.4):
        self.current_state = "exploration"
        self.inertia = inertia
        self.state_durations = {"exploration": 0, "focused_reasoning": 0, "planning": 0, "execution": 0, "recovery": 0}
        
    def transition(self, metrics: Dict[str, float]) -> str:
        entropy = metrics.get("entropy", 0.5)
        coherence = metrics.get("coherence", 0.5)
        surprise = metrics.get("surprise", 0.0)
        pressure = metrics.get("pressure", 0.0)
        energy = metrics.get("energy", 1.0)
        
        # State transitions
        next_state = self.current_state
        
        if pressure > 0.8:
            next_state = "recovery"
        elif self.current_state == "recovery":
            if pressure < 0.4 and energy > 0.5:
                next_state = "exploration"
        elif self.current_state == "exploration":
            if surprise > 0.7:
                # Surprise indicates novelty, stay in exploration to map it
                next_state = "exploration"
            elif coherence > 0.7 and energy > 0.6:
                next_state = "focused_reasoning"
            elif metrics.get("has_goals", 0.0) > 0:
                next_state = "planning"
        elif self.current_state == "focused_reasoning":
            if entropy > 0.6 or energy < 0.3:
                next_state = "recovery"
            elif metrics.get("has_goals", 0.0) > 0:
                next_state = "planning"
        elif self.current_state == "planning":
            if metrics.get("plan_ready", 0.0) > 0:
                next_state = "execution"
            elif energy < 0.2:
                next_state = "recovery"
        elif self.current_state == "execution":
            if metrics.get("execution_done", 0.0) > 0:
                next_state = "exploration"
            elif surprise > 0.8:  # unexpected execution failure
                next_state = "focused_reasoning"
            elif energy < 0.15:
                next_state = "recovery"
                
        # Apply inertia/stickiness
        if next_state != self.current_state and np.random.rand() < self.inertia:
            # Revert change due to inertia
            pass
        else:
            self.current_state = next_state
            
        for s in self.state_durations:
            if s == self.current_state:
                self.state_durations[s] += 1
            else:
                self.state_durations[s] = 0
                
        return self.current_state
        
    def get_status(self) -> Dict[str, Any]:
        return {
            "current_state": self.current_state,
            "durations": self.state_durations
        }
