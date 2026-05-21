import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ThoughtHabit:
    """A recurring sequence of cognitive actions."""
    habit_id: str
    sequence: List[str]
    success_rate: float = 1.0
    activation_count: int = 0

class ThoughtPrograms:
    """COGNITIVE HABITS: Automates recurring reasoning patterns."""
    def __init__(self):
        self.library: Dict[str, ThoughtHabit] = {}
        self.active_program: Optional[str] = None
        
    def learn_habit(self, transition_history: List[tuple]):
        if len(transition_history) < 3: return
        recent = [t[1] for t in transition_history[-3:]]
        habit_id = f"habit_{'_'.join(recent)}"
        if habit_id not in self.library:
            self.library[habit_id] = ThoughtHabit(habit_id, recent)
        else:
            self.library[habit_id].activation_count += 1

    def execute_step(self) -> Optional[str]:
        if not self.active_program or self.active_program not in self.library: return None
        return self.library[self.active_program].sequence[0] 

class Reflection:
    """SELF-CORRECTION: Monitoring cognitive health."""
    def __init__(self, pressure_threshold=0.8, pressure_decay=0.85):
        self.pressure_threshold = pressure_threshold
        self.pressure_decay = pressure_decay
        self.quality_history: List[float] = []
        self.current_pressure = 0.0
        
        # Self-evaluation metrics
        self.reasoning_quality = 0.5
        self.epistemic_score = 0.5
        self.prediction_accuracy = 0.5
        self.cognitive_mode = "exploration" # exploration, focused_reasoning, recovery, abstraction, prediction
        self.history = []
        # ISSUE 17: Continuous Self-Model
        self.identity_field = None
        self.history_metrics = {
            "epistemic": [], "reasoning": [], "prediction": [], "goal": []
        }

    def analyze_self(self, state: Dict[str, Any], latent_field=None) -> Dict[str, Any]:
        """
        Evaluates system performance to modulate hyper-parameters.
        Now includes continuous self-model identity tracking.
        """
        entropy = state.get("entropy", 0.5)
        coherence = state.get("coherence", 0.5)
        surprise = state.get("surprise", 0.5)
        
        import numpy as np
        identity_dissonance = 0.0
        if latent_field is not None:
            if self.identity_field is None:
                self.identity_field = latent_field.copy()
            else:
                self.identity_field = 0.95 * self.identity_field + 0.05 * latent_field
                inorm = np.linalg.norm(self.identity_field)
                if inorm > 0: self.identity_field /= inorm
                identity_dissonance = float(np.linalg.norm(latent_field - self.identity_field))
        energy = state.get("energy", 1.0)
        
        # Self-evaluation updates
        self.prediction_accuracy = 1.0 - surprise
        self.epistemic_score = coherence * (1.0 - entropy)
        self.reasoning_quality = self.epistemic_score * self.prediction_accuracy
        
        self.history_metrics["prediction"].append(self.prediction_accuracy)
        self.history_metrics["epistemic"].append(self.epistemic_score)
        self.history_metrics["reasoning"].append(self.reasoning_quality)
        for k in self.history_metrics:
            if len(self.history_metrics[k]) > 50: self.history_metrics[k].pop(0)
        
        # FIX: Normalize pressure components to prevent permanent saturation
        raw_pressure = (0.3 * entropy) + (0.2 * (1.0 - coherence)) + (0.3 * surprise) + (0.2 * identity_dissonance)
        
        # FIX: Introduce pressure decay (EMA) for stability
        self.current_pressure = (self.current_pressure * self.pressure_decay) + (raw_pressure * (1.0 - self.pressure_decay))
        pressure = self.current_pressure
        
        confidence = (0.4 * coherence) + (0.3 * state.get("stability", 1.0)) + (0.3 * (1.0 - surprise))
        self.quality_history.append(float(confidence))
        if len(self.quality_history) > 50: self.quality_history.pop(0)
        
        # Cognitive mode switching
        if pressure > 0.8: self.cognitive_mode = "recovery"
        elif surprise > 0.7: self.cognitive_mode = "exploration"
        elif entropy < 0.3 and coherence > 0.7: self.cognitive_mode = "abstraction"
        elif energy > 0.8: self.cognitive_mode = "focused_reasoning"
        else: self.cognitive_mode = "prediction"
        
        modulation = {
            "increase_inhibition": pressure > 0.5,
            "rebucket_pressure": pressure > 0.7,
            "attention_shift": pressure > 0.6 or surprise > 0.7,
            "memory_reset": pressure > 0.9 and confidence < 0.3,
            "mode": self.cognitive_mode
        }
        
        return {
            "confidence": confidence,
            "reflection_pressure": pressure,
            "trigger_correction": pressure > self.pressure_threshold,
            "modulation_signals": modulation,
            "metrics": {
                "epistemic": self.epistemic_score,
                "reasoning": self.reasoning_quality,
                "prediction": self.prediction_accuracy,
                "identity_dissonance": identity_dissonance
            }
        }
