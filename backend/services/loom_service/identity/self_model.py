from typing import Dict, Any, List

class SelfModel:
    """
    Maintains the persistent definition of the cognitive entity itself:
    capabilities, metadata, constraints, and current self-evaluation states.
    """
    def __init__(self, name: str = "DocLoom_Cognitive_Agent"):
        self.name = name
        self.version = "4.0.0"
        self.capabilities = ["text_extraction", "semantic_weaving", "symbolic_reasoning", "hierarchical_planning"]
        self.active_role = "research_assistant"
        self.metrics_snapshot = {
            "average_reasoning_quality": 0.8,
            "epistemic_level": 0.85,
            "adaptability": 0.9
        }
        
    def check_capability(self, action_name: str) -> bool:
        """Determines if the agent is equipped to perform the action."""
        # Simple capability matching
        for cap in self.capabilities:
            if cap in action_name or action_name in cap:
                return True
        return False
        
    def get_summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "role": self.active_role,
            "capabilities": self.capabilities,
            "performance": self.metrics_snapshot
        }
