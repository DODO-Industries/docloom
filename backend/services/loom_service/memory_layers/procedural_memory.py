from typing import Dict, List, Any

class ProceduralMemory:
    """
    Stores motor/agentic routines, recipes, and success-validated task schemas.
    """
    def __init__(self):
        # Maps task name -> list of actions/steps
        self.procedures: Dict[str, List[str]] = {}

    def save_procedure(self, name: str, steps: List[str]) -> None:
        """Stores a sequence of action steps under a procedure name key."""
        self.procedures[name] = list(steps)

    def retrieve_procedure(self, name: str) -> List[str]:
        """Retrieves action recipe steps for a specific procedure."""
        return self.procedures.get(name, [])
