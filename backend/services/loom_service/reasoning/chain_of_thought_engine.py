from typing import List, Dict, Any

class ChainOfThoughtEngine:
    """
    Manages, logs, and structures step-by-step logical expansions during problem solving.
    """
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []

    def start_thought_chain(self, root_problem: str) -> None:
        """Initializes a new reasoning chain."""
        self.steps = [{"step": 0, "thought": f"Root Problem: {root_problem}", "justification": "Init"}]

    def append_thought_step(self, thought: str, justification: str) -> None:
        """Adds a reasoning step to the chain."""
        step_num = len(self.steps)
        self.steps.append({
            "step": step_num,
            "thought": thought,
            "justification": justification
        })

    def get_reasoning_trace(self) -> List[Dict[str, Any]]:
        """Returns the full sequential trace of thoughts."""
        return self.steps
