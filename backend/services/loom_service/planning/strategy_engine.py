from typing import Dict, Any

class StrategyEngine:
    """
    Modulates planning parameters (e.g. exploration breadth, selection heuristics)
    based on current cognitive mode and task success history.
    """
    def __init__(self):
        pass
        
    def determine_strategy(self, cognitive_mode: str, history_metrics: Dict[str, Any]) -> Dict[str, Any]:
        # Default strategy parameters
        strategy = {
            "search_breadth": 2,
            "max_depth": 3,
            "backtracking_allowed": True,
            "heuristic": "expected_utility"
        }
        
        if cognitive_mode == "exploration":
            strategy["search_breadth"] = 4
            strategy["max_depth"] = 2
            strategy["heuristic"] = "surprise_minimization"
        elif cognitive_mode == "recovery":
            strategy["search_breadth"] = 1
            strategy["max_depth"] = 1
            strategy["backtracking_allowed"] = False
            strategy["heuristic"] = "energy_conservation"
        elif cognitive_mode == "focused_reasoning":
            strategy["search_breadth"] = 2
            strategy["max_depth"] = 5
            strategy["heuristic"] = "strict_preconditions"
            
        # Adjust based on historical success rate
        success_rate = history_metrics.get("success_rate", 1.0)
        if success_rate < 0.6:
            # Plan more cautiously
            strategy["search_breadth"] = max(1, strategy["search_breadth"] - 1)
            strategy["max_depth"] = min(6, strategy["max_depth"] + 1)
            
        return strategy
