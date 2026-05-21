from typing import Dict, Any

class PolicyOptimizer:
    """
    Optimizes priority schedulers and state transition parameters dynamically.
    """
    def __init__(self):
        pass

    def optimize_priorities(self, current_weights: Dict[str, float], performance_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Adjusts scheduled subsystem allocation weights based on performance stats."""
        updated_weights = dict(current_weights)
        slow_subsystem = performance_metrics.get("slowest_subsystem")
        loops = performance_metrics.get("loops_detected", [])

        # If a subsystem is a bottleneck, allocate more priority weight so it gets processed faster
        if slow_subsystem and slow_subsystem in updated_weights:
            updated_weights[slow_subsystem] = min(updated_weights[slow_subsystem] * 1.2, 1.0)

        # If cycles/loops are happening in a subsystem, penalize its scheduling weight to throttle execution
        for loop in loops:
            sub = loop.get("subsystem")
            if sub in updated_weights:
                updated_weights[sub] = max(updated_weights[sub] * 0.8, 0.1)

        return updated_weights
