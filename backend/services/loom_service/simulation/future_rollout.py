from typing import Dict, Any, List

class FutureRollout:
    """
    Executes forward projection loops (rolling out state updates multi-hop).
    """
    def __init__(self):
        pass

    def project_trajectory(self, start_state: Dict[str, Any], actions: List[str], step_simulator: Any) -> List[Dict[str, Any]]:
        """
        Projects state changes step-by-step for a chain of actions.
        step_simulator is a causal simulator instance.
        """
        trajectory = []
        curr_state = dict(start_state)

        for act in actions:
            # We mock the params matching the simulator pattern
            next_state = step_simulator.simulate_action_effects(curr_state, act, {})
            trajectory.append({
                "action": act,
                "resulting_state": next_state
            })
            curr_state = next_state

        return trajectory
