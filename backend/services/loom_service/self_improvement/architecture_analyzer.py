from typing import List, Dict, Any

class ArchitectureAnalyzer:
    """
    Parses and analyzes cognitive logs/traces to pinpoint processing bottlenecks or loops.
    """
    def __init__(self):
        pass

    def analyze_traces(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identifies slow component triggers or cycles in subsystem events."""
        duration_by_subsystem = {}
        loop_candidate_count = {}

        for tr in traces:
            sub = tr.get("subsystem", "unknown")
            dur = tr.get("duration", 0.0)
            action = tr.get("action", "")

            # Track durations
            duration_by_subsystem[sub] = duration_by_subsystem.get(sub, 0.0) + dur

            # Simple heuristic for tracking potential repetitive actions (loops)
            action_key = (sub, action)
            loop_candidate_count[action_key] = loop_candidate_count.get(action_key, 0) + 1

        # Check for components with excessively repetitive execution signatures
        loops_detected = []
        for (sub, act), count in loop_candidate_count.items():
            if count >= 5:  # Arbitrary repetition threshold
                loops_detected.append({"subsystem": sub, "action": act, "count": count})

        # Find slowest subsystem
        bottleneck = None
        max_duration = -1.0
        for sub, total_dur in duration_by_subsystem.items():
            if total_dur > max_duration:
                max_duration = total_dur
                bottleneck = sub

        return {
            "slowest_subsystem": bottleneck,
            "total_subsystem_durations": duration_by_subsystem,
            "loops_detected": loops_detected
        }
