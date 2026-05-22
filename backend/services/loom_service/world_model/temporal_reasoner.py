from typing import Dict, Any, List

class TemporalReasoner:
    """
    Evaluates temporal relationships and bounds (before, after, during, overlaps)
    between events or time intervals in working memory.
    """
    def __init__(self):
        pass

    def evaluate_relation(self, interval_a: Dict[str, float], interval_b: Dict[str, float]) -> str:
        """
        Determines the relation (Allen's interval algebra subset) between two intervals.
        Each interval is a dictionary with 'start' and 'end' keys.
        """
        start_a, end_a = interval_a.get("start", 0.0), interval_a.get("end", 0.0)
        start_b, end_b = interval_b.get("start", 0.0), interval_b.get("end", 0.0)

        if end_a < start_b:
            return "before"
        elif start_a > end_b:
            return "after"
        elif start_a >= start_b and end_a <= end_b:
            return "during"
        elif start_a <= start_b and end_a >= end_b:
            return "contains"
        elif start_a < start_b and end_a > start_b and end_a < end_b:
            return "overlaps_before"
        elif start_a > start_b and start_a < end_b and end_a > end_b:
            return "overlaps_after"
        elif start_a == start_b and end_a == end_b:
            return "equals"
        return "complex"

    def sequence_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sorts a list of event dicts based on their timestamp or start field."""
        return sorted(events, key=lambda x: x.get("timestamp", x.get("start", 0.0)))
