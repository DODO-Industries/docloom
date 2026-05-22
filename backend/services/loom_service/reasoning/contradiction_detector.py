from typing import List, Dict, Any, Tuple

class ContradictionDetector:
    """
    Scans working memory or the logic graph to detect logical inconsistencies
    (e.g., A is True and NOT A is also True).
    """
    def __init__(self):
        pass

    def detect_contradictions(self, facts: Dict[str, Any]) -> List[Tuple[str, Any, Any]]:
        """
        Checks for conflicting state values or inverted beliefs.
        Returns a list of contradiction tuples: (fact_name, value_a, value_b).
        """
        contradictions = []
        for key, val in facts.items():
            # Check for standard contradiction (e.g. active flag and inactive flag)
            if key.startswith("not_"):
                positive_key = key[4:]
                if positive_key in facts:
                    if facts[positive_key] == val:  # e.g. A and not_A are both True
                        contradictions.append((positive_key, facts[positive_key], val))
            
            # Check boolean values that conflict with direct negation keys
            if isinstance(val, bool):
                opposite_key = f"not_{key}"
                if opposite_key in facts and facts[opposite_key] == val:
                    contradictions.append((key, val, facts[opposite_key]))
        return contradictions
