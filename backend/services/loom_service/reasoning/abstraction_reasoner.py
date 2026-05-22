from typing import List, Dict, Any

class AbstractionReasoner:
    """
    Identifies patterns in logs or observations to generalize rules or schemas.
    """
    def __init__(self):
        pass

    def generalize_rule(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts common features and patterns from multiple concrete event instances.
        For example: if all instances have action='read_file' and file_ext='csv',
        generalizes to rule: 'reads csv files'.
        """
        if not instances:
            return {}

        common_keys = set(instances[0].keys())
        for inst in instances[1:]:
            common_keys.intersection_update(inst.keys())

        generalized_properties = {}
        for key in common_keys:
            vals = {inst[key] for inst in instances if key in inst}
            if len(vals) == 1:
                generalized_properties[key] = next(iter(vals))

        return {
            "generalized_properties": generalized_properties,
            "instances_count": len(instances),
            "confidence": 0.5 + (0.1 * min(len(instances), 5))
        }
