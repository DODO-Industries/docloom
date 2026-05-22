from typing import List, Dict, Any

class HallucinationDetector:
    """
    Cross-checks assertions and LLM output statements against facts in verified semantic memory.
    """
    def __init__(self):
        pass

    def check_statement(self, statement: Dict[str, Any], semantic_memory: Any) -> Dict[str, Any]:
        """
        Cross-checks a statement fact against the memory database.
        statement format: {"subject": "A", "predicate": "related_to", "object": "B"}
        """
        sub = statement.get("subject", "")
        pred = statement.get("predicate", "")
        obj = statement.get("object", "")

        memory_record = semantic_memory.query_concept(sub)
        if not memory_record:
            return {"verified": False, "reason": f"Subject '{sub}' not found in semantic memory."}

        # Check relationships
        relations = memory_record.get("relations", [])
        for rel in relations:
            if rel.get("to") == obj and rel.get("type") == pred:
                return {"verified": True, "reason": "Exact relation match in memory."}

        return {
            "verified": False,
            "reason": f"Relation {sub} --({pred})--> {obj} is unsupported by memory state."
        }
