from typing import Dict, List, Set, Optional

class TheoremEngine:
    """
    Executes forward and backward chaining over production rules (if conditions then conclusion).
    """
    def __init__(self):
        # Rules format: { "id": "rule1", "if": ["A", "B"], "then": "C" }
        self.rules: List[Dict[str, Any]] = []

    def register_rule(self, rule_id: str, conditions: List[str], conclusion: str) -> None:
        """Registers an implication rule."""
        self.rules.append({"id": rule_id, "if": conditions, "then": conclusion})

    def forward_chaining(self, facts: Set[str]) -> Set[str]:
        """Runs forward chaining from initial facts to derive all reachable conclusions."""
        known_facts = set(facts)
        agenda = list(facts)

        while agenda:
            curr = agenda.pop(0)
            for rule in self.rules:
                if curr in rule["if"]:
                    # Check if all conditions are satisfied
                    if all(cond in known_facts for cond in rule["if"]):
                        conclusion = rule["then"]
                        if conclusion not in known_facts:
                            known_facts.add(conclusion)
                            agenda.append(conclusion)
        return known_facts

    def backward_chaining(self, target: str, facts: Set[str]) -> bool:
        """Determines if a target fact can be proven given existing facts via backward chaining."""
        if target in facts:
            return True

        for rule in self.rules:
            if rule["then"] == target:
                # To prove target, we must prove all conditions
                all_conditions_met = True
                for cond in rule["if"]:
                    if not self.backward_chaining(cond, facts):
                        all_conditions_met = False
                        break
                if all_conditions_met:
                    return True
        return False
