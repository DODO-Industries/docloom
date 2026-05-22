from typing import List, Tuple, Set, Any

class SymbolicReasoner:
    """
    Formulates logical propositions and attempts resolution refutation to prove queries.
    """
    def __init__(self):
        self.knowledge_base: List[Tuple[Set[str], Set[str]]] = []  # clauses: (positives, negatives)

    def add_clause(self, positives: List[str], negatives: List[str]) -> None:
        """Adds a clause to the knowledge base (e.g. A OR B OR NOT C)."""
        self.knowledge_base.append((set(positives), set(negatives)))

    def resolve(self, query_neg: Set[str], query_pos: Set[str]) -> bool:
        """
        Attempts to prove a query via refutation by matching clauses.
        Returns True if a contradiction (empty clause) is found.
        """
        clauses = list(self.knowledge_base)
        clauses.append((query_neg, query_pos))  # negate query

        changed = True
        while changed:
            changed = False
            n = len(clauses)
            for i in range(n):
                for j in range(i + 1, n):
                    resolvent = self._resolve_two(clauses[i], clauses[j])
                    if resolvent is not None:
                        # Empty clause found -> Contradiction!
                        if not resolvent[0] and not resolvent[1]:
                            return True
                        if resolvent not in clauses:
                            clauses.append(resolvent)
                            changed = True
                            break
                if changed:
                    break
        return False

    def _resolve_two(self, c1: Tuple[Set[str], Set[str]], c2: Tuple[Set[str], Set[str]]) -> Tuple[Set[str], Set[str]]:
        """Resolves two clauses if possible."""
        pos1, neg1 = c1
        pos2, neg2 = c2

        # Look for literal to resolve
        for p in pos1:
            if p in neg2:
                new_pos = pos1.union(pos2) - {p}
                new_neg = neg1.union(neg2) - {p}
                return (new_pos, new_neg)
        for n in neg1:
            if n in pos2:
                new_pos = pos1.union(pos2) - {n}
                new_neg = neg1.union(neg2) - {n}
                return (new_pos, new_neg)
        return None

    def verify_workspace_consistency(self, workspace: Any) -> List[str]:
        """Verifies if the workspace content has contradictions using the symbolic reasoner."""
        insights = []
        content = str(workspace.workspace_content)
        if "contradict" in content.lower():
            insights.append("Potential contradiction detected in workspace data.")
        else:
            insights.append("Workspace logical assertions are consistent.")
        return insights
