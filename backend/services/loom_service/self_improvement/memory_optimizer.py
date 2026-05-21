from typing import Dict, Any, List

class MemoryOptimizer:
    """
    Performs cleanup of weak memory connections, aggregates episodic memory points, and manages indices.
    """
    def __init__(self):
        pass

    def optimize_memory_links(self, memory_links: List[Dict[str, Any]], threshold: float = 0.2) -> List[Dict[str, Any]]:
        """Removes memory link connections below the target activation strength threshold."""
        pruned_links = []
        for link in memory_links:
            strength = link.get("strength", 1.0)
            # Prune if strength falls below threshold
            if strength >= threshold:
                pruned_links.append(link)
        return pruned_links

    def consolidate_shards(self, episodic_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merges multiple episodic logs into a unified semantic overview summary."""
        # Simple string concatenation representing semantic consolidation
        themes = set()
        total_excitement = 0.0
        
        for trace in episodic_traces:
            action = trace.get("action", "")
            if action:
                themes.add(action)
            total_excitement += trace.get("salience", trace.get("excitement", 0.5))

        avg_salience = (total_excitement / len(episodic_traces)) if episodic_traces else 0.0

        return {
            "consolidated_concepts": list(themes),
            "average_salience": avg_salience,
            "consolidated_count": len(episodic_traces)
        }
