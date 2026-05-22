import time
from typing import List, Dict, Any

class AutobiographicalMemory:
    """
    Chronological registry of significant cognitive events, milestones,
    successes, and failures in the agent's life cycle.
    """
    def __init__(self, max_records: int = 500):
        self.records: List[Dict[str, Any]] = []
        self.max_records = max_records
        
    def log_event(self, event_type: str, description: str, impact_score: float = 0.5):
        record = {
            "timestamp": time.time(),
            "type": event_type,
            "description": description,
            "impact": impact_score
        }
        self.records.append(record)
        # Prune if exceeding bounds, keeping high-impact events first
        if len(self.records) > self.max_records:
            # Sort by impact, keep top 80% and chronological remainder
            self.records.sort(key=lambda x: x["impact"], reverse=True)
            high_impact = self.records[:int(self.max_records * 0.8)]
            # Restore chronological order for the subset
            high_impact.sort(key=lambda x: x["timestamp"])
            self.records = high_impact
            
    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.records[-limit:]
