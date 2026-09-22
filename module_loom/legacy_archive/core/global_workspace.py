from typing import Dict, Any, List, Optional
import time

class GlobalWorkspace:
    """
    Implements Baars' Global Workspace Theory (GWT).
    Subsystems write coalition items with a given salience.
    The workspace broadcasts the highest-salience item to all modules.
    """
    def __init__(self, salience_decay: float = 0.9):
        self.workspace_content: Dict[str, Dict[str, Any]] = {}
        self.salience_decay = salience_decay
        self.broadcast_history: List[Dict[str, Any]] = []
        
    def publish(self, sender: str, content: Any, salience: float = 1.0):
        self.workspace_content[sender] = {
            "content": content,
            "salience": salience,
            "timestamp": time.time()
        }
        
    def broadcast(self) -> Optional[Dict[str, Any]]:
        """Broadcasts the message with the highest salience."""
        if not self.workspace_content:
            return None
            
        # Find dominant sender
        dominant_sender = max(self.workspace_content.keys(), key=lambda k: self.workspace_content[k]["salience"])
        winning_coalition = self.workspace_content[dominant_sender]
        
        broadcast_packet = {
            "sender": dominant_sender,
            "content": winning_coalition["content"],
            "salience": winning_coalition["salience"],
            "timestamp": winning_coalition["timestamp"]
        }
        
        self.broadcast_history.append(broadcast_packet)
        if len(self.broadcast_history) > 100:
            self.broadcast_history.pop(0)
            
        # Decay all items
        for sender in list(self.workspace_content.keys()):
            self.workspace_content[sender]["salience"] *= self.salience_decay
            if self.workspace_content[sender]["salience"] < 0.1:
                del self.workspace_content[sender]
                
        return broadcast_packet
        
    def get_content_by_sender(self, sender: str) -> Optional[Any]:
        if sender in self.workspace_content:
            return self.workspace_content[sender]["content"]
        return None
        
    def clear(self):
        self.workspace_content.clear()

    def publish_event(self, event_name: str, data: Any):
        """Compatibility wrapper for publishing events."""
        self.publish(sender=event_name, content=data, salience=1.0)
