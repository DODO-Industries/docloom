from typing import Dict, List, Any, Callable

class SubsystemBus:
    """
    Coordinates event communication across subsystems via standard pub-sub message routing.
    """
    def __init__(self):
        # Maps event_type -> list of subscriber callables
        self.subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}

    def subscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribes a callback to receive events of a specific type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    def publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Broadcasts an event message payload to all subscribers of the type."""
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(payload)
                except Exception:
                    pass  # Prevent a failing subscriber from breaking the bus
