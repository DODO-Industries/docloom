from typing import Dict, Any, Callable

class EventRouter:
    """
    Routes specific event signals to target handler components in working memory.
    """
    def __init__(self):
        self.route_map: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    def register_route(self, event_name: str, handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """Binds a specific event type to its executor function."""
        self.route_map[event_name] = handler

    def route_event(self, event_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Routes event to target handler and returns standard execution mapping feedback."""
        if event_name not in self.route_map:
            return {"routed": False, "error": f"No registered handler for event '{event_name}'"}
            
        try:
            res = self.route_map[event_name](payload)
            return {"routed": True, "error": None, "result": res}
        except Exception as e:
            return {"routed": False, "error": str(e)}
