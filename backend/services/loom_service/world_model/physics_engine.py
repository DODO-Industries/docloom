from typing import Dict, Any, List

class PhysicsEngine:
    """
    Qualitative Physics Engine. Simulates containment, gravity, support,
    and state changes (e.g. file deletion propagating through dependencies).
    """
    def __init__(self):
        pass
        
    def simulate_spill(self, obj_state: Dict[str, Any]) -> Dict[str, Any]:
        """Spilling liquid propagates to containment objects."""
        new_state = dict(obj_state)
        if obj_state.get("type") == "cup" and obj_state.get("filled_with") and obj_state.get("tipped", False):
            liquid = new_state.pop("filled_with")
            new_state["spilled"] = True
            new_state["environment_status"] = f"{liquid}_spilled_on_table"
        return new_state
        
    def simulate_gravity(self, object_properties: Dict[str, Any]) -> Dict[str, Any]:
        """Gravity affects unsupported physical objects."""
        new_properties = dict(object_properties)
        if object_properties.get("physical", False) and not object_properties.get("supported", False):
            new_properties["position_y"] = 0.0  # Falls to base level
            new_properties["velocity_y"] = -9.81
            new_properties["status"] = "fallen"
        return new_properties
        
    def simulate_dependency_cascade(self, action: str, target: str, system_graph: Any) -> List[str]:
        """Predicts what other components are affected if target undergoes action (e.g. deletion)."""
        affected = []
        if action == "delete" and hasattr(system_graph, "graph"):
            # Find downstream dependents (nodes that have 'depends_on' target)
            for u, v, data in system_graph.graph.edges(data=True):
                if v == target and data.get("relation") == "depends_on":
                    affected.append(u)
        return affected
