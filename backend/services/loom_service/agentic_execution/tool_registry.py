from typing import Dict, Any, List, Optional

class ToolRegistry:
    """
    Declares and registers available tools alongside their JSON-schemas for agent consumption.
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, name: str, description: str, parameters: Dict[str, Any]) -> None:
        """Adds a tool definition to the registry."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters
        }

    def get_tool_definition(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific tool's definition."""
        return self.tools.get(name)

    def list_all_tools(self) -> List[Dict[str, Any]]:
        """Lists all registered tools."""
        return list(self.tools.values())
