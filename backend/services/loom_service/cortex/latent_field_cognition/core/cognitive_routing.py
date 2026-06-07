from typing import Dict, Any, List

class CognitiveRouter:
    """
    Analyzes semantic patterns, concept metadata, and input schemas
    to route tasks to specialized cognitive subcomponents.
    """
    def __init__(self):
        # Specific concepts that trigger specific routes
        self.multimodal_keywords = {"image", "picture", "diagram", "draw", "plot", "audio", "voice", "sound", "listen"}
        self.symbolic_keywords = {"prove", "solve", "math", "logical", "contradicts", "consistent", "theorem", "calculate"}
        self.planning_keywords = {"build", "create", "construct", "plan", "goal", "execute", "run", "do"}
        
    def route(self, input_data: Dict[str, Any]) -> str:
        text = str(input_data.get("text", "")).lower()
        
        # Check explicit multimodal payloads
        if "image_data" in input_data or "audio_data" in input_data:
            return "multimodal"
            
        # Parse keywords
        words = set(text.split())
        
        if words.intersection(self.multimodal_keywords):
            return "multimodal"
        elif words.intersection(self.symbolic_keywords):
            return "symbolic_reasoning"
        elif words.intersection(self.planning_keywords):
            return "planning"
            
        # Default route is standard semantic graph retrieval and workspace consolidation
        return "semantic_substrate"
