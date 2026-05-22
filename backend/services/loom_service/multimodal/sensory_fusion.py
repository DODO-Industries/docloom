from typing import Dict, Any, List

class SensoryFusion:
    """
    Fuses outputs from text parsing, speech audio, and visual layout bounding boxes
    into a single unified perceptual state.
    """
    def __init__(self):
        pass

    def fuse_signals(self, visual_data: Dict[str, Any], audio_data: Dict[str, Any], text_data: List[str]) -> Dict[str, Any]:
        """Integrates all modalities to produce a consolidated contextual frame."""
        fused_context = {
            "primary_focus": audio_data.get("text", "") or (text_data[0] if text_data else ""),
            "spatial_anchors": visual_data.get("detected_objects", []),
            "confidence_score": min(visual_data.get("confidence", 1.0), audio_data.get("confidence", 1.0)),
            "modalities_present": []
        }
        
        if visual_data:
            fused_context["modalities_present"].append("visual")
        if audio_data:
            fused_context["modalities_present"].append("audio")
        if text_data:
            fused_context["modalities_present"].append("text")
            
        return fused_context
