from typing import Dict, Any, List

class MultimodalMemory:
    """
    Temporary sensory buffer storing raw visual features, audio transcripts, and aligned layouts.
    """
    def __init__(self):
        self.sensory_register: Dict[str, List[Dict[str, Any]]] = {
            "visual": [],
            "auditory": [],
            "aligned_text": []
        }

    def store_sensory_input(self, modality: str, data: Dict[str, Any]) -> None:
        """Saves sensory inputs to the registry."""
        if modality in self.sensory_register:
            self.sensory_register[modality].append(data)
            # Cap at 50 to prevent memory blowups
            if len(self.sensory_register[modality]) > 50:
                self.sensory_register[modality].pop(0)

    def retrieve_modality_data(self, modality: str) -> List[Dict[str, Any]]:
        """Retrieves raw files/embeddings stored for the selected modality."""
        return self.sensory_register.get(modality, [])
