from typing import List, Dict, Any

class TextAligner:
    """
    Coordinates spatial alignment between visual bounding boxes and corresponding text segments.
    """
    def __init__(self):
        pass

    def align_text_to_layout(self, ocr_results: List[Dict[str, Any]], layout_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Matches OCR words/phrases to logical layout blocks (headers, paragraphs, tables)."""
        alignments = []
        for text_item in ocr_results:
            text_bbox = text_item.get("bbox", [0, 0, 0, 0])
            best_match = None
            best_overlap = 0.0

            for block in layout_blocks:
                block_bbox = block.get("bbox", [0, 0, 0, 0])
                overlap = self._compute_bbox_overlap(text_bbox, block_bbox)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = block.get("id")

            alignments.append({
                "text": text_item.get("text"),
                "aligned_to_block_id": best_match,
                "overlap_ratio": best_overlap
            })
        return alignments

    def _compute_bbox_overlap(self, box_a: List[float], box_b: List[float]) -> float:
        """Calculates area overlap of two boxes."""
        x_left = max(box_a[0], box_b[0])
        y_top = max(box_a[1], box_b[1])
        x_right = min(box_a[2], box_b[2])
        y_bottom = min(box_a[3], box_b[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        return intersection / max(area_a, 1e-6)
