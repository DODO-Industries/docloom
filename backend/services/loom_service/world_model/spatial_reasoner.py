from typing import List, Dict, Any

class SpatialReasoner:
    """
    reasons about bounding boxes (coordinates, overlapping regions,
    containment, spatial margins) of parsed document elements or physical entities.
    """
    def __init__(self):
        pass
        
    def check_containment(self, inner_bbox: List[float], outer_bbox: List[float]) -> bool:
        """Determines if inner_bbox is fully contained inside outer_bbox."""
        if not inner_bbox or not outer_bbox or len(inner_bbox) < 4 or len(outer_bbox) < 4:
            return False
        # format: [x0, y0, x1, y1]
        return (inner_bbox[0] >= outer_bbox[0] and
                inner_bbox[1] >= outer_bbox[1] and
                inner_bbox[2] <= outer_bbox[2] and
                inner_bbox[3] <= outer_bbox[3])
                
    def check_overlap(self, bbox_a: List[float], bbox_b: List[float]) -> float:
        """Computes the Jaccard-like overlap area between two bounding boxes."""
        if not bbox_a or not bbox_b or len(bbox_a) < 4 or len(bbox_b) < 4:
            return 0.0
            
        x_left = max(bbox_a[0], bbox_b[0])
        y_top = max(bbox_a[1], bbox_b[1])
        x_right = min(bbox_a[2], bbox_b[2])
        y_bottom = min(bbox_a[3], bbox_b[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
        area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
        
        union_area = area_a + area_b - intersection_area
        return float(intersection_area / union_area) if union_area > 0 else 0.0
