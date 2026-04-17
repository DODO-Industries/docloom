import json
import re
from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("SemanticLinker")

class SemanticLinker:
    @staticmethod
    def link_semantic_context(docloom_json, vertical_margin=60, horizontal_tolerance=20):
        """
        Scans the extracted DocLoom JSON and injects surrounding text 
        (captions, paragraphs) directly into the image data nodes based on spatial proximity.
        """
        log_service(logger, "Initializing Spatial Semantic Linking...", "info")
        
        if isinstance(docloom_json, list):
            pages = docloom_json
        else:
            pages = docloom_json.get("pages", [])

        images_linked = 0

        for page in pages:
            content = page.get("content", []) 
            
            text_blocks = [item for item in content if item.get("type") in ["paragraph", "heading"]]
            images = [item for item in content if item.get("type") == "image"]

            # Process each image one by one
            for img in images:
                img_bbox = img.get("bbox")
                
                # --- THE RESCUE MISSION: Extract bbox from metadata if missing ---
                if not img_bbox and "metadata" in img:
                    match = re.search(r"bbox=\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]", img["metadata"])
                    if match:
                        img_bbox = [
                            float(match.group(1)), 
                            float(match.group(2)), 
                            float(match.group(3)), 
                            float(match.group(4))
                        ]
                
                # If we still don't have coordinates, skip it
                if not img_bbox or len(img_bbox) != 4: 
                    continue
                
                img_x0, img_top, img_x1, img_bottom = img_bbox[0], img_bbox[1], img_bbox[2], img_bbox[3]
                
                context_above = []
                context_below = []

                for text in text_blocks:
                    txt_bbox = text.get("bbox")
                    if not txt_bbox or len(txt_bbox) != 4: continue
                    
                    txt_x0, txt_top, txt_x1, txt_bottom = txt_bbox[0], txt_bbox[1], txt_bbox[2], txt_bbox[3]
                    
                    # --- THE COLUMN LOCK (With Tolerance) ---
                    img_left_expanded = img_x0 - horizontal_tolerance
                    img_right_expanded = img_x1 + horizontal_tolerance

                    horizontal_overlap = max(img_left_expanded, txt_x0) < min(img_right_expanded, txt_x1)
                    
                    if not horizontal_overlap:
                        continue # Skip this text, it's in a different column

                    # --- THE PROXIMITY CHECK ---
                    # Calculate the exact pixel gap between the blocks
                    gap_above = img_top - txt_bottom
                    gap_below = txt_top - img_bottom

                    # Allow up to -15 pixels of overlap/bleed!
                    if -15 <= gap_above <= vertical_margin:
                        context_above.append(text.get("text", ""))
                        
                    elif -15 <= gap_below <= vertical_margin:
                        context_below.append(text.get("text", ""))

                # --- INJECT INTO JSON ---
                if context_above or context_below:
                    img["semantic_context"] = {
                        "text_above": "\n".join(context_above).strip(), 
                        "text_below": "\n".join(context_below).strip()
                    }
                    images_linked += 1

        log_service(logger, f"Successfully linked semantic context for {images_linked} images.", "info")
        return docloom_json