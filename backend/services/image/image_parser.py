import base64
from io import BytesIO
from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("ImageParser")

class ImageParser:
    
    @staticmethod
    def _merge_overlapping_boxes(boxes, merge_margin=5.0):
        """
        Uses spatial clustering to merge adjacent or overlapping bounding boxes.
        Solves the phenomenon where a PDF generator slices a single image into 50 horizontal strips.
        """
        if not boxes:
            return []
            
        merged = []
        for box in boxes:
            placed = False
            for m in merged:
                # Math: Calculate Intersection-Over-Union bounding logic + Dilation margin
                x_overlap = not (box[0] > m[2] + merge_margin or box[2] < m[0] - merge_margin)
                y_overlap = not (box[1] > m[3] + merge_margin or box[3] < m[1] - merge_margin)
                
                if x_overlap and y_overlap:
                    # Enlarge the cluster's domain map
                    m[0] = min(m[0], box[0])
                    m[1] = min(m[1], box[1])
                    m[2] = max(m[2], box[2])
                    m[3] = max(m[3], box[3])
                    m[4] = f"{m[4]}+{box[4]}" # Tag tracker
                    placed = True
                    break
            
            if not placed:
                merged.append(list(box))
                
        # Phase 2 Agglomeration: Ensure chained overlaps are totally fused
        final_merged = []
        for box in merged:
            placed = False
            for m in final_merged:
                if not (box[0] > m[2] + merge_margin or box[2] < m[0] - merge_margin) and \
                   not (box[1] > m[3] + merge_margin or box[3] < m[1] - merge_margin):
                    m[0] = min(m[0], box[0])
                    m[1] = min(m[1], box[1])
                    m[2] = max(m[2], box[2])
                    m[3] = max(m[3], box[3])
                    placed = True
                    break
            if not placed:
                final_merged.append(list(box))
                
        return final_merged

    @staticmethod
    def _find_vector_clusters(page_obj, margin=15.0):
        """
        Calculates density clusters of pure vector graphics (lines, curves, rects).
        This autonomously finds Bar Charts, Donut Charts, and native vector drawn illustrations 
        that bypass traditional Image XObject detection.
        """
        shapes = []
        # 1. Combine all topological vector drawings
        for element in page_obj.curves + page_obj.lines + page_obj.rects:
            # Noise Floor: Ignore massive page-sized invisible borders or tiny 1x1 specks
            width = element.get('x1', 0) - element.get('x0', 0)
            height = element.get('bottom', 0) - element.get('top', 0)
            if width < 2 or height < 2: continue
            if width > page_obj.width * 0.9 and height > page_obj.height * 0.9: continue
            
            shapes.append([
                element['x0'], element['top'], element['x1'], element['bottom'], "vector"
            ])
            
        # 2. Cluster heavily dense vector zones
        clusters = ImageParser._merge_overlapping_boxes(shapes, merge_margin=margin)
        
        valid_charts = []
        for c in clusters:
            width = c[2] - c[0]
            height = c[3] - c[1]
            # 3. Only keep structurally significant clusters (Ignore random singular styling lines)
            if width > 50 and height > 50: 
                valid_charts.append(c)
                
        return valid_charts

    @staticmethod
    def extract_images(page_obj, safe_top, safe_bottom):
        """Extract images and vector diagrams optimally via Spatial Clustering mechanics."""
        images = []
        raw_boxes = []
        page_area = page_obj.width * page_obj.height
        
        # --- [1] Mine standard XObject Images ---
        for img in page_obj.images:
            x0 = img.get("x0", 0)
            img_top = img.get("top", 0)
            x1 = img.get("x1", 0)
            bottom = img.get("bottom", 0)
            
            width = x1 - x0
            height = bottom - img_top
            area = width * height
            
            # Statistical Noise Filter: Ignore unseeable tracking pixels
            if width < 5 or height < 5 or area < 25:
                continue
                
            # Statistical Washout Filter: Ignore full-page watermark backgrounds if the page has text layers
            if area > page_area * 0.85 and len(page_obj.chars) > 20:
                continue
                
            if safe_top < img_top < safe_bottom:
                raw_boxes.append([x0, img_top, x1, bottom, "xobject"])
                
        # --- [2] Mine undiscovered Vector Charts & Diagrams ---
        vector_boxes = ImageParser._find_vector_clusters(page_obj)
        raw_boxes.extend(vector_boxes)

        # --- [3] Mathematically Fuse Colliding Matrix Elements ---
        smart_bboxes = ImageParser._merge_overlapping_boxes(raw_boxes, merge_margin=2.0)
        
        # --- [4] Physical Extraction ---
        for index, bbox_data in enumerate(smart_bboxes):
            x0, top, x1, bottom, tag = bbox_data
            
            # Boundary constraint mapping
            x0 = max(0, x0)
            top = max(0, top)
            x1 = min(page_obj.width, x1)
            bottom = min(page_obj.height, bottom)
            
            if x1 <= x0 or bottom <= top:
                continue
                
            bbox = [round(x0, 2), round(top, 2), round(x1, 2), round(bottom, 2)]
            
            try:
                # Command pipeline to Ghostscript/PDFium native crop tools
                cropped_page = page_obj.crop((x0, top, x1, bottom))
                pil_image = cropped_page.to_image(resolution=150).original
                
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                images.append({
                    "type": "image",
                    "top": round(top, 2),
                    "bottom": round(bottom, 2),
                    "bbox": bbox,
                    "base64_data": base64_image,
                    "metadata": f"Image_Cluster_p{page_obj.page_number}_{index + 1} (Components: {tag}) at coords: bbox={bbox}"
                })
            except Exception as e:
                log_service(logger, f"Skipped unrenderable masking object on p{page_obj.page_number}: {e}", "warning")
                continue

        return images
