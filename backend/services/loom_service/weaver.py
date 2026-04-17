import msgpack
import statistics
import uuid
import re
from collections import Counter
from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("LoomWeaver")

class LoomWeaver:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.doc_root_id = self._add_node("Document Root", "root")

    def _generate_id(self, prefix="node"):
        return f"{prefix}_{uuid.uuid4().hex[:6]}"

    def _compress_bbox(self, bbox):
        """Compress float BBox into integer array to save space."""
        if not bbox or not isinstance(bbox, list) or len(bbox) < 4:
            return None
        return [int(round(x)) for x in bbox]

    def weave(self, all_pages, output_path):
        """
        Transforms flat extraction data into a DAG with parent-child 
        and semantic relationships, serialized to binary .loom format.
        """
        log_service(logger, "Weaving binary graph structure...", "info")
        
        # 1. Statistical Thresholding
        all_items = [item for page in all_pages for item in page.get("content", [])]
        font_sizes = [item.get("font_size") for item in all_items if item.get("font_size")]
        
        if not font_sizes:
            threshold = 14
        else:
            mode_size = Counter(font_sizes).most_common(1)[0][0]
            std_dev = statistics.stdev(font_sizes) if len(font_sizes) > 1 else 0
            threshold = mode_size + (1.5 * std_dev)
            log_service(logger, f"Dynamic threshold sync: {threshold:.1f}pt", "debug")

        current_hierarchy_stack = [self.doc_root_id]
        
        # Keep track of paragraphs for spatial anchoring
        paragraph_nodes = []

        for page in all_pages:
            page_num = page.get("page_number")
            page_id = self._add_node(f"Page {page_num}", "page", {"num": page_num})
            self.connect(self.doc_root_id, page_id, "contains")
            current_hierarchy_stack = [self.doc_root_id, page_id]

            for item in page.get("content", []):
                item_type = item.get("type")
                text = item.get("text", "")
                raw_bbox = item.get("bbox")
                compressed_bbox = self._compress_bbox(raw_bbox)
                font_size = item.get("font_size", 0)
                
                # --- ADVANCED HIERARCHY ENGINE ---
                is_heading = (item_type == "heading" or font_size >= threshold)
                node_type = "heading" if is_heading else item_type
                
                meta = {
                    "bbox": compressed_bbox,
                    "col": item.get("column_id"),
                    "font_size": font_size
                }
                
                if "base64_data" in item: meta["binary"] = item["base64_data"]
                if "data" in item: meta["table"] = item["data"]
                
                node_id = self._add_node(text if text else f"[{item_type}]", node_type, meta)
                
                if is_heading:
                    # Pop from stack until we find a parent with a larger font size
                    # current_hierarchy_stack[0] is root, [1] is page
                    while len(current_hierarchy_stack) > 2:
                        parent_id = current_hierarchy_stack[-1]
                        parent_node = self.nodes[parent_id]
                        if parent_node["t"] == "heading":
                            # Use a small epsilon for float comparison
                            if font_size >= (parent_node["m"].get("font_size", 0) - 0.5):
                                current_hierarchy_stack.pop()
                            else:
                                break
                        else:
                            current_hierarchy_stack.pop()
                    
                    self.connect(current_hierarchy_stack[-1], node_id, "contains")
                    current_hierarchy_stack.append(node_id)
                else:
                    # Non-headings (paragraphs, tables, images) belong to current active heading
                    self.connect(current_hierarchy_stack[-1], node_id, "body_text" if node_type == "paragraph" else "contains")

                # --- HEALING & ANCHORING ---
                if node_type == "paragraph":
                    # Hyphenation Healer: merge split words like atten-\ntion
                    cleaned_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
                    if cleaned_text != text:
                        self.nodes[node_id]["c"] = cleaned_text
                    
                    paragraph_nodes.append({"id": node_id, "text": cleaned_text, "bbox": compressed_bbox, "page": page_num})

                # --- SPATIAL & KEYWORD ANCHORS ---
                if node_type in ["image", "table"] and compressed_bbox:
                    # 1. Coordinate Anchor
                    self._apply_spatial_anchor(node_id, compressed_bbox, page_num, paragraph_nodes)
                    
                    # 2. Keyword Anchor: Search for "Fig" or "Table" + Number
                    # Attempt to extract number from metadata or index if available
                    # For now, let's look for any figure/table refs in nearby paragraphs
                    self._apply_keyword_anchor(node_id, node_type, item, paragraph_nodes)

        # Final Bundle
        bundle = {
            "v": "1.1",
            "g": {"n": self.nodes, "e": self.edges}
        }
        
        with open(output_path, "wb") as f:
            f.write(msgpack.packb(bundle, use_bin_type=True))
            
        log_service(logger, f"DocLoom Wave complete. Created {output_path}", "info")

    def _apply_spatial_anchor(self, node_id, node_bbox, page_num, paragraph_nodes, margin=60):
        """Links images/tables to nearby paragraphs on the same page."""
        for p in paragraph_nodes:
            if p["page"] != page_num: continue
            p_bbox = p["bbox"]
            if not p_bbox: continue
            
            # Distance logic: gap between bottom of node and top of p, or vice versa
            gap_below = p_bbox[1] - node_bbox[3]
            gap_above = node_bbox[1] - p_bbox[3]
            
            if (0 <= gap_below <= margin) or (0 <= gap_above <= margin):
                # We check horizontal overlap too for sanity
                h_overlap = max(node_bbox[0], p_bbox[0]) < min(node_bbox[2], p_bbox[2])
                if h_overlap:
                    self.connect(node_id, p["id"], "visual_evidence")

    def _apply_keyword_anchor(self, node_id, node_type, item, paragraph_nodes):
        """Links images/tables to paragraphs mentioned as 'Figure X' or 'Table X'."""
        text_id = item.get("metadata", "")
        # Extract number from "Image_Cluster_p1_3" -> "3"
        nums = re.findall(r'_(\d+)$', text_id)
        if not nums: return
        
        target_num = nums[0]
        keyword = "Fig" if node_type == "image" else "Table"
        pattern = f"{keyword}.*?{target_num}"
        
        for p in paragraph_nodes:
            if re.search(pattern, p["text"], re.IGNORECASE):
                self.connect(node_id, p["id"], "keyword_anchor")

    def _add_node(self, content, node_type, metadata=None):
        node_id = self._generate_id(node_type)
        self.nodes[node_id] = {
            "t": node_type,
            "c": content[:5000], # Buffer limit
            "m": metadata or {}
        }
        return node_id

    def connect(self, source_id, target_id, rel_type):
        self.edges.append({"f": source_id, "t": target_id, "r": rel_type})
