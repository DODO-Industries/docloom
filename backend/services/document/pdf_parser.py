import math
import pdfplumber
import camelot
import json
import os
import time
import sys
import gc
import datetime
import re
import unicodedata
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from backend.services.document.table_parsing import TableParser
from backend.services.image.image_parser import ImageParser
from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("DocumentProcessor")

class DocumentProcessor:
    def __init__(self):
        pass

    @staticmethod
    def _calculate_dynamic_metrics(chars):
        """
        Calculates dynamic font thresholds using Standard Deviation
        for flawless Heading vs Paragraph splits.
        """
        if not chars:
            return 10.0, 10.0, 0.0

        sizes = [round(c['size'], 1) for c in chars]
        mode_font_size = Counter(sizes).most_common(1)[0][0]
        
        if len(sizes) > 1:
            try:
                std_dev_size = statistics.stdev(sizes)
            except Exception:
                std_dev_size = 0.0
        else:
            std_dev_size = 0.0

        # For rotated text, height might be normalized width or real height
        # Use the larger dimension as a proxy for 'line verticality' if needed
        heights = [abs(c['bottom'] - c['top']) for c in chars]
        avg_char_height = sum(heights) / len(heights) if heights else 10.0

        return mode_font_size, avg_char_height, std_dev_size

    @staticmethod
    def _get_rotation_degree(matrix):
        """
        Extracts exact floating-point degrees from PDF character metadata.
        Uses Geometric Matrix Calculus (atan2) for enterprise-grade precision.
        """
        if not matrix or len(matrix) < 2:
            return 0.0
        # angle = atan2(b, a) * (180/PI)
        angle = math.degrees(math.atan2(matrix[1], matrix[0]))
        return round(angle % 360, 1)

    @staticmethod
    def _normalize_coordinate_space(char, angle):
        """
        Dynamic coordinate re-mapping based on detected angle.
        Swaps X/Y logic for 90/270 and applies rotation transforms for diagonal text.
        """
        if angle == 0.0:
            return {**char, "nx0": char["x0"], "nx1": char["x1"], "ntop": char["top"], "nbottom": char["bottom"], "angle": 0.0}
            
        # Un-rotate coordinates to a canonical horizontal space
        rad = math.radians(-angle) 
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        # Center points for rotation
        cx = (char["x0"] + char["x1"]) / 2
        cy = (char["top"] + char["bottom"]) / 2
        
        # Mapping: Rotate point (cx, cy) back to 0-degree space
        nx = cx * cos_a - cy * sin_a
        ny = cx * sin_a + cy * cos_a
        
        # Reconstruct normalized bounding box
        half_w = char["width"] / 2
        half_h = abs(char["bottom"] - char["top"]) / 2
        
        return {
            **char,
            "nx0": nx - half_w,
            "nx1": nx + half_w,
            "ntop": ny - half_h,
            "nbottom": ny + half_h,
            "angle": angle
        }

    @staticmethod
    def _get_vertical_projection_profiles(chars, page_width):
        """
        Scans X-axis for white space valleys using a sharper smoothing window.
        Returns tuples of (start_x, end_x) for detected gutters.
        """
        if not chars:
            return []
            
        avg_char_width = sum(c['width'] for c in chars) / len(chars) if chars else 5.0
        # Narrower threshold for academic papers
        gutter_thresh = avg_char_width * 2.0 

        hist = [0] * int(page_width + 1)
        for c in chars:
            x0 = int(c['x0'])
            x1 = int(c['x1'])
            for x in range(max(0, x0), min(int(page_width), x1 + 1)):
                hist[x] += 1
                
        # Sharper window for gutter precision
        window = 2
        smoothed = [0.0] * len(hist)
        for i in range(len(hist)):
            start = max(0, i - window)
            end = min(len(hist), i + window + 1)
            smoothed[i] = sum(hist[start:end]) / (end - start)
                
        gutters = []
        in_gap = False
        gap_start = 0
        for i, val in enumerate(smoothed):
            if val < 0.1: # Sharper valley detection
                if not in_gap:
                    in_gap = True
                    gap_start = i
            else:
                if in_gap:
                    gap_width = i - gap_start
                    if gap_width > gutter_thresh:
                        gutters.append((gap_start, i))
                    in_gap = False
        return gutters

    @staticmethod
    def _process_single_page(pdf_path, page_num):
        page_data = {"page_number": page_num, "content": []}
        chars = []
        
        # Optimize I/O: Open the PDF stream exactly once for text and images
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num - 1]
            page_height = page.height
            page_width = page.width
            
            safe_top = page_height * 0.05
            safe_bottom = page_height * 0.95
            
            for char in page.chars:
                char_top = char.get("top", 0)
                if safe_top < char_top < safe_bottom:
                    # Ligature resolution + Memory Optimization using basic dict slots
                    norm_text = unicodedata.normalize('NFKC', char.get("text", ""))
                    
                    # Enterprise Coordinate Detection: Extract Rotation Matrix
                    matrix = char.get("matrix", (1, 0, 0, 1, 0, 0))
                    angle = DocumentProcessor._get_rotation_degree(matrix)
                    
                    chars.append({
                        "text": norm_text,
                        "fontname": char.get("fontname", ""),
                        "size": char.get("size", 10.0),
                        "x0": round(char.get("x0", 0), 2),
                        "x1": round(char.get("x1", 0), 2),
                        "top": round(char_top, 2),
                        "bottom": round(char.get("bottom", 0), 2),
                        "width": round(char.get("x1", 0) - char.get("x0", 0), 2),
                        "matrix": matrix,
                        "angle": angle
                    })
            
            # Frankenstein CID Font Trap (OCR Fallback)
            full_text = "".join([c["text"] for c in chars])
            if "(cid:" in full_text and len(full_text) > 10:
                clean_len = len(re.sub(r'\(cid:\d+\)', '', full_text))
                if (clean_len / len(full_text)) < 0.6:
                    log_service(logger, f"Fragmented CID Font detected on Page {page_num}. Engaging OCR Engine...", "warning")
                    try:
                        os.environ['OMP_THREAD_LIMIT'] = '1'
                        import pytesseract
                        pil_img = page.to_image(resolution=150).original
                        ocr_text = pytesseract.image_to_string(pil_img)
                        page_data["content"] = [{"type": "paragraph", "text": ocr_text.strip(), "column_id": 1}]
                        return page_data
                    except ImportError:
                        log_service(logger, "pytesseract not installed. Reverting to garbled CID text.", "error")
                        
            images = ImageParser.extract_images(page, safe_top, safe_bottom)
            gutters = DocumentProcessor._get_vertical_projection_profiles(chars, page_width)
            tables = TableParser.extract_tables(pdf_path, page_num, page, page_height, gutters)
            
        page_data["content"] = DocumentProcessor._reconstruct_page(chars, tables, images, page_width, gutters)
        gc.collect()
        return page_data

    @staticmethod
    def _reconstruct_page(chars, tables, images, page_width, gutters):
        """
        Implements Recursive XY-Cut with Rotation Normalization.
        Groups characters into 'Angle Buckets' before line formation.
        """
        structured_content = []
        text_chars = []
        
        # 1. Z-Order Stack Evaluation
        for char in chars:
            overlapped = False
            for t in tables:
                bbox = t.get('bbox', [0,0,0,0]) 
                if (bbox[0] <= char['x0'] <= bbox[2] and bbox[1] <= char['top'] and char['bottom'] <= bbox[3]):
                    overlapped = True
                    break
            if not overlapped:
                for img in images:
                    bbox = img.get('bbox', [0,0,0,0])
                    if (bbox[0] <= char['x0'] <= bbox[2] and bbox[1] <= char['top'] and char['bottom'] <= bbox[3]):
                        overlapped = True
                        break
            if not overlapped:
                text_chars.append(char)

        if not text_chars:
            return DocumentProcessor._finalize_content(structured_content, tables, images)

        # 2. Angle Bucketing & Vector-Based Grouping
        angle_buckets = {}
        for char in text_chars:
            angle = char.get("angle", 0.0)
            if angle not in angle_buckets:
                angle_buckets[angle] = []
            angle_buckets[angle].append(DocumentProcessor._normalize_coordinate_space(char, angle))

        for angle, bucket_chars in angle_buckets.items():
            # Process each rotation group separately
            mode_size, avg_height, std_dev_size = DocumentProcessor._calculate_dynamic_metrics(bucket_chars)
            heading_thresh = mode_size + (1.3 * std_dev_size) if std_dev_size > 0.5 else mode_size * 1.15
            
            # Sort bucket chars by Normalized coordinates (ntop, nx0)
            bucket_chars.sort(key=lambda c: (c['ntop'], c['nx0']))
            
            # 3. Line Construction (Normalized Space)
            lines = []
            if bucket_chars:
                curr_line = [bucket_chars[0]]
                line_glue = avg_height * 0.3
                for i in range(1, len(bucket_chars)):
                    if abs(bucket_chars[i]['ntop'] - curr_line[-1]['ntop']) < line_glue:
                        curr_line.append(bucket_chars[i])
                    else:
                        lines.append(curr_line)
                        curr_line = [bucket_chars[i]]
                lines.append(curr_line)

            # 4. Horizontal Slicing (Normalized Space)
            temp_slices = []
            if lines:
                curr_slice = [lines[0]]
                slice_gap_thresh = avg_height * 1.3
                for i in range(1, len(lines)):
                    # Check gap between nbottom of last line and ntop of curr line
                    last_line_bottom = max(c['nbottom'] for c in curr_slice[-1])
                    curr_line_top = min(c['ntop'] for c in lines[i])
                    
                    if (curr_line_top - last_line_bottom) < slice_gap_thresh:
                        curr_slice.append(lines[i])
                    else:
                        temp_slices.append(curr_slice)
                        curr_slice = [lines[i]]
                temp_slices.append(curr_slice)

            # 5. Recursive Column Recovery (Context-Aware)
            for slice_lines in temp_slices:
                slice_chars = [c for line in slice_lines for c in line]
                
                slice_min_x = min(c['nx0'] for c in slice_chars)
                slice_max_x = max(c['nx1'] for c in slice_chars)
                slice_width = slice_max_x - slice_min_x
                
                relevant_gutters = [g for g in gutters if slice_min_x < g[0] and g[1] < slice_max_x]
                
                # Filter gutters: Only those in the central 50% of the slice
                center_x = slice_min_x + (slice_width / 2)
                central_gutters = [g for g in relevant_gutters if g[0] < center_x < g[1]]
                
                if central_gutters:
                    gutter = central_gutters[0]
                    left_chars = [c for c in slice_chars if c['nx1'] <= gutter[0]]
                    right_chars = [c for c in slice_chars if c['nx0'] >= gutter[1]]
                    
                    if left_chars: structured_content.extend(DocumentProcessor._extract_blocks_from_chars(left_chars, heading_thresh, 1, angle))
                    if right_chars: structured_content.extend(DocumentProcessor._extract_blocks_from_chars(right_chars, heading_thresh, 2, angle))
                else:
                    structured_content.extend(DocumentProcessor._extract_blocks_from_chars(slice_chars, heading_thresh, 0, angle))

        # 7. Spatial Injection for Spanning Objects
        for t in tables:
            t['column_id'] = 0 # Default spanning
            bbox = t.get('bbox', [0,0,0,0])
            if (bbox[2] - bbox[0]) < page_width * 0.6:
                t['column_id'] = 1 if bbox[0] < page_width/2 else 2
                
        for img in images:
            img['column_id'] = 0
            bbox = img.get('bbox', [0,0,0,0])
            if (bbox[2] - bbox[0]) < page_width * 0.6:
                img['column_id'] = 1 if bbox[0] < page_width/2 else 2

        return DocumentProcessor._finalize_content(structured_content, tables, images)

    @staticmethod
    def _extract_blocks_from_chars(chars, heading_thresh, column_id, rotation_angle=0.0):
        """
        Reconstructs semantic text blocks from characters using normalized font-aware grouping.
        """
        if not chars:
            return []
            
        # Optimization: Sort by normalized coordinates for semantic flow
        chars.sort(key=lambda c: (c['ntop'], c['nx0']))
        
        blocks = []
        curr_block_chars = []
        
        def finalize_block(b_chars):
            if not b_chars: return
            text = ""
            # Screen space bbox (original)
            bbox = [b_chars[0]['x0'], b_chars[0]['top'], b_chars[0]['x1'], b_chars[0]['bottom']]
            # Normalized space bbox for intra-bucket grouping
            n_bbox = [b_chars[0]['nx0'], b_chars[0]['ntop'], b_chars[0]['nx1'], b_chars[0]['nbottom']]
            
            size_sum = 0
            for i, c in enumerate(b_chars):
                char_text = c.get('text', '')
                space = ""
                if i > 0:
                    prev_c = b_chars[i-1]
                    # Same line space recovery (in normalized space)
                    if abs(c['ntop'] - prev_c['ntop']) < (c['size'] * 0.4):
                        gap = c['nx0'] - prev_c['nx1']
                        if gap > (c['size'] * 0.16):
                            space = " "
                text += space + char_text
                size_sum += c['size']
                
                # Expand original screen-space bbox
                bbox[0] = min(bbox[0], c['x0'])
                bbox[1] = min(bbox[1], c['top'])
                bbox[2] = max(bbox[2], c['x1'])
                bbox[3] = max(bbox[3], c['bottom'])
                
                # Expand normalized bbox
                n_bbox[1] = min(n_bbox[1], c['ntop'])
                n_bbox[3] = max(n_bbox[3], c['nbottom'])
            
            avg_size = size_sum / len(b_chars)
            blocks.append({
                "text": text.strip(),
                "type": "heading" if avg_size >= heading_thresh else "paragraph",
                "column_id": column_id,
                "font_size": avg_size,
                "bbox": bbox,
                "top": bbox[1], "bottom": bbox[3], # For sorting/merging 
                "ntop": n_bbox[1], "nbottom": n_bbox[3], # For intra-bucket post-merge
                "rotation": rotation_angle
            })

        for i, char in enumerate(chars):
            if i == 0:
                curr_block_chars.append(char)
                continue
            
            prev_char = chars[i-1]
            # Split based on normalized vertical gap
            v_gap = char['ntop'] - prev_char['nbottom']
            size_diff = abs(char['size'] - prev_char['size'])
            
            if v_gap > (prev_char['size'] * 0.5) or size_diff > 1.5:
                finalize_block(curr_block_chars)
                curr_block_chars = [char]
            else:
                curr_block_chars.append(char)
        
        finalize_block(curr_block_chars)
        
        # Semantic post-merging within the same bucket
        merged_blocks = []
        if blocks:
            curr = blocks[0]
            for next_b in blocks[1:]:
                # Merge paragraphs that are close together in canonical space
                if (curr['column_id'] == next_b['column_id'] and 
                    curr['type'] == next_b['type'] and
                    abs(next_b['ntop'] - curr['nbottom']) < (curr['font_size'] * 0.8)):
                    
                    if curr['text'].endswith('-'):
                        curr['text'] = curr['text'][:-1] + next_b['text']
                    else:
                        curr['text'] += " " + next_b['text']
                    
                    # Merge screen-space bboxes
                    curr['bbox'][0] = min(curr['bbox'][0], next_b['bbox'][0])
                    curr['bbox'][1] = min(curr['bbox'][1], next_b['bbox'][1])
                    curr['bbox'][2] = max(curr['bbox'][2], next_b['bbox'][2])
                    curr['bbox'][3] = max(curr['bbox'][3], next_b['bbox'][3])
                    curr['bottom'] = curr['bbox'][3]
                    curr['nbottom'] = next_b['nbottom']
                else:
                    merged_blocks.append(curr)
                    curr = next_b
            merged_blocks.append(curr)
            
        for b in merged_blocks:
            b.pop('ntop', None)
            b.pop('nbottom', None)
            
        return merged_blocks

    @staticmethod
    def _finalize_content(structured_content, tables, images):
        for t in tables:
            structured_content.append({"type": "table", "data": t["data"], "top": t["top"], "column_id": t.get("column_id", 0)})
        for img in images:
            structured_content.append({
                "type": "image", 
                "metadata": img["metadata"], 
                "base64_data": img.get("base64_data"),
                "top": img["top"], 
                "column_id": img.get("column_id", 0)
            })

        # Correct Human Reading Flow: Read columns left-to-right (1, 2, 3), 
        # then spanning items (tables/images) with column_id=0 treated as appending at the end.
        structured_content.sort(key=lambda x: (x.get('column_id', 0) if x.get('column_id', 0) != 0 else 999, x.get('top', 0)))
        
        # Add 'bbox' array for semantic linking
        for item in structured_content:
            if 'bbox' not in item:
                if 'x0' in item and 'top' in item and 'x1' in item and 'bottom' in item:
                    item['bbox'] = [round(item['x0'], 2), round(item['top'], 2), round(item['x1'], 2), round(item['bottom'], 2)]

        # Strip internal metrics logic 
        for item in structured_content:
            item.pop('top', None)
            item.pop('bottom', None)
            item.pop('is_bold', None)
            item.pop('x0', None)
            item.pop('x1', None)
            item.pop('coordinates', None)

        return structured_content

    @staticmethod
    def _strip_exclusion_zones(all_pages):
        """Pre-computes and strips repeating Header/Footer watermarks globally."""
        total_pages = len(all_pages)
        if total_pages < 3: return all_pages
        
        freq_map = {}
        for page in all_pages:
            for item in page.get("content", []):
                if item.get("type") in ["paragraph", "heading"]:
                    raw_text = item.get("text", "").strip()
                    # Strip dynamic numbers so "Page 1" and "Page 2" hash to "Page "
                    normalized_text = re.sub(r'\d+', '', raw_text) 
                    if 0 < len(normalized_text) < 100: # Usually headers/footers aren't massive paragraphs
                        freq_map[normalized_text] = freq_map.get(normalized_text, 0) + 1
                        
        # If it appears on > 80% pages identically, it's a structural watermark
        exclusion_zones = {text for text, count in freq_map.items() if count >= (total_pages * 0.8)}
        
        for page in all_pages:
            page["content"] = [item for item in page["content"] if re.sub(r'\d+', '', item.get("text", "").strip()) not in exclusion_zones]
            
        return all_pages
    
    @staticmethod
    def stream_document_pipeline(pdf_path, max_workers=4):
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

        # Non-blocking async execution over OS-level process streams
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {
                executor.submit(DocumentProcessor._process_single_page, pdf_path, page_num): page_num 
                for page_num in range(1, total_pages + 1)
            }
            for future in as_completed(future_to_page):
                yield future.result()

def main():
    pdf_path = "assets/1706.03762v7.pdf"
    processor = DocumentProcessor()
    
    start_time = time.time()
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
    
    log_service(logger, f"Starting Engine with Dynamic Metrics and NFKC on {total_pages} pages...", "info")
    all_pages = []
    
    completed_count = 0
    for page_json in processor.stream_document_pipeline(pdf_path, max_workers=4):
        completed_count += 1
        log_service(logger, f"[{completed_count}/{total_pages}] Extracted Page {page_json['page_number']}", "debug")
        all_pages.append(page_json)

    all_pages.sort(key=lambda x: x["page_number"])
    all_pages = TableParser.heal_cross_page_tables(all_pages)
    all_pages = DocumentProcessor._strip_exclusion_zones(all_pages)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("backend", "results", "pdf_parser_test")
    os.makedirs(results_dir, exist_ok=True)
    
    output_filename = os.path.join(results_dir, f"result_{timestamp}.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, indent=4, ensure_ascii=False)

    duration = time.time() - start_time
    log_service(logger, f"Success! Engine extracted {len(all_pages)} pages in {duration:.2f} seconds.", "info")
    log_service(logger, f"Results saved to: {output_filename}", "info")


if __name__ == '__main__':
    main()