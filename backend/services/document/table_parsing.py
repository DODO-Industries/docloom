import camelot

class TableParser:
    @staticmethod
    def is_valid_data_table(table_data):
        """Rejects layout formatting that Camelot mistakes for tables."""
        if not table_data or len(table_data) == 0:
            return False
            
        num_rows = len(table_data)
        num_cols = len(table_data[0])
        total_cells = num_rows * num_cols
        
        if total_cells <= 1 or num_cols <= 1:
            return False

        empty_cells = 0
        total_word_count = 0
        max_words_in_cell = 0
        non_empty_cells = 0

        for row in table_data:
            for cell in row:
                cell_str = str(cell).strip()
                if not cell_str:
                    empty_cells += 1
                else:
                    words = cell_str.split()
                    word_count = len(words)
                    total_word_count += word_count
                    max_words_in_cell = max(max_words_in_cell, word_count)
                    non_empty_cells += 1

        if non_empty_cells < 2: return False
        
        empty_ratio = empty_cells / total_cells
        if empty_ratio > 0.8: 
            return False

        if num_cols == 2:
            col1_words = sum(len(str(row[0]).split()) for row in table_data)
            col2_words = sum(len(str(row[1]).split()) for row in table_data)
            avg_col1 = col1_words / num_rows if num_rows else 0
            avg_col2 = col2_words / num_rows if num_rows else 0
            
            if avg_col1 < 2 and max_words_in_cell > 20:
                return False 
                
            # Reject word-alignment networks (e.g. Attention visualizations mapping word to word)
            if avg_col1 <= 1.2 and avg_col2 <= 1.2 and num_rows > 5:
                return False

        return True

    @staticmethod
    def extract_tables(pdf_path, page_num, page_obj, page_height, gutters=None):
        if gutters is None: gutters = []
        tables = []
        # Check if pdfplumber detects any table structure (lines/intersections)
        likely_tables = page_obj.find_tables()
        if not likely_tables:
            return tables

        try:
            cam_tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='lattice', suppress_stdout=True)
            if not cam_tables or len(cam_tables) == 0:
                col_x = [f"{g[0] + (g[1]-g[0])/2:.1f}" for g in gutters]
                columns_arg = ",".join(col_x) if col_x else None
                
                if columns_arg:
                    cam_tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='stream', split_text=True, columns=[columns_arg], suppress_stdout=True)
                else:
                    cam_tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='stream', split_text=True, suppress_stdout=True)
            
            for t in cam_tables:
                raw_data = t.df.values.tolist()
                if TableParser.is_valid_data_table(raw_data):
                    # Translate Camelot's bottom-up coords to pdfplumber top-down coords
                    # Camelot: (x0, y0_bottom, x1, y1_top)
                    top_coord = page_height - t._bbox[3]
                    bottom_coord = page_height - t._bbox[1]
                    
                    bbox_top_down = [t._bbox[0], top_coord, t._bbox[2], bottom_coord]
                    
                    tables.append({
                        "type": "table",
                        "bbox": bbox_top_down, 
                        "top": top_coord,
                        "bottom": bottom_coord,
                        "data": raw_data
                    })
        except Exception:
            pass
            
        return tables

    @staticmethod
    def heal_cross_page_tables(all_pages):
        """Merges floating tables that split seamlessly across physical pages."""
        for i in range(len(all_pages) - 1):
            curr_content = all_pages[i].get("content", [])
            next_content = all_pages[i+1].get("content", [])
            
            if not curr_content or not next_content: continue
            
            last_item = curr_content[-1]
            first_item = next_content[0]
            
            if last_item.get("type") == "table" and first_item.get("type") == "table":
                curr_cols = last_item.get("data", [[]])[0] if last_item.get("data") else []
                next_cols = first_item.get("data", [[]])[0] if first_item.get("data") else []
                
                # Semantic Bridging: Same num columns identically merges
                if len(curr_cols) == len(next_cols) and len(curr_cols) > 0:
                    if last_item["data"][0] == first_item["data"][0]:
                        last_item["data"].extend(first_item["data"][1:]) # Skip the repeated header
                    else:
                        last_item["data"].extend(first_item.get("data", []))
                    next_content.pop(0) # Remove orphaned fragment
                    
        return all_pages
