# 📊 Table Parser (`table_parsing.py`)

## 1. Overview
The `TableParser` class is a specialized utility that bridges `pdfplumber` (which handles text and lines) with `camelot` (a dedicated table extraction library). It solves a critical problem in Data Extraction: distinguishing beautiful document layout tables (e.g., side-by-side text columns) from **genuine numeric or text matrices**.

---

## 2. Block-by-Block Explanation

### `is_valid_data_table(table_data)`
* **Purpose:** Camelot often generates "fake" tables if a PDF uses invisible boundary boxes to format paragraphs horizontally. This method filters out noise.
* **The Logic:**
  1. **Cell Volume Validation:** `if total_cells <= 1 or num_cols <= 1: return False`. Any matrix that doesn't have multiple dimensions is discarded.
  2. **Emptiness Check:** Calculates an `empty_ratio` across the table's cells. `if empty_ratio > 0.8: return False`. Layout grids usually have massive empty spaces; true data matrices are dense.
  3. **Bi-Column Layout Check:** If `num_cols == 2`, it explicitly checks if the first column contains very few words but the second contains huge paragraphs (`max_words_in_cell > 20`). This pattern proves it's just a layout column, not a data table, so it blocks the extraction.

### `extract_tables(pdf_path, page_num, page_obj, page_height)`
* **Purpose:** Locates tables and pulls them into the system while aligning their coordinate mapping planes.
* **The Code Flow:**
  - **Early Exit:** `likely_tables = page_obj.find_tables()`. Running Camelot is CPU-intensive. First, it uses `pdfplumber` to instantly verify if physical table boundary lines exist on the page. If `pdfplumber` sees no intersections, the entire function halts and returns `[]`. 
  - **Geometry Alignment (THE PRO MOVE):** 
    PDFs natively use a mathematical bottom-up coordinate space `(0,0 is at the bottom left)`. Camelot returns raw PDF bounding boxes.
    However, our engine uses a standard visual top-down coordinate space `(0 is at the top of the page screen)`.
    ```python
    # Translates Camelot's bottom-up coords to pdfplumber's top-down coords
    top_coord = page_height - t._bbox[3]
    bottom_coord = page_height - t._bbox[1]
    ```
    This snippet reverses the Y-axis so `top_coord` correctly identifies distance from the top, allowing pixel-perfect z-order occlusion when passed back to the `DocumentProcessor`.

### `heal_cross_page_tables(all_pages)`
* **Purpose:** Fixes Table Fragmentation.
* **Special Logic:** If a monolithic financial ledger was computationally chopped sequentially into Page 1, Page 2, and Page 3, this routine acts as the mathematical bridge. It iterates across the global layout. If consecutive elements (the last element of Page X and the first element of Page Y) evaluate as tables and share the exact same structural column dimensions, they are surgically merged. This emits a single, unbroken JSON Table Array into the LLM flow, completely hiding the physical page break!
