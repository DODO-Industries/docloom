# 🧠 Document Processor (`pdf_parser.py`)

## 1. Overview
The `pdf_parser.py` acts as the entire "Engine" orchestrating the data parsing. It relies heavily on statistical derivation (e.g. `avg`, `stdev`, `histograms`) instead of fixed hardcodes `(e.g., space > 10px)` to understand PDF layouts autonomously.

---

## 2. Special Methods & Block logic

### `_calculate_dynamic_metrics(chars)`
* **Purpose:** Determines exactly what the "Baseline Style" of the document is.
* **Special Logic:** It grabs every character on the page and calculates:
   1. `mode_font_size`: What font size is the most common? This represents the 'Paragraph' text.
   2. `std_dev_size`: It applies standard deviation `statistics.stdev()` formula. Instead of hardcoding `size > 14`, we test if text is `Mode + (1.5 * StdDev)`. This dynamically identifies Heading text perfectly on tiny mobile PDFs and massive poster PDFs alike.

### `_get_vertical_projection_profiles(chars, page_width)`
* **Purpose:** Mathematical column isolation.
* **Special Logic:** 
  It projects characters onto a 1D width map. Instead of relying on a rigid noisy histogram, it executes an **O(N) Kernel Density Estimation (Gaussian Box Blur Proxy)**. This smooths out messy texts seamlessly to precisely discover the True Center of white-space valley gutters, remaining lightning-quick!

### `_process_single_page(pdf_path, page_num)`
* **Special Parts Breakdown:**
  * **Frankenstein CID Fallback:** PDF Fonts occasionally corrupt, trapping text as gibberish variables like `(cid:14)`. The engine pre-analyzes alphanumeric density via Regex. If actual characters take up `<60%` of the block load, it autonomously suspends parsing, imports `pytesseract`, generates a physical render, and completely extracts the text contextually using Tesseract OCR fallback, successfully rescuing the document!

### `_reconstruct_page(chars, tables, images, page_width)`
* **Purpose:** The core layout synthesis brain.
* **Special Parts Breakdown:**
  1. **Z-Order Stack Overlap Masking:** Ignores any text completely encapsulated by bounds of a parsed picture.
  2. **DBSCAN Density Line Aggregation & Math Formulas:** Compiles chunks line-by-line horizontally. If it detects a high density of symbols (`=, +, -, ∑, √`), it enacts **Semantic Math Grouping**. It temporarily relaxes the Y-axis constraints exponentially, ensuring dense equations with massive superscript/subscript vertical variance stay bound together structurally!
  3. **Spatial Column Tagging:** It injects strict integer IDs natively mapping which column blocks belong to.
  
### `_strip_exclusion_zones(all_pages)`
* **Purpose:** Data Washout logic.
* **Special Logic:** Recursively compares floating paragraphs against the finalized document layout array. Any isolated strings acting identically across `>80%` of pages (e.g., standard "Confidential" watermarks, Legal Disclaimers, Repeating Titles) are permanently deleted to stop irrelevant token-bloat in your LLMs.

### `_finalize_content()`
* **Purpose:** Collapses all dimensions into readable JSON sequence without destroying semantics.
* **The Magic Sorcery:**
  ```python
  structured_content.sort(key=lambda x: (x.get('column_id', 0) if x.get('column_id', 0) != 0 else 999, x.get('top', 0)))
  ```
  This forces the array to sort cleanly by column group (`column 1` blocks top-to-bottom, then `column 2`), perfectly placing inline Tables and Images chronologically alongside the exact text sentence they sit physical adjacent to. Any spanning structure `0` gets pushed to `999`, dropping to the trailing end of the dataset to be processed independently.

### `stream_document_pipeline()`
* **Purpose:** Handles concurrency. 
* **Optimized Detail:** Wraps executing processes tightly in `@staticmethod` rules; `ProcessPoolExecutor` only serializes base scalar variables into its worker forks, halting any Memory / RAM buildup issues historically associated with Python class parallel processing.
