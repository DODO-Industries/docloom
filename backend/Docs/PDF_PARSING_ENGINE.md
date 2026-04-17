# 📄 DocLoom PDF Parsing Engine

Welcome to the documentation for the DocLoom **World-Class PDF Parsing Engine**. This system acts as the foundational layer to meticulously parse, separate, and analyze Document Layouts (Tables, Text Layouts, Images) before converting them into machine-readable JSON formats that preserve read-order semantics.

Rather than acting as a rigid "script" looking for strict font sizes (e.g., 14pt = header) or absolute X/Y spacings, this tool operates autonomously across any arbitrary PDF using an **"Engine" mindset** driven by Dynamic Layout Audits.

---

## 🏛️ System Architecture 

To facilitate single responsibility, maintainability, and code scaling, the PDF Parser components are separated into domain-specific drivers:

1. **`TableParser` (`table_parsing.py`)** 
   - Uses Camelot to execute the heavy lifting of identifying grid-based structures.
   - Evaluates "fake" layout tables (often created by formatting tools) versus **genuine data tables** using strict cell emptiness algorithms.
   - Outputs robust JSON mappings equipped with pixel-perfect `bbox` (bounding box) arrays to enforce spatial awareness.

2. **`ImageParser` (`image_parser.py`)**
   - Targets explicit embedded imagery objects.
   - Outputs boundary arrays (`x0, top, x1, bottom`) and associated coordinate metadata.

3. **`DocumentProcessor` (`pdf_parser.py`)**  *(The Engine)*
   - Orchestrator of the `Table` and `Image` parsers.
   - Analyzes everything else (Lines, Paragraphs, Headings, Blocks) utilizing complex data-driven audits, Z-Order rendering semantics, and spatial clustering algorithms.
   - Handles multi-threading using Python's `ProcessPoolExecutor`.

---

## ⚙️ How It Works (The Atomic Workflow)

Every task performed passes through a concurrent processing logic. Here is the lifecycle of a single page inside `_process_single_page()`:

### Phase 1: Efficient Data I/O & Memory Constraints
The most expensive operation in processing PDFs is disk I/O. The Engine opens the file exactly **once** per child process, obtaining a `page` object, and actively closes the file handle immediately to prevent descriptor leaks. Furthermore, core layout logic is strictly separated into `@staticmethod` signatures; this ensures that Python's `ProcessPoolExecutor` only pickels pure variables (like file paths) rather than duplicating the entire class Object into RAM, successfully preventing memory bloat on massive 1,000+ page PDFs.

During character extraction, the parser completely neutralizes **Ligature Blindness**:
Modern PDFs often join characters like `f` and `i` into a single, specialized symbol (`ﬁ`). If a system extracts this natively, terms like "ﬁle" fail text searches and LLM retrieval. We apply `unicodedata.normalize('NFKC', text)` on every character the moment it leaves the PDF, translating visual characters directly into their universally standardized letters.

### Phase 2: Z-Order Stack Rendering
Many academic PDFs and brochures superimpose text over graphics, or tables hide invisible text layer characters behind them.
Once Tables and Images are generated, their exact topological bounds are known. The Engine creates a **Z-Order Stack** mapping:
It evaluates the incoming raw text. If any raw text characters lie *under* the boundary area of a parsed `table` or `image`, that text is intentionally ignored and popped off the rendering stack. This strictly prevents duplicate extraction and content overlapping.

### Phase 3: The Page Visual Audit (No Hardcodes)
To understand what is an "indent," a "column," or a "heading" without human input, the engine surveys the extracted text and derives purely statistical realities.
* **Average Character Width/Height:** Evaluates the mean widths to understand what the gap of a "spacebar" should actually look like on this unique page.
* **StdDev Heading Thresholding:** It calculates the `mode` (most common) font size to figure out what typical body text looks like. Then it calculates the statistical **Standard Deviation** (`statistics.stdev`). Any text whose font size is `Mode + (1.5 * StdDev)` is definitively flagged as a `heading`, seamlessly adapting to both 8pt financial journals and 16pt children's books.

### Phase 4: Vertical Projection Profiles (Column Detection)
Rather than guessing where a column begins or ends, the system fires off a `_get_vertical_projection_profiles()` execution. 
This behaves like an overarching shadow map: it sums up all character width coverage across the X-Axis of the entire page to build a 1D Histogram array.
Where text exists, the histogram has high values. Where there is a **white-space column gutter**, the histogram falls entirely to zero. To adapt natively to dense mobile views or "pocket" PDFs, the boundary width of this gutter isn't a hardcoded pixel limit; it is evaluated dynamically as `Average Character Width * 3`. The system isolates these valid valleys and permanently bans horizontal merging operations traversing across these fault lines.

### Phase 5: DBSCAN Block Construction
To weave floating strings into contextual paragraphs, the engine scans left-to-right (`x0`), dropping character data into sequential buffers.
* If a horizontal gap crosses a **projection profile gutter** or is larger than `3.5 * Average Width`, it forces a Line Break.
* If a gap behaves normally (`>1.5 * Average Width`), it adds exactly one semantic space `(" ")`.

### Phase 6: Spatial Column Clustering & Semantic Merging
Because the system identified columns early on, text chunks are bucketed directly into spatial Columns (grouped by their average X-start position).
Every text block is tagged with an explicit integer `column_id` natively dictating spatial sequence (Column 1, Column 2, etc.). The array is sorted from absolute-left to absolute-right, and child-blocks are sorted from top to bottom. This guarantees the reconstruction of the perfect **human reading order**.

As the final layer of polish, consecutive floating Blocks sharing the exact same font size, column boundary, and minimal vertical gap are snapped dynamically together into full-sentence `paragraph` objects.

### Phase 7: Document Materialization & Logical Layout Sorting
The Engine assesses the identified `Tables` and `Images` and tests their geometries against the text column structure:
* **Spanning Layouts:** If an image or table spans significantly wider than a localized column distance, it is assigned `column_id: 0`. It floats independently and is evaluated after text flow.
* **Interleaved Layouts:** If the table/image physically locks into the spatial boundaries of a recognized text column (X-axis matching), it dynamically absorbs that text column's `column_id`. 

Finally, the master array executes a highly precise multi-key structural sort:
```python
lambda x: (column_id, top)
```
This iterates sequentially through Column 1 top-to-bottom (perfectly interlacing Tables logically between any paragraphs they separate physically), then Column 2, and pushes broad `0` objects to the trailing end!
All complex statistical metadata arrays (bounding boxes, font info, rendering coords) are purged natively, leaving clean JSON. The vital `column_id` persists on every object, enabling downstream LLMs to map layout hierarchies precisely natively.

*(The result represents the final machine-aware JSON sent linearly to frontend renders or your chunking/embedding pipeline.)*
