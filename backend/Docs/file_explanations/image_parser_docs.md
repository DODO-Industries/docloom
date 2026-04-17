# 🖼️ Image Parser (`image_parser.py`)

## 1. Overview
The `ImageParser` is a highly robust mathematical clustering engine. Instead of merely pulling XObjects (embedded JPEGs) that the PDF officially flags, it utilizes Intersection-over-Union (IOU) bounds and Spatial Density to actively discover, heal, and physically extract complex vector charts, fractured grids, and raw images while discarding hidden noise. 

---

## 2. Block-by-Block Explanation

### `_merge_overlapping_boxes(boxes, merge_margin=5.0)`
* **Purpose:** The Spatial Healing Engine. Often, PDF generators slice a single image into 50 disjointed horizontal strips to save memory. Left unchecked, the engine would extract 50 broken slivers.
* **The Logic:**
  1. **Intersection-Over-Union (IOU):** It tests if the boundary matrix of one box intersects with another, including a physical dilation margin (`merge_margin`) to account for micro-gaps.
  2. **Agglomeration:** If two boundaries touch, they are instantly destroyed and replaced by a single `min/max` maximized perimeter spanning both elements.
  3. **Multi-Pass Healing:** It loops twice to ensure tightly packed "chains" of slices correctly inherit a single massive parent border globally.

### `_find_vector_clusters(page_obj, margin=15.0)`
* **Purpose:** The Chart & Graph Discoverer. Many PDFs draw graphs and illustrations purely using PDF-native coordinate geometry (lines, curves, rectangles), effectively rendering them completely invisible to standard image extraction.
* **The Logic:**
  1. **Vector Harvesting:** It plucks all `paths`, `lines`, and `curves` drawn natively onto the page.
  2. **Noise Gating:** Simple styling borders or 1px separator lines are ignored mathematically ($width > 2$).
  3. **DBSCAN Density Clustering:** It funnels all raw vector paths into the `_merge_overlapping_boxes` healer. When identical paths cluster tightly, their bounds rapidly merge into one large solid shape.
  4. **Diagram Authentication:** If a merged node expands larger than `50x50px`, the system statistically confirms it's a graphical diagram (unlike standard design lines) and injects it into the rendering pipeline!

### `extract_images(page_obj, safe_top, safe_bottom)`
* **Purpose:** The Orchestrator and Physical Renderer.
* **The Logic:**
  1. **Mine Standard Images with Math Noise Filters:** Pulls natively declared XObjects, but implements intense statistical checks:
     - **Phantom Pixel Deletion:** If image tracking dots compute an $Area < 25$, they are ignored.
     - **Washout Mitigation (Watermarks):** If a background graphic stretches across $>85\%$ of the physical page space, yet actual text `len(chars) > 20` exists, the engine deduces it is merely decorative wallpaper and deletes it to avoid distracting LLMs.
  2. **Physical Rendering & Extraction Strategy:**
     Because we now possess the geometric bounding boxes of all native images *and* newly discovered Chart Vectors, we can execute the **Pro Move**.
     Using Ghostscript/`pdfplumber`, It calls `page_obj.crop()` perfectly around the mathematical coordinates. Then it renders that bounding box out via `.to_image(resolution=150)`. 
  3. **RAM-Safe Encoding:** Instead of clogging disk space by saving `.png` files during batch extractions, the graphic is fed instantly into a `BytesIO()` buffer and encoded as a flawless `utf-8` **Base64 string**, ready to be embedded universally directly within DocLoom's output JSON hierarchy.
