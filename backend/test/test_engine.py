import os
import time
import json
import datetime
import glob

# Make sure we can run imports from the project root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.services.document.pdf_parser import DocumentProcessor
from backend.services.document.table_parsing import TableParser
import pdfplumber

def run_test_on_pdf(target_pdf, results_path, run_timestamp):
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] STARTING ENGINE: {os.path.basename(target_pdf)}")
    processor = DocumentProcessor()
    
    start_time = time.time()
    
    try:
        with pdfplumber.open(target_pdf) as pdf:
            total_pages = len(pdf.pages)
    except Exception as e:
        print(f"FAILED to open PDF {target_pdf}: {e}")
        return
        
    print(f"Target acquired: {total_pages} pages detected.")
    
    all_pages = []
    completed_count = 0
    
    # 1. Pipeline Execution
    for page_json in processor.stream_document_pipeline(target_pdf, max_workers=os.cpu_count() or 4):
        completed_count += 1
        pct = (completed_count / total_pages) * 100
        print(f"  -> Extracted Page {page_json['page_number']} ({pct:.1f}%)")
        all_pages.append(page_json)

    # 2. Structural Realignment
    all_pages.sort(key=lambda x: x["page_number"])
    
    # 3. Post-Processing: Domain-Driven Math Healing
    print("Applying Math & Domain-Driven Healing (Cross-Page Table Fusion & Watermark Deletion)...")
    all_pages = TableParser.heal_cross_page_tables(all_pages)
    all_pages = DocumentProcessor._strip_exclusion_zones(all_pages)
    
    print("Performing Spatial Semantic Linking (Injecting text context into Image Nodes)...")
    from backend.services.document.semantic_linker import SemanticLinker
    all_pages = SemanticLinker.link_semantic_context(all_pages)
    
    # 4. Binary Serialization (.loom File Creation)
    print("Generating Intelligence-Ready .loom Binary Graph...")
    from backend.services.loom_service.weaver import LoomWeaver
    from backend.services.loom_service.viewer import LoomViewer
    
    base_name = os.path.splitext(os.path.basename(target_pdf))[0]
    output_folder = os.path.join(results_path, f"{base_name}_{run_timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    
    loom_path = os.path.join(output_folder, f"{base_name}.loom")
    weaver = LoomWeaver()
    weaver.weave(all_pages, loom_path)
    
    duration = time.time() - start_time
    
    # 5. Save Hierarchical Folder JSON Payload Map
    table_index = 1
    image_index = 1
    hierarchical_pages = json.loads(json.dumps(all_pages)) # Deep copy for popping
    
    for page in hierarchical_pages:
        for item in page.get("content", []):
            if item.get("type", "") == "table":
                table_filename = f"table_{table_index}.json"
                table_path = os.path.join(output_folder, table_filename)
                with open(table_path, "w", encoding="utf-8") as f:
                    json.dump({"table_id": table_index, "data": item.pop("data")}, f, indent=4, ensure_ascii=False)
                item["table_file"] = table_filename
                table_index += 1
            elif item.get("type", "") == "image":
                image_filename = f"image_{image_index}.json"
                image_path = os.path.join(output_folder, image_filename)
                with open(image_path, "w", encoding="utf-8") as f:
                    json.dump({"image_id": image_index, "base64_data": item.pop("base64_data", "")}, f, indent=4, ensure_ascii=False)
                item["image_file"] = image_filename
                image_index += 1
                
    main_output_filename = os.path.join(output_folder, "main_document.json")
    with open(main_output_filename, "w", encoding="utf-8") as f:
        json.dump(hierarchical_pages, f, indent=4, ensure_ascii=False)
 
    # 6. Semantic Integrity Audit
    viewer = LoomViewer(loom_path)
    viewer.audit()
    
    print(f"SUCCESS: {base_name} complete in {duration:.2f} seconds! ({duration/total_pages:.2f}s per page)")
    print(f"Inventory: intelligence-ready outputs generated at: {output_folder}")


if __name__ == '__main__':
    # Define test parameters
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))
    test_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Check if we should create the results dir
    results_dir = os.path.join(test_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Discover all target documents
    pdf_files = glob.glob(os.path.join(assets_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in {assets_dir}.")
        sys.exit(1)
        
    print("=========================================")
    print(" DOCLOOM ENGINE - ENTERPRISE STRESS TEST ")
    print("=========================================")
    print(f"Files to process: {len(pdf_files)}")
    print(f"Output directory: {results_dir}\n")
    
    for pdf_path in pdf_files:
        run_test_on_pdf(pdf_path, results_dir, timestamp)
        
    print("=========================================")
    print("ALL TESTS FINISHED.")
    print("=========================================")
