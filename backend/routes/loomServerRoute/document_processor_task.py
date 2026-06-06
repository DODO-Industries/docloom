import os
import sys
import datetime
import time
import json

# Resolve Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.services.document.pdf_parser import DocumentProcessor
from backend.services.document.table_parsing import TableParser
from backend.services.document.semantic_linker import SemanticLinker
from backend.routes.loomServerRoute.state_manager import (
    get_cognition_state,
    set_cognition_state,
    get_shard_groups,
    set_shard_groups,
    load_brain_file,
    init_cognition_engine,
    save_brain_file,
    auto_cluster_shard_groups
)

# Global process status registry
PROCESS_STATUS = {}

def update_status(job_id, progress, msg):
    PROCESS_STATUS[job_id] = {"p": progress, "m": msg, "t": time.time()}

def get_process_status(job_id):
    return PROCESS_STATUS.get(job_id, {"p": 0, "m": "Job not found"})

def process_pdf_task(pdf_path: str, output_root: str, job_id: str):
    """Background task to process a PDF into a .loom graph."""
    update_status(job_id, 5, "Initializing Pipeline...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(output_root, f"{base_name}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        update_status(job_id, 15, f"Reading PDF: {base_name}...")
        processor = DocumentProcessor()
        
        all_pages = []
        for page_json in processor.stream_document_pipeline(pdf_path):
            all_pages.append(page_json)
            # Rough progress estimate based on pages (placeholder)
            curr = 15 + (len(all_pages) * 2)
            update_status(job_id, min(40, curr), f"Parsing Pages ({len(all_pages)})...")

        update_status(job_id, 45, "Post-Processing Structure...")
        all_pages.sort(key=lambda x: x["page_number"])
        all_pages = TableParser.heal_cross_page_tables(all_pages)
        all_pages = DocumentProcessor._strip_exclusion_zones(all_pages)
        all_pages = SemanticLinker.link_semantic_context(all_pages)
        
        update_status(job_id, 65, "Weaving Knowledge Shards into Real-time Field...")
        state = get_cognition_state()
        if state is None:
            if not load_brain_file():
                init_cognition_engine()
                state = get_cognition_state()
                
        from backend.routes.loomServerRoute.weaver_helper import RealtimeWeaver
        real_weaver = RealtimeWeaver()
        shards = real_weaver.weave_into_state(all_pages, state, save_callback=save_brain_file)
        
        # Re-cluster to include the new shards
        total_concepts_list = list(state.semantic_field.concept_embeddings.keys())
        concept_count = len(total_concepts_list)
        if concept_count > 0:
            cluster_count = min(12, max(2, concept_count // 15))
            try:
                new_groups = auto_cluster_shard_groups(n_clusters=cluster_count)
                set_shard_groups(new_groups)
            except:
                set_shard_groups({})
                
        save_brain_file()
        
        update_status(job_id, 85, "Finalizing Multi-Shard Atlas...")
        table_idx, img_idx = 1, 1
        for page in all_pages:
            for item in page.get("content", []):
                if item.get("type") == "table" and "data" in item:
                    fname = f"table_{table_idx}.json"
                    with open(os.path.join(output_folder, fname), "w") as f:
                        json.dump({"id": table_idx, "data": item.pop("data")}, f, indent=4)
                    item["table_file"] = fname
                    table_idx += 1
                elif item.get("type") == "image" and "base64_data" in item:
                    fname = f"image_{img_idx}.json"
                    with open(os.path.join(output_folder, fname), "w") as f:
                        json.dump({"id": img_idx, "base64": item.pop("base64_data")}, f, indent=4)
                    item["image_file"] = fname
                    img_idx += 1
            
        with open(os.path.join(output_folder, "main_document.json"), "w", encoding="utf-8") as f:
            json.dump(all_pages, f, indent=4, ensure_ascii=False)
            
        update_status(job_id, 100, f"SUCCESS: Ingested {len(shards)} thought cells into Real-time Substrate.")
        print(f"Background process SUCCESS: {base_name}")
    except Exception as e:
        update_status(job_id, -1, f"FAILED: {str(e)}")
        print(f"Background process FAILED for {base_name}: {e}")
