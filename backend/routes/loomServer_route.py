import os
import sys
import shutil
import json
import datetime
import time
from typing import List, Optional
from contextlib import asynccontextmanager

# Add project root to sys.path for robust imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from fastapi import FastAPI, APIRouter, HTTPException, Query, Request, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import msgpack

# Service Imports
from backend.services.loom_service.orchestration.loomServer_Service import LoomServerService
from backend.services.document.pdf_parser import DocumentProcessor
from backend.services.document.table_parsing import TableParser
from backend.services.document.semantic_linker import SemanticLinker
from backend.services.loom_service.substrate.substrate_weaver import SubstrateWeaver
from backend.services.loom_service.substrate.neural_viewer import NeuralViewer
from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("LoomServerRoute")
router = APIRouter(prefix="/loom", tags=["Loom"])

# Paths
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, exist_ok=True)

VISUALIZER_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "utils", "webVisualizer", "loom_explorer.html"))

# Service Singleton
_service = None

def get_service():
    global _service
    if _service is None:
        _service = LoomServerService()
    return _service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eager loading on startup for millisecond response times
    log_service(logger, "Starting DocLoom Neural Engine (Eager Load)...", "info")
    try:
        get_service()
        log_service(logger, "Neural Engine LOADED and ready for real-time inference.", "info")
    except Exception as e:
        log_service(logger, f"FAILED to eager load Neural Engine: {e}", "error")
    yield
    log_service(logger, "Shutting down Neural Engine...", "info")

# Initialize app
app = FastAPI(title="DocLoom Neural Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background Task
# Global process status registry
PROCESS_STATUS = {}

def update_status(job_id, progress, msg):
    PROCESS_STATUS[job_id] = {"p": progress, "m": msg, "t": time.time()}

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
        
        update_status(job_id, 65, "Weaving Knowledge Shards...")
        loom_path = os.path.join(output_folder, f"{base_name}.loom")
        weaver = SubstrateWeaver() # Now using singleton transformer internally
        weaver.weave(all_pages, loom_path)
        
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
            
        atlas_path = os.path.join(output_folder, f"atlas_{base_name}.loom")
        with open(os.path.join(output_folder, "main_document.json"), "w", encoding="utf-8") as f:
            json.dump(all_pages, f, indent=4, ensure_ascii=False)
            
        update_status(job_id, 100, f"SUCCESS: {atlas_path}")
        print(f"Background process SUCCESS: {base_name}")
    except Exception as e:
        update_status(job_id, -1, f"FAILED: {str(e)}")
        print(f"Background process FAILED for {base_name}: {e}")

# Models
class ProcessRequest(BaseModel):
    text: str
    extract_entities: bool = True
    extract_concepts: bool = True
    generate_embeddings: bool = True

class NeuralActivationRequest(BaseModel):
    master_path: str
    query: str
    depth: int = 3
    width: int = 2

# Routes
@router.get("/api/status")
async def get_task_status(job_id: str = Query(...)):
    """Returns the current progress and status of a background job."""
    return PROCESS_STATUS.get(job_id, {"p": 0, "m": "Job not found"})

@router.post("/api/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = f"job_{int(time.time())}"
    file_path = os.path.join(RESULTS_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if file.filename.lower().endswith('.pdf'):
        background_tasks.add_task(process_pdf_task, file_path, RESULTS_DIR, job_id)
        return {"success": True, "status": "Processing Started", "job_id": job_id}
    
    if file.filename.lower().endswith('.txt'):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            service = get_service()
            shards = service.process_text(text, True, True, True)
            return {"success": True, "status": "Text Woven", "shards": shards, "job_id": job_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    return {"success": True, "status": "Uploaded", "job_id": job_id}

@router.get("/visualizer", response_class=HTMLResponse)
async def get_visualizer():
    if not os.path.exists(VISUALIZER_PATH):
        return f"<h1>Explorer HTML not found</h1>"
    with open(VISUALIZER_PATH, "r", encoding="utf-8") as f:
        return f.read()

@router.get("/api/load")
async def load_atlas(path: str = Query(...)):
    if not os.path.exists(path): raise HTTPException(status_code=404)
    return FileResponse(path)

@router.get("/api/load_shard")
async def load_shard(master_path: str = Query(...), shard_id: str = Query(...)):
    """Loads a specific binary shard file from the atlas directory."""
    try:
        master_path = master_path.strip("\"'")
        base_dir = os.path.dirname(master_path)
        # The master atlas is usually named 'atlas_FILENAME.loom'
        # Shards are named 'FILENAME_shard_X.loom'
        atlas_filename = os.path.basename(master_path)
        base_doc_name = atlas_filename.replace("atlas_", "").replace(".loom", "")
        
        shard_path = os.path.join(base_dir, f"{base_doc_name}_shard_{shard_id}.loom")
        
        if not os.path.exists(shard_path):
            # Fallback to simple shard_X.loom just in case
            alt_path = os.path.join(base_dir, f"shard_{shard_id}.loom")
            if os.path.exists(alt_path):
                shard_path = alt_path
            else:
                raise HTTPException(status_code=404, detail=f"Shard {shard_id} not found. Looked for: {shard_path}")
        
        return FileResponse(shard_path)
    except Exception as e:
        log_service(logger, f"ERROR loading shard {shard_id}: {e}", "error")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/activate")
async def activate_graph(request: NeuralActivationRequest):
    viewer = NeuralViewer(request.master_path)
    if not viewer.data: raise HTTPException(status_code=500)
    service = get_service()
    return service.activate_loom(viewer, request.query, request.depth, request.width)

@router.post("/api/live_ingest")
async def live_ingest(request: ProcessRequest):
    service = get_service()
    results = service.process_text(request.text, request.extract_entities, request.extract_concepts, True)
    return {"success": True, "shards": results}

@router.get("/api/search")
async def search_query(q: str = Query(...)):
    service = get_service()
    return {"query": q, "embedding": service.get_query_embedding(q)}

# Finalize App
app.include_router(router)

LIBS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "utils", "webVisualizer", "libs"))
if os.path.exists(LIBS_DIR):
    app.mount("/libs", StaticFiles(directory=LIBS_DIR), name="libs")

if __name__ == "__main__":
    import uvicorn
    print(f"\nLoom Visualizer available at: http://localhost:8000/loom/visualizer")
    uvicorn.run(app, host="0.0.0.0", port=8000)
