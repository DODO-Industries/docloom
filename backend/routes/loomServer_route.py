import os
import sys
import shutil
import json
import datetime
import time
import numpy as np
from typing import List, Optional, Dict
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
from backend.services.loom_service.cognition.attractor_dynamics import get_embedding_model, safe_normalize

logger = setup_logger("LoomServerRoute")
router = APIRouter(prefix="/loom", tags=["Loom"])

# Paths — brain_state.json lives inside the project so the user can see it
RESULTS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "backend", "results"))
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, exist_ok=True)

COGNITION_STATE_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, ".brain_data"))
if not os.path.exists(COGNITION_STATE_DIR):
    os.makedirs(COGNITION_STATE_DIR, exist_ok=True)

VISUALIZER_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "Vizualization", "index.html"))

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
        
        update_status(job_id, 65, "Weaving Knowledge Shards into Real-time Field...")
        global _cognition_state, _shard_groups
        if _cognition_state is None:
            if not load_brain_file():
                init_cognition_engine()
                
        from backend.services.loom_service.weaver import RealtimeWeaver
        real_weaver = RealtimeWeaver()
        shards = real_weaver.weave_into_state(all_pages, _cognition_state, save_callback=save_brain_file)
        
        # Re-cluster to include the new shards
        total_concepts_list = list(_cognition_state.semantic_field.concept_embeddings.keys())
        concept_count = len(total_concepts_list)
        if concept_count > 0:
            cluster_count = min(12, max(2, concept_count // 15))
            try:
                _shard_groups = auto_cluster_shard_groups(n_clusters=cluster_count)
            except:
                _shard_groups = {}
                
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
                
            global _cognition_state, _shard_groups
            if _cognition_state is None:
                if not load_brain_file():
                    init_cognition_engine()
                    
            from backend.services.loom_service.weaver import RealtimeWeaver
            pages = [{"page_number": 1, "content": [{"type": "paragraph", "text": text}]}]
            real_weaver = RealtimeWeaver()
            shards = real_weaver.weave_into_state(pages, _cognition_state, save_callback=save_brain_file)
            
            # Re-cluster to include the new shards
            total_concepts_list = list(_cognition_state.semantic_field.concept_embeddings.keys())
            concept_count = len(total_concepts_list)
            if concept_count > 0:
                cluster_count = min(12, max(2, concept_count // 15))
                try:
                    _shard_groups = auto_cluster_shard_groups(n_clusters=cluster_count)
                except:
                    _shard_groups = {}
                    
            save_brain_file()
            
            legacy_shards = [{"text": s["text"]} for s in shards]
            return {"success": True, "status": "Text Woven Directly into Realtime Field", "shards": legacy_shards, "job_id": job_id}
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

# --- PHASE 4: COGNITIVE SUBSTRATE SIMULATION ENDPOINTS ---

_cognition_state = None
_cognition_pipeline = None
_cognition_controller = None
_shard_groups: Dict[str, List[str]] = {}  # shard label -> list of thought concept strings

class CognitionTickRequest(BaseModel):
    text: str
    intensity: float = 0.9
    goal: Optional[str] = None
    command: Optional[str] = None
    dominant_concept: Optional[str] = None

def init_cognition_engine():
    global _cognition_state, _cognition_pipeline, _cognition_controller
    from backend.services.loom_service.cognition.unified_cognitive_field import GlobalCognitiveState
    from backend.services.loom_service.cognition.core.cognitive_controller import CognitiveController
    from backend.services.loom_service.orchestration.cognition_pipeline import CognitionPipeline
    from backend.services.loom_service.planning.hierarchical_planner import HierarchicalPlanner
    from backend.services.loom_service.reasoning.symbolic_reasoner import SymbolicReasoner
    from backend.services.loom_service.safety.action_sandbox import ActionSandbox
    from backend.services.loom_service.agentic_execution.autonomous_executor import AutonomousExecutor

    _cognition_state = GlobalCognitiveState()
    _planner = HierarchicalPlanner()
    _executor = AutonomousExecutor()
    _reasoner = SymbolicReasoner()
    _sandbox = ActionSandbox()

    _cognition_state.planner = _planner
    _cognition_state.executor = _executor
    _cognition_state.reasoner = _reasoner

    _cognition_controller = CognitiveController(global_state=_cognition_state)
    _cognition_pipeline = CognitionPipeline(
        core_workspace=_cognition_controller.workspace,
        scheduler=_cognition_controller.scheduler,
        safety_sandbox=_sandbox,
        controller=_cognition_controller
    )

def project_concepts_to_3d(concepts_dict: dict):
    """
    Project concepts (registered in global cognitive state) to 3D space using NumPy-based PCA.
    Returns: (coords_dict, latent_field_proj, working_latent_proj)
    """
    import numpy as np
    concepts_list = list(concepts_dict.keys())
    if not concepts_list:
        return {}, [0, 0, 0], [0, 0, 0]
        
    embeddings = [concepts_dict[c] for c in concepts_list]
    X = np.array(embeddings)
    
    latent_field = getattr(_cognition_state, "latent_field", None)
    working_latent = getattr(_cognition_state, "working_latent", None)
    
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    # 1 Sample: Place in center, slightly offsets orbs to make them visible
    if n_samples == 1:
        coords = {concepts_list[0]: [0.0, 0.0, 0.0]}
        lf_proj = [100.0, 50.0, -50.0]
        wl_proj = [-100.0, -50.0, 50.0]
        return coords, lf_proj, wl_proj
        
    # 2 Samples: Space them far apart along the X axis so they don't clump
    if n_samples == 2:
        coords = {
            concepts_list[0]: [-250.0, 0.0, 0.0],
            concepts_list[1]: [250.0, 0.0, 0.0]
        }
        lf_proj = [0.0, 100.0, 0.0]
        wl_proj = [0.0, -100.0, 0.0]
        return coords, lf_proj, wl_proj
        
    # 3 or more Samples: Calculate PCA projections
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # Calculate covariance and eigenvalues
    cov = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort eigenvalues descending and take top 3
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    top_3_vectors = eigenvectors[:, :3]
    
    # Project data
    X_projected = np.dot(X_centered, top_3_vectors)
    
    # Auto-scale coordinate system so that the furthest node is exactly 350.0 units away from center.
    # This prevents the concepts from clumping together in the center.
    max_val = np.max(np.abs(X_projected))
    scale = 350.0
    if max_val > 1e-5:
        scale = 350.0 / max_val
    else:
        # If mathematically identical, add random jitter/spread noise
        for i in range(n_samples):
            X_projected[i] += np.random.normal(0, 0.1, 3)
        max_val = np.max(np.abs(X_projected))
        if max_val > 1e-5:
            scale = 350.0 / max_val
            
    X_projected = X_projected * scale
    
    coords = {}
    for i, c in enumerate(concepts_list):
        coords[c] = [
            float(X_projected[i, 0]),
            float(X_projected[i, 1]),
            float(X_projected[i, 2])
        ]
        
    lf_proj = [0.0, 0.0, 0.0]
    if latent_field is not None:
        centered_lf = latent_field - X_mean
        lf_p = np.dot(centered_lf, top_3_vectors)
        lf_proj = [float(lf_p[0]) * scale, float(lf_p[1]) * scale, float(lf_p[2]) * scale]
        
    wl_proj = [0.0, 0.0, 0.0]
    if working_latent is not None:
        centered_wl = working_latent - X_mean
        wl_p = np.dot(centered_wl, top_3_vectors)
        wl_proj = [float(wl_p[0]) * scale, float(wl_p[1]) * scale, float(wl_p[2]) * scale]
        
    # Cache PCA parameters for query projection
    if _cognition_state is not None:
        _cognition_state._pca_mean = X_mean
        _cognition_state._pca_vectors = top_3_vectors
        _cognition_state._pca_scale = scale
        
    return coords, lf_proj, wl_proj

def save_brain_file():
    global _cognition_state, _shard_groups
    if _cognition_state is None:
        return
    try:
        # Serialize concept embeddings
        emb_dict = {}
        for k, v in _cognition_state.semantic_field.concept_embeddings.items():
            emb_dict[k] = v.tolist()

        emb_file_path = os.path.join(COGNITION_STATE_DIR, "brain_embeddings.bin")
        with open(emb_file_path, "wb") as f:
            f.write(msgpack.packb(emb_dict, use_bin_type=True))

        # 2. Coordinates & Velocities: raw float32 binary arrays
        concept_keys = sorted(list(_cognition_state.concept_coords.keys()))
        
        coords_list = []
        for k in concept_keys:
            coords_list.append(_cognition_state.concept_coords[k].astype(np.float32))
        
        if coords_list:
            coords_array = np.vstack(coords_list)
            coords_bytes = coords_array.tobytes()
        else:
            coords_bytes = b""
            
        coords_file_path = os.path.join(COGNITION_STATE_DIR, "coordinates.bin")
        with open(coords_file_path, "wb") as f:
            f.write(coords_bytes)
            
        vel_list = []
        for k in concept_keys:
            vel_list.append(_cognition_state.concept_velocities[k].astype(np.float32))
            
        if vel_list:
            vel_array = np.vstack(vel_list)
            vel_bytes = vel_array.tobytes()
        else:
            vel_bytes = b""
            
        vel_file_path = os.path.join(COGNITION_STATE_DIR, "velocities.bin")
        with open(vel_file_path, "wb") as f:
            f.write(vel_bytes)

        # Serialize active assemblies
        assemblies = []
        for a in _cognition_state.active_assemblies:
            assemblies.append({
                "id": a.assembly_id,
                "dominant": a.dominant_concept,
                "activations": a.node_activations,
                "confidence": a.confidence,
                "stability": a.stability,
                "basin_strength": a.meta_data.get("basin_strength", 0.0)
            })

        # Serialize working memory graph
        wm_nodes = []
        for node, data in _cognition_state.memory.graph.nodes(data=True):
            wm_nodes.append({
                "id": node,
                "activation": float(data.get("activation", 0.0)),
                "stability": float(data.get("stability", 0.0)),
                "temporal_depth": int(data.get("temporal_depth", 0))
            })
        wm_edges = []
        for u, v, data in _cognition_state.memory.graph.edges(data=True):
            wm_edges.append({
                "from": u,
                "to": v,
                "weight": float(data.get("weight", 0.0))
            })
            
        wm_data = {
            "nodes": wm_nodes,
            "edges": wm_edges
        }
        wm_file_path = os.path.join(COGNITION_STATE_DIR, "working_memory.bin")
        with open(wm_file_path, "wb") as f:
            f.write(msgpack.packb(wm_data, use_bin_type=True))

        # Serialize causal links
        causal_edges = []
        for u, v, data in _cognition_state.causal.causal_matrix.edges(data=True):
            causal_edges.append({
                "from": u,
                "to": v,
                "weight": float(data.get("weight", 0.0)),
                "frequency": int(data.get("frequency", 1))
            })
            
        causal_file_path = os.path.join(COGNITION_STATE_DIR, "causal_links.bin")
        with open(causal_file_path, "wb") as f:
            f.write(msgpack.packb(causal_edges, use_bin_type=True))

        # Serialize meta shards
        meta_shards = []
        for meta_id, shard in _cognition_state.meta_shards.meta_store.items():
            meta_shards.append({
                "id": meta_id,
                "label": shard.emergent_label,
                "children": shard.children,
                "stability": float(shard.stability),
                "abstraction_level": int(shard.abstraction_level)
            })

        # Persist concept-to-thought mapping (for click-to-expand)
        concept_thoughts = getattr(_cognition_state, '_concept_thoughts', {})
        
        # Serialize physics tensors
        physics_file_path = os.path.join(COGNITION_STATE_DIR, "physics_tensors.bin")
        with open(physics_file_path, "wb") as f:
            f.write(msgpack.packb(_cognition_state.physics_tensors, use_bin_type=True))
            
        # Serialize replay logs (episodic tensors) MsgPack + Zstd
        import zstandard as zstd
        replay_list = [v.tolist() for v in _cognition_state.episodic_tensors]
        cctx = zstd.ZstdCompressor()
        compressed_replay = cctx.compress(msgpack.packb(replay_list, use_bin_type=True))
        replay_file_path = os.path.join(COGNITION_STATE_DIR, "replay.bin")
        with open(replay_file_path, "wb") as f:
            f.write(compressed_replay)

        metadata = {
            "tick": _cognition_state.tick,
            "entropy": _cognition_state.entropy,
            "coherence": _cognition_state.coherence,
            "pressure": _cognition_state.pressure,
            "surprise": _cognition_state.surprise,
            "energy_budget": _cognition_state.energy_budget,
            "latent_field": _cognition_state.latent_field.tolist() if _cognition_state.latent_field is not None else None,
            "working_latent": _cognition_state.working_latent.tolist() if _cognition_state.working_latent is not None else None,
            "concept_keys": concept_keys,
            "active_assemblies": assemblies,
            "attention_focus": list(_cognition_state.attention_focus),
            "meta_shards": meta_shards,
            "shard_groups": _shard_groups,
            "concept_thoughts": concept_thoughts
        }

        metadata_file_path = os.path.join(COGNITION_STATE_DIR, "metadata.bin")
        with open(metadata_file_path, "wb") as f:
            f.write(msgpack.packb(metadata, use_bin_type=True))
    except Exception as e:
        print("Error saving brain state file:", e)

def load_brain_file():
    global _cognition_state, _cognition_pipeline, _cognition_controller, _shard_groups
    
    metadata_file_path = os.path.join(COGNITION_STATE_DIR, "metadata.bin")
    legacy_file_path = os.path.join(COGNITION_STATE_DIR, "brain_state.json")
    
    if not os.path.exists(metadata_file_path) and not os.path.exists(legacy_file_path):
        return False
        
    try:
        init_cognition_engine()
        
        # Check if new metadata.bin exists
        if os.path.exists(metadata_file_path):
            with open(metadata_file_path, "rb") as f:
                metadata = msgpack.unpackb(f.read(), raw=False)
                
            _cognition_state.tick = metadata.get("tick", 0)
            _cognition_state.entropy = metadata.get("entropy", 0.5)
            _cognition_state.coherence = metadata.get("coherence", 0.5)
            _cognition_state.pressure = metadata.get("pressure", 0.0)
            _cognition_state.surprise = metadata.get("surprise", 0.0)
            _cognition_state.energy_budget = metadata.get("energy_budget", 1.0)
            
            lf = metadata.get("latent_field")
            if lf is not None:
                _cognition_state.latent_field = np.array(lf)
            wl = metadata.get("working_latent")
            if wl is not None:
                _cognition_state.working_latent = np.array(wl)
                
            concept_keys = metadata.get("concept_keys", [])
            _shard_groups = metadata.get("shard_groups", {})
            _cognition_state._concept_thoughts = metadata.get("concept_thoughts", {})
            _cognition_state.attention_focus = metadata.get("attention_focus", [])
            
            # Load active assemblies
            from backend.services.loom_service.cognition.attractor_dynamics import CognitiveAssembly
            _cognition_state.active_assemblies = []
            for a in metadata.get("active_assemblies", []):
                _cognition_state.active_assemblies.append(CognitiveAssembly(
                    assembly_id=a["id"],
                    dominant_concept=a["dominant"],
                    node_activations=a["activations"],
                    confidence=a["confidence"],
                    stability=a["stability"],
                    phase_coherence=_cognition_state.coherence,
                    meta_data={"basin_strength": a["basin_strength"]}
                ))
                
            # Load meta shards
            from backend.services.loom_service.cognition.attractor_dynamics import MetaShard
            _cognition_state.meta_shards.meta_store.clear()
            for ms in metadata.get("meta_shards", []):
                _cognition_state.meta_shards.meta_store[ms["id"]] = MetaShard(
                    meta_id=ms["id"],
                    children=ms["children"],
                    field_vec=np.zeros(1),
                    abstraction_level=ms["abstraction_level"],
                    stability=ms["stability"],
                    emergent_label=ms["label"]
                )
                
            # Load concept coordinates from raw float32 binary
            coords_file_path = os.path.join(COGNITION_STATE_DIR, "coordinates.bin")
            if os.path.exists(coords_file_path) and concept_keys:
                with open(coords_file_path, "rb") as f:
                    coords_bytes = f.read()
                if coords_bytes:
                    coords_array = np.frombuffer(coords_bytes, dtype=np.float32).reshape(len(concept_keys), -1)
                    _cognition_state.concept_coords = {k: coords_array[idx] for idx, k in enumerate(concept_keys)}
            else:
                _cognition_state.concept_coords = {}
                
            # Load concept velocities from raw float32 binary
            vel_file_path = os.path.join(COGNITION_STATE_DIR, "velocities.bin")
            if os.path.exists(vel_file_path) and concept_keys:
                with open(vel_file_path, "rb") as f:
                    vel_bytes = f.read()
                if vel_bytes:
                    vel_array = np.frombuffer(vel_bytes, dtype=np.float32).reshape(len(concept_keys), -1)
                    _cognition_state.concept_velocities = {k: vel_array[idx] for idx, k in enumerate(concept_keys)}
            else:
                _cognition_state.concept_velocities = {}
                
            # Load causal links from msgpack
            causal_file_path = os.path.join(COGNITION_STATE_DIR, "causal_links.bin")
            _cognition_state.causal.causal_matrix.clear()
            if os.path.exists(causal_file_path):
                with open(causal_file_path, "rb") as f:
                    causal_edges = msgpack.unpackb(f.read(), raw=False)
                for l in causal_edges:
                    _cognition_state.causal.causal_matrix.add_edge(
                        l["from"],
                        l["to"],
                        weight=l["weight"],
                        frequency=l["frequency"],
                        last_seen=time.time()
                    )
                    
            # Load working memory graph from msgpack
            wm_file_path = os.path.join(COGNITION_STATE_DIR, "working_memory.bin")
            _cognition_state.memory.graph.clear()
            if os.path.exists(wm_file_path):
                with open(wm_file_path, "rb") as f:
                    wm_data = msgpack.unpackb(f.read(), raw=False)
                for n in wm_data.get("nodes", []):
                    _cognition_state.memory.graph.add_node(
                        n["id"],
                        activation=n["activation"],
                        stability=n["stability"],
                        temporal_depth=n["temporal_depth"]
                    )
                for e in wm_data.get("edges", []):
                    _cognition_state.memory.graph.add_edge(
                        e["from"],
                        e["to"],
                        weight=e["weight"]
                    )
                    
            # Load physics tensors
            physics_file_path = os.path.join(COGNITION_STATE_DIR, "physics_tensors.bin")
            if os.path.exists(physics_file_path):
                with open(physics_file_path, "rb") as f:
                    _cognition_state.physics_tensors = msgpack.unpackb(f.read(), raw=False)
            else:
                _cognition_state.physics_tensors = {}
                
            # Load replay logs (episodic tensors) from Zstd + MsgPack
            replay_file_path = os.path.join(COGNITION_STATE_DIR, "replay.bin")
            _cognition_state.episodic_tensors = []
            if os.path.exists(replay_file_path):
                import zstandard as zstd
                dctx = zstd.ZstdDecompressor()
                with open(replay_file_path, "rb") as f:
                    compressed_replay = f.read()
                if compressed_replay:
                    replay_list = msgpack.unpackb(dctx.decompress(compressed_replay), raw=False)
                    _cognition_state.episodic_tensors = [np.array(v) for v in replay_list]
                    
        # Backward-compatible fallback to legacy brain_state.json
        elif os.path.exists(legacy_file_path):
            with open(legacy_file_path, "r", encoding="utf-8") as f:
                state_dict = json.load(f)
                
            _cognition_state.tick = state_dict.get("tick", 0)
            _cognition_state.entropy = state_dict.get("entropy", 0.5)
            _cognition_state.coherence = state_dict.get("coherence", 0.5)
            _cognition_state.pressure = state_dict.get("pressure", 0.0)
            _cognition_state.surprise = state_dict.get("surprise", 0.0)
            _cognition_state.energy_budget = state_dict.get("energy_budget", 1.0)
            
            lf = state_dict.get("latent_field")
            if lf is not None:
                _cognition_state.latent_field = np.array(lf)
            wl = state_dict.get("working_latent")
            if wl is not None:
                _cognition_state.working_latent = np.array(wl)
                
            from backend.services.loom_service.cognition.attractor_dynamics import CognitiveAssembly
            _cognition_state.active_assemblies = []
            for a in state_dict.get("active_assemblies", []):
                _cognition_state.active_assemblies.append(CognitiveAssembly(
                    assembly_id=a["id"],
                    dominant_concept=a["dominant"],
                    node_activations=a["activations"],
                    confidence=a["confidence"],
                    stability=a["stability"],
                    phase_coherence=_cognition_state.coherence,
                    meta_data={"basin_strength": a["basin_strength"]}
                ))
                
            _cognition_state.memory.graph.clear()
            wm = state_dict.get("working_memory", {})
            for n in wm.get("nodes", []):
                _cognition_state.memory.graph.add_node(
                    n["id"],
                    activation=n["activation"],
                    stability=n["stability"],
                    temporal_depth=n["temporal_depth"]
                )
            for e in wm.get("edges", []):
                _cognition_state.memory.graph.add_edge(
                    e["from"],
                    e["to"],
                    weight=e["weight"]
                )
                
            _cognition_state.causal.causal_matrix.clear()
            for l in state_dict.get("causal_links", []):
                _cognition_state.causal.causal_matrix.add_edge(
                    l["from"],
                    l["to"],
                    weight=l["weight"],
                    frequency=l["frequency"],
                    last_seen=time.time()
                )
                
            from backend.services.loom_service.cognition.attractor_dynamics import MetaShard
            _cognition_state.meta_shards.meta_store.clear()
            for ms in state_dict.get("meta_shards", []):
                _cognition_state.meta_shards.meta_store[ms["id"]] = MetaShard(
                    meta_id=ms["id"],
                    children=ms["children"],
                    field_vec=np.zeros(1),
                    abstraction_level=ms["abstraction_level"],
                    stability=ms["stability"],
                    emergent_label=ms["label"]
                )
                
            _cognition_state.attention_focus = state_dict.get("attention_focus", [])
            _shard_groups = state_dict.get("shard_groups", {})
            _cognition_state._concept_thoughts = state_dict.get("concept_thoughts", {})
            
            coords_dict = state_dict.get("concept_coords", {})
            _cognition_state.concept_coords = {k: np.array(v) for k, v in coords_dict.items()}
            
            vel_dict = state_dict.get("concept_velocities", {})
            _cognition_state.concept_velocities = {k: np.array(v) for k, v in vel_dict.items()}
            
            _cognition_state.physics_tensors = state_dict.get("physics_tensors", {})
            
            # Immediately save as binary state
            save_brain_file()
            
        # Load concept embeddings
        _cognition_state.semantic_field.concept_embeddings.clear()
        emb_file_path_bin = os.path.join(COGNITION_STATE_DIR, "brain_embeddings.bin")
        emb_file_path_json = os.path.join(COGNITION_STATE_DIR, "brain_embeddings.json")
        if os.path.exists(emb_file_path_bin):
            with open(emb_file_path_bin, "rb") as f:
                emb_dict = msgpack.unpackb(f.read(), raw=False)
        elif os.path.exists(emb_file_path_json):
            with open(emb_file_path_json, "r", encoding="utf-8") as f:
                emb_dict = json.load(f)
        else:
            emb_dict = {}
            
        for k, v in emb_dict.items():
            _cognition_state.semantic_field.concept_embeddings[k] = np.array(v)
            
        return True
    except Exception as e:
        print("Failed to load brain state file:", e)
        import traceback
        traceback.print_exc()
        return False

@router.post("/api/cognition/init")
async def init_cognition_state_route():
    try:
        if not load_brain_file():
            init_cognition_engine()
            save_brain_file()
        return {"success": True, "status": "Cognitive Substrate Initialized & Loaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/cognition/reset")
async def reset_cognition_state_route():
    try:
        # Delete all files in COGNITION_STATE_DIR to clear all state variables
        for filename in os.listdir(COGNITION_STATE_DIR):
            file_path = os.path.join(COGNITION_STATE_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
        init_cognition_engine()
        save_brain_file()
        return {"success": True, "status": "Cognitive Substrate Reset & Cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CognitionQueryRequest(BaseModel):
    query: str
    top_k: int = 5

def auto_cluster_shard_groups(n_clusters: int = 8) -> Dict[str, List[str]]:
    """
    Auto-derive shard groups based on hyper position latent matching
    and vector space physics 3D coordinate attraction.
    """
    global _cognition_state
    if _cognition_state is None:
        return {}
    emb_map = _cognition_state.semantic_field.concept_embeddings
    if not emb_map:
        return {}

    concepts = list(emb_map.keys())
    n_samples = len(concepts)

    if n_samples == 0:
        return {}

    # 1. Project concepts to 3D coords
    vecs = np.array([emb_map[c] for c in concepts], dtype=np.float32)
    coords = {}
    if n_samples == 1:
        coords[concepts[0]] = np.array([0.0, 0.0, 0.0])
    elif n_samples == 2:
        coords[concepts[0]] = np.array([-250.0, 0.0, 0.0])
        coords[concepts[1]] = np.array([250.0, 0.0, 0.0])
    else:
        X_mean = np.mean(vecs, axis=0)
        X_centered = vecs - X_mean
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        top_3_vectors = eigenvectors[:, :3]
        X_projected = np.dot(X_centered, top_3_vectors)
        max_val = np.max(np.abs(X_projected))
        scale = 350.0
        if max_val > 1e-5:
            scale = 350.0 / max_val
        else:
            for i in range(n_samples):
                X_projected[i] += np.random.normal(0, 0.1, 3)
            max_val = np.max(np.abs(X_projected))
            if max_val > 1e-5:
                scale = 350.0 / max_val
        X_projected = X_projected * scale
        for i, c in enumerate(concepts):
            coords[c] = X_projected[i]

    # 2. Build attraction graph
    adj = {c: [] for c in concepts}
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            c1, c2 = concepts[i], concepts[j]
            v1, v2 = vecs[i], vecs[j]
            p1, p2 = coords[c1], coords[c2]
            
            cos_sim = float(np.dot(v1, v2))
            dist_3d = float(np.linalg.norm(p1 - p2))
            
            # Attracted if cosine similarity >= 0.4 and 3D distance <= 150
            if cos_sim >= 0.4 and dist_3d <= 150.0:
                adj[c1].append(c2)
                adj[c2].append(c1)

    # 3. Find connected components (DFS)
    visited = set()
    components = []
    for c in concepts:
        if c not in visited:
            comp = []
            queue = [c]
            visited.add(c)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)

    # 4. Filter core clusters (size >= 3)
    core_clusters = []
    unclustered = []
    for comp in components:
        if len(comp) >= 3:
            core_clusters.append(comp)
        else:
            unclustered.extend(comp)

    # 5. Incremental cluster growth
    final_clusters = {f"Cluster_{k+1}": list(c) for k, c in enumerate(core_clusters)}
    
    for item in unclustered:
        best_cluster = None
        min_dist = float('inf')
        v_item = emb_map[item]
        p_item = coords[item]
        
        for cluster_name, members in final_clusters.items():
            member_coords = [coords[m] for m in members]
            centroid = np.mean(member_coords, axis=0)
            dist = float(np.linalg.norm(p_item - centroid))
            
            member_vecs = [emb_map[m] for m in members]
            avg_sim = float(np.mean([np.dot(v_item, mv) for mv in member_vecs]))
            
            if dist <= 150.0 and avg_sim >= 0.4:
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = cluster_name
                    
        if best_cluster:
            final_clusters[best_cluster].append(item)
        else:
            final_clusters[f"Unassigned_{item[:10]}"] = [item]

    # Rebuild meta shards dynamically from clusters of size >= 3
    _cognition_state.meta_shards.meta_store.clear()
    for cluster_name, members in final_clusters.items():
        if len(members) >= 3:
            child_vecs = [emb_map[m] for m in members]
            activations = [0.9 for _ in members]
            try:
                _cognition_state.meta_shards.create_meta_shard(
                    members, child_vecs, activations, _cognition_state.semantic_field
                )
            except Exception as e:
                print("Failed to create meta shard for cluster:", e)

    return final_clusters

@router.get("/api/cognition/stress_shards")
async def get_stress_shards():
    """Returns the raw array of shards from test_shards_100.json for streaming."""
    test_path = os.path.join(RESULTS_DIR, "test_shards_100.json")
    if not os.path.exists(test_path):
        raise HTTPException(status_code=404, detail="Test shards file not found")
    try:
        with open(test_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {"success": True, "shards": raw}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/cognition/inject_shard")
async def inject_single_shard(request: Request):
    """
    Registers, links, and ticks a single concept shard dynamically.
    Enables frontend to stream loading one-by-one to visualize physics attraction.
    """
    global _cognition_state, _shard_groups
    
    try:
        body = await request.json()
        shard = body.get("shard")
        prev_concept = body.get("prev_concept")
        
        if _cognition_state is None:
            if not load_brain_file():
                init_cognition_engine()
                
        if not hasattr(_cognition_state, '_concept_thoughts') or _cognition_state._concept_thoughts is None:
            _cognition_state._concept_thoughts = {}
            
        all_concepts = []
        seen_concepts = set(_cognition_state.semantic_field.concept_embeddings.keys())
        
        def clean_text(x):
            if x is None: return None
            if not isinstance(x, str): x = str(x)
            x = x.strip()
            return x if x else None
            
        def register_concept(concept, activation=0.9, stability=0.9, temporal_depth=0):
            concept = clean_text(concept)
            if not concept: return
            if concept in seen_concepts: return
            seen_concepts.add(concept)
            try: _cognition_state.semantic_field.register_concept(concept)
            except: pass
            try:
                _cognition_state.memory.graph.add_node(
                    concept, activation=activation, stability=stability, temporal_depth=temporal_depth
                )
            except: pass
            all_concepts.append(concept)
            
        def connect(a, b, weight=0.7, frequency=3):
            a = clean_text(a); b = clean_text(b)
            if not a or not b or a == b: return
            try:
                _cognition_state.causal.causal_matrix.add_edge(
                    a, b, weight=weight, frequency=frequency, last_seen=time.time()
                )
            except: pass

        def extract_strings(obj):
            results = []
            if isinstance(obj, str):
                txt = clean_text(obj)
                if txt: results.append(txt)
            elif isinstance(obj, list):
                for item in obj: results.extend(extract_strings(item))
            elif isinstance(obj, dict):
                preferred_keys = ["text", "thought", "thoughts", "content", "sentence", "idea", "concept", "description", "message", "value"]
                for key in preferred_keys:
                    if key in obj: results.extend(extract_strings(obj[key]))
                for k, v in obj.items():
                    if k.lower() in ["id", "uuid", "timestamp", "created_at", "updated_at"]: continue
                    results.extend(extract_strings(v))
            return results

        def process_cluster(cluster_name, thoughts):
            if not thoughts: return
            cleaned = []
            for t in thoughts:
                t = clean_text(t)
                if not t: continue
                cleaned.append(t)
                register_concept(t)
            if cluster_name:
                _cognition_state._concept_thoughts[cluster_name] = cleaned
            for i in range(len(cleaned) - 1):
                connect(cleaned[i], cleaned[i + 1], weight=0.85, frequency=5)

        # Decode shard formats
        if isinstance(shard, str):
            register_concept(shard)
        elif isinstance(shard, dict):
            if "label" in shard and "thoughts" in shard:
                cluster_name = clean_text(shard.get("label"))
                thoughts = extract_strings(shard.get("thoughts", []))
                process_cluster(cluster_name, thoughts)
            elif "text" in shard and "thoughts" in shard:
                concept = clean_text(shard.get("text"))
                thoughts = extract_strings(shard.get("thoughts", []))
                if concept:
                    register_concept(concept)
                    _cognition_state._concept_thoughts[concept] = thoughts
                for t in thoughts: register_concept(t)
                for t in thoughts: connect(concept, t, weight=0.9, frequency=5)
                for i in range(len(thoughts) - 1):
                    connect(thoughts[i], thoughts[i + 1], weight=0.8, frequency=4)
            else:
                extracted = extract_strings(shard)
                for text in extracted: register_concept(text)
                for i in range(len(extracted) - 1):
                    connect(extracted[i], extracted[i + 1], weight=0.55, frequency=2)
        elif isinstance(shard, list):
            for item in shard:
                if isinstance(item, str):
                    register_concept(item)
                else:
                    extracted = extract_strings(item)
                    for text in extracted: register_concept(text)
                    for i in range(len(extracted) - 1):
                        connect(extracted[i], extracted[i + 1], weight=0.55, frequency=2)
                        
        # Sequential trace connection across shard packages
        if prev_concept and all_concepts:
            connect(prev_concept, all_concepts[0], weight=0.35, frequency=1)
            
        # Re-cluster total concept pool
        total_concepts_list = list(_cognition_state.semantic_field.concept_embeddings.keys())
        concept_count = len(total_concepts_list)
        if concept_count > 0:
            cluster_count = min(12, max(2, concept_count // 15))
            try: _shard_groups = auto_cluster_shard_groups(n_clusters=cluster_count)
            except: _shard_groups = {}
        else:
            _shard_groups = {}
            
        # Tick the engine to trigger sync force
        if all_concepts:
            try: _cognition_state.process_tick({all_concepts[0]: 0.95})
            except: pass
            
        save_brain_file()
        state = await get_cognition_state()
        
        last_c = all_concepts[-1] if all_concepts else prev_concept
        return {
            "success": True,
            "added": len(all_concepts),
            "last_concept": last_c,
            "state": state
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/cognition/stress_test")
async def run_cognition_stress_test():
    """
    Universal cognition stress loader.
    
    Supports ALL formats:
    
    1. Flat string array
    [
        "thought 1",
        "thought 2"
    ]

    2. Simple objects
    [
        {"text": "hello"},
        {"thought": "world"}
    ]

    3. Label + thoughts
    [
        {
            "label": "Physics",
            "thoughts": [...]
        }
    ]

    4. Text + thoughts
    [
        {
            "text": "Gravity",
            "thoughts": [...]
        }
    ]

    5. Nested structures
    {
        "data": [...]
    }

    6. Arbitrary JSON trees
    {
        "universe": {
            "physics": {
                "gravity": [...]
            }
        }
    }

    7. Mixed formats together
    """

    global _cognition_state, _shard_groups

    try:
        import os
        import json
        import time
        import traceback
        from fastapi import HTTPException

        # =========================================================
        # ADDITIVE ENGINE LOADING
        # =========================================================

        if _cognition_state is None:
            if not load_brain_file():
                init_cognition_engine()

        if not hasattr(_cognition_state, '_concept_thoughts') or _cognition_state._concept_thoughts is None:
            _cognition_state._concept_thoughts = {}

        # =========================================================
        # LOAD FILE
        # =========================================================

        test_path = os.path.join(RESULTS_DIR, "test_shards_100.json")

        if not os.path.exists(test_path):
            raise HTTPException(
                status_code=404,
                detail=f"Test file not found: {test_path}"
            )

        with open(test_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if raw is None:
            raise HTTPException(
                status_code=400,
                detail="JSON file is empty"
            )

        # =========================================================
        # STORAGE
        # =========================================================

        all_concepts = []
        seen_concepts = set(_cognition_state.semantic_field.concept_embeddings.keys())

        # =========================================================
        # UTILITIES
        # =========================================================

        def clean_text(x):
            if x is None:
                return None

            if not isinstance(x, str):
                x = str(x)

            x = x.strip()

            if not x:
                return None

            return x

        # ---------------------------------------------------------

        def register_concept(
            concept,
            activation=0.9,
            stability=0.9,
            temporal_depth=0
        ):
            """
            Safely register concept everywhere
            """

            concept = clean_text(concept)

            if not concept:
                return

            if concept in seen_concepts:
                return

            seen_concepts.add(concept)

            try:
                _cognition_state.semantic_field.register_concept(concept)
            except Exception:
                pass

            try:
                _cognition_state.memory.graph.add_node(
                    concept,
                    activation=activation,
                    stability=stability,
                    temporal_depth=temporal_depth
                )
            except Exception:
                pass

            all_concepts.append(concept)

        # ---------------------------------------------------------

        def connect(a, b, weight=0.7, frequency=3):
            """
            Safe causal connection
            """

            a = clean_text(a)
            b = clean_text(b)

            if not a or not b:
                return

            if a == b:
                return

            try:
                _cognition_state.causal.causal_matrix.add_edge(
                    a,
                    b,
                    weight=weight,
                    frequency=frequency,
                    last_seen=time.time()
                )
            except Exception:
                pass

        # ---------------------------------------------------------

        def extract_strings(obj):
            """
            Recursively extract ALL strings from any JSON structure
            """

            results = []

            # --------------------------------------------
            # STRING
            # --------------------------------------------

            if isinstance(obj, str):
                txt = clean_text(obj)

                if txt:
                    results.append(txt)

            # --------------------------------------------
            # LIST
            # --------------------------------------------

            elif isinstance(obj, list):
                for item in obj:
                    results.extend(extract_strings(item))

            # --------------------------------------------
            # DICT
            # --------------------------------------------

            elif isinstance(obj, dict):

                # Primary fields
                preferred_keys = [
                    "text",
                    "thought",
                    "thoughts",
                    "content",
                    "sentence",
                    "idea",
                    "concept",
                    "description",
                    "message",
                    "value"
                ]

                # First extract preferred fields
                for key in preferred_keys:
                    if key in obj:
                        results.extend(extract_strings(obj[key]))

                # Then extract everything else
                for k, v in obj.items():

                    # skip metadata-like keys
                    if k.lower() in [
                        "id",
                        "uuid",
                        "timestamp",
                        "created_at",
                        "updated_at"
                    ]:
                        continue

                    results.extend(extract_strings(v))

            return results

        # ---------------------------------------------------------

        def process_cluster(cluster_name, thoughts):
            """
            Register cluster thoughts + internal links
            """

            if not thoughts:
                return

            cleaned = []

            for t in thoughts:
                t = clean_text(t)

                if not t:
                    continue

                cleaned.append(t)

                register_concept(t)

            # Store expandable memory
            if cluster_name:
                _cognition_state._concept_thoughts[
                    cluster_name
                ] = cleaned

            # Internal causal chain
            for i in range(len(cleaned) - 1):
                connect(
                    cleaned[i],
                    cleaned[i + 1],
                    weight=0.85,
                    frequency=5
                )

        # =========================================================
        # SMART FORMAT DETECTION
        # =========================================================

        # --------------------------------------------
        # CASE 1: ROOT LIST
        # --------------------------------------------

        if isinstance(raw, list):

            for item in raw:

                # --------------------------------
                # STRING ITEM
                # --------------------------------

                if isinstance(item, str):
                    register_concept(item)

                # --------------------------------
                # OBJECT ITEM
                # --------------------------------

                elif isinstance(item, dict):

                    # LABEL + THOUGHTS
                    if "label" in item and "thoughts" in item:

                        cluster_name = clean_text(item.get("label"))

                        thoughts = extract_strings(
                            item.get("thoughts", [])
                        )

                        process_cluster(
                            cluster_name,
                            thoughts
                        )

                    # TEXT + THOUGHTS
                    elif "text" in item and "thoughts" in item:

                        concept = clean_text(item.get("text"))

                        thoughts = extract_strings(
                            item.get("thoughts", [])
                        )

                        if concept:
                            register_concept(concept)

                            _cognition_state._concept_thoughts[
                                concept
                            ] = thoughts

                        for t in thoughts:
                            register_concept(t)

                        # Connect concept to thoughts
                        for t in thoughts:
                            connect(
                                concept,
                                t,
                                weight=0.9,
                                frequency=5
                            )

                        # Chain thoughts
                        for i in range(len(thoughts) - 1):
                            connect(
                                thoughts[i],
                                thoughts[i + 1],
                                weight=0.8,
                                frequency=4
                            )

                    # GENERIC OBJECT
                    else:

                        extracted = extract_strings(item)

                        for text in extracted:
                            register_concept(text)

                        # Local chaining
                        for i in range(len(extracted) - 1):
                            connect(
                                extracted[i],
                                extracted[i + 1],
                                weight=0.55,
                                frequency=2
                            )

        # --------------------------------------------
        # CASE 2: ROOT OBJECT
        # --------------------------------------------

        elif isinstance(raw, dict):

            extracted = extract_strings(raw)

            for text in extracted:
                register_concept(text)

            # Sequential linking
            for i in range(len(extracted) - 1):
                connect(
                    extracted[i],
                    extracted[i + 1],
                    weight=0.5,
                    frequency=2
                )

        # --------------------------------------------
        # CASE 3: FALLBACK
        # --------------------------------------------

        else:

            extracted = extract_strings(raw)

            for text in extracted:
                register_concept(text)

        # =========================================================
        # GLOBAL CAUSAL NETWORK
        # =========================================================

        for i in range(len(all_concepts) - 1):

            connect(
                all_concepts[i],
                all_concepts[i + 1],
                weight=0.35,
                frequency=1
            )

        # =========================================================
        # AUTO CLUSTERING
        # =========================================================

        total_concepts_list = list(_cognition_state.semantic_field.concept_embeddings.keys())
        concept_count = len(total_concepts_list)

        if concept_count > 0:

            cluster_count = min(
                12,
                max(
                    2,
                    concept_count // 15
                )
            )

            try:
                _shard_groups = auto_cluster_shard_groups(
                    n_clusters=cluster_count
                )
            except Exception:
                _shard_groups = {}

        else:
            _shard_groups = {}

        # =========================================================
        # INTER-CLUSTER LINKS
        # =========================================================

        try:

            cluster_labels = list(_shard_groups.keys())

            for i in range(len(cluster_labels) - 1):

                c1 = _shard_groups[cluster_labels[i]]
                c2 = _shard_groups[cluster_labels[i + 1]]

                if c1 and c2:

                    connect(
                        c1[0],
                        c2[0],
                        weight=0.25,
                        frequency=1
                    )

        except Exception:
            pass

        # =========================================================
        # INITIAL ACTIVATION
        # =========================================================

        if all_concepts:

            try:
                _cognition_state.process_tick({
                    all_concepts[0]: 0.95
                })
            except Exception:
                pass

        # =========================================================
        # SAVE
        # =========================================================

        try:
            save_brain_file()
        except Exception:
            pass

        # =========================================================
        # RESPONSE
        # =========================================================

        state = await get_cognition_state()

        return {
            "success": True,
            "loaded_concepts": len(all_concepts),
            "clusters": len(_shard_groups),
            "sample_concepts": all_concepts[:10],
            "state": state
        }

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Stress test failed: {str(e)}"
        )
        
@router.get("/api/cognition/shards")
async def get_shard_groups():
    """Returns current shard groupings (label -> [thought concepts]) for 3D cluster halos."""
    global _shard_groups
    return {"success": True, "shard_groups": _shard_groups if _shard_groups else {}}

@router.post("/api/cognition/query")
async def query_cognition_memory(request: CognitionQueryRequest):
    global _cognition_state
    if _cognition_state is None:
        load_brain_file() or init_cognition_engine()
        save_brain_file()
        
    try:
        q = request.query.strip()
        if not q:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        # Embed the query using the shared model (imported from attractor_dynamics)
        model = get_embedding_model()
        q_vec = safe_normalize(np.array(model.encode(q), dtype=np.float32))
        
        # 1. Compute cosine similarity with all dynamic concept coordinates
        sims = {}
        for concept, vec in _cognition_state.concept_coords.items():
            sims[concept] = float(np.dot(q_vec, vec))
            
        if not sims:
            return {
                "success": True,
                "query": q,
                "query_coords": [0.0, 0.0, 0.0],
                "matches": []
            }

        # 2. Wave Propagation & Field Excitation Search
        # Initialize nodes with energy proportional to cosine similarity
        node_energies = {c: max(0.0, sim) for c, sim in sims.items()}
        
        # Select initial seed concepts (top 3 highest initial energy)
        sorted_seeds = sorted(node_energies.items(), key=lambda x: x[1], reverse=True)
        seed_count = min(3, len(sorted_seeds))
        seeds = [c for c, eng in sorted_seeds[:seed_count]]
        
        import networkx as nx
        G = _cognition_state.causal.causal_matrix
        
        # Accumulator for total energy received by each node during propagation
        accumulated_resonance = {c: 0.0 for c in sims}
        paths = {c: [c] for c in seeds}
        
        # 5 steps of wave propagation
        for step in range(5):
            next_energies = {c: 0.0 for c in sims}
            for u, energy in node_energies.items():
                if energy < 0.05:
                    continue
                
                accumulated_resonance[u] += energy
                
                # Propagate to causal neighbors
                if u in G:
                    edges = G.edges(u, data=True)
                    for _, v, data in edges:
                        if v not in sims:
                            continue
                        weight = data.get("weight", 0.5)
                        # Energy transfer is proportional to causal edge weight and propagation coefficient
                        transfer = energy * weight * 0.45
                        next_energies[v] += transfer
                        
                        # Record path flow
                        if u in paths and (v not in paths or len(paths[v]) > len(paths[u]) + 1):
                            paths[v] = paths[u] + [v]
                            
            # Self-excitation in stable attractors (they sustain energy better)
            for c in sims:
                stability = _cognition_state.physics_tensors.get(c, {}).get("stability", 0.5)
                attractor_resonance = node_energies[c] * stability * 0.15
                next_energies[c] += attractor_resonance
                
            # Apply global decay/leakage to suppress noise
            node_energies = {c: next_energies[c] * 0.75 for c in sims}
            
        # 3. Finalize resonance scores
        results = []
        for concept in sims.keys():
            sim = sims[concept]
            resonance = accumulated_resonance.get(concept, 0.0)
            # Blend direct semantic similarity with propagation resonance
            score = sim * 0.35 + resonance * 0.65
            path = paths.get(concept, [])
            source_seed = path[0] if path else None
            
            results.append({
                "concept": concept,
                "similarity": sim,
                "score": score,
                "path": path,
                "source_seed": source_seed
            })
            
        # Sort by final graph-decayed score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        top_matches = results[:request.top_k]
        
        # Excite the brain state: recall boosts activation/energy and reinforces traversed causal paths
        for match in top_matches:
            concept = match["concept"]
            score = match["score"]
            
            # Boost activation in physics tensors
            if concept in _cognition_state.physics_tensors:
                tensors = _cognition_state.physics_tensors[concept]
                tensors["activation"] = min(1.0, tensors.get("activation", 0.5) + score * 0.20)
                tensors["energy"] = min(1.0, tensors.get("energy", 0.5) + score * 0.15)
                tensors["stability"] = min(1.0, tensors.get("stability", 0.5) + 0.05)
                
            # Boost in working memory if present
            if _cognition_state.memory.graph.has_node(concept):
                _cognition_state.memory.graph.nodes[concept]["activation"] = min(
                    1.0, _cognition_state.memory.graph.nodes[concept]["activation"] + score * 0.20
                )
                
            # Strengthen causal paths traversed during propagation
            path = match["path"]
            if len(path) > 1:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    if _cognition_state.causal.causal_matrix.has_edge(u, v):
                        edge_data = _cognition_state.causal.causal_matrix[u][v]
                        edge_data["weight"] = min(1.0, edge_data.get("weight", 0.5) + 0.06)
                        edge_data["frequency"] = edge_data.get("frequency", 1) + 1
                        
        save_brain_file()
        
        # Project query vector to 3D space using saved PCA parameters
        q_proj_3d = [0.0, 0.0, 0.0]
        pca_mean = getattr(_cognition_state, "_pca_mean", None)
        pca_vectors = getattr(_cognition_state, "_pca_vectors", None)
        pca_scale = getattr(_cognition_state, "_pca_scale", None)
        
        if pca_mean is not None and pca_vectors is not None and pca_scale is not None:
            centered_q = q_vec - pca_mean
            q_p = np.dot(centered_q, pca_vectors)
            q_proj_3d = [
                float(q_p[0]) * pca_scale,
                float(q_p[1]) * pca_scale,
                float(q_p[2]) * pca_scale
            ]
            
        return {
            "success": True,
            "query": q,
            "query_coords": q_proj_3d,
            "matches": top_matches
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.get("/api/cognition/narrative")
async def get_cognition_narrative():
    narrative_stream = [
        {"text": "Massive objects warp the fabric of spacetime", "intensity": 0.9, "dominant_concept": "gravity"},
        {"text": "Gravity is the curvature of the universe", "intensity": 0.9, "dominant_concept": "spacetime"},
        {"text": "Light bends around massive galaxies", "intensity": 0.95, "dominant_concept": "gravitational_lensing"},
        {"text": "Spacetime continuum defines the path of light", "intensity": 0.9, "dominant_concept": "relativity"},
        {"text": "Black holes contain a singularity of infinite density", "intensity": 0.95, "dominant_concept": "singularity"},
        {"text": "Entropy inside a black hole is proportional to its area", "intensity": 0.85, "dominant_concept": "thermodynamics"},
        {"text": "The event horizon is the point of no return", "intensity": 0.9, "dominant_concept": "event_horizon"},
        {"text": "Hawking radiation slowly evaporates black holes", "intensity": 0.8, "dominant_concept": "hawking_radiation"}
    ]
    return {"narrative": narrative_stream}

@router.get("/api/cognition/state")
async def get_cognition_state():
    global _cognition_state, _cognition_pipeline, _cognition_controller
    
    if _cognition_state is None:
        init_cognition_engine()
        save_brain_file()
        
    try:
        # Project concepts to 3D via PCA
        coords, lf_proj, wl_proj = project_concepts_to_3d(_cognition_state.concept_coords)
        
        # Gather active assemblies
        assemblies = []
        for assy in _cognition_state.active_assemblies:
            assemblies.append({
                "id": assy.assembly_id,
                "dominant": assy.dominant_concept,
                "activations": assy.node_activations,
                "confidence": assy.confidence,
                "stability": assy.stability,
                "basin_strength": assy.meta_data.get("basin_strength", 0.0)
            })
            
        # Gather working memory graph
        wm_nodes = []
        wm_edges = []
        for node, data in _cognition_state.memory.graph.nodes(data=True):
            wm_nodes.append({
                "id": node,
                "activation": float(data.get("activation", 0.0)),
                "stability": float(data.get("stability", 0.0)),
                "temporal_depth": int(data.get("temporal_depth", 0))
            })
        for u, v, data in _cognition_state.memory.graph.edges(data=True):
            wm_edges.append({
                "from": u,
                "to": v,
                "weight": float(data.get("weight", 0.0))
            })
            
        # Gather meta shards
        meta_shards = []
        for meta_id, shard in _cognition_state.meta_shards.meta_store.items():
            meta_shards.append({
                "id": meta_id,
                "label": shard.emergent_label,
                "children": shard.children,
                "stability": float(shard.stability),
                "abstraction_level": int(shard.abstraction_level)
            })
            
        # Gather causal paths
        causal_links = []
        for u, v, data in _cognition_state.causal.causal_matrix.edges(data=True):
            causal_links.append({
                "from": u,
                "to": v,
                "weight": float(data.get("weight", 0.0)),
                "frequency": int(data.get("frequency", 1))
            })
            
        return {
            "success": True,
            "tick": _cognition_state.tick,
            "metrics": {
                "entropy": float(_cognition_state.entropy),
                "coherence": float(_cognition_state.coherence),
                "surprise": float(_cognition_state.surprise),
                "energy": float(_cognition_state.energy_budget),
                "pressure": float(_cognition_state.pressure),
                "mode": str(_cognition_state.reflection.cognitive_mode)
            },
            "concepts_3d": coords,
            "latent_field_3d": lf_proj,
            "working_latent_3d": wl_proj,
            "active_assemblies": assemblies,
            "working_memory": {
                "nodes": wm_nodes,
                "edges": wm_edges
            },
            "attention_focus": list(_cognition_state.attention_focus),
            "meta_shards": meta_shards,
            "causal_links": causal_links,
            "shard_groups": _shard_groups,
            "concept_thoughts": getattr(_cognition_state, '_concept_thoughts', {}),
            "physics_tensors": _cognition_state.physics_tensors,
            "pipeline": {
                "status": "idle",
                "reason": None,
                "decision": {}
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/cognition/tick")
async def tick_cognition_state(request: CognitionTickRequest):
    global _cognition_state, _cognition_pipeline, _cognition_controller, _shard_groups

    if _cognition_state is None:
        if not load_brain_file():
            init_cognition_engine()
            save_brain_file()
        
    try:
        # 1. Evolve continuous vector field
        import re
        shards = [s.strip() for s in re.split(r'(?<=[.!?])\s+', request.text) if len(s.strip()) > 5]
        if not shards:
            shards = [request.text]
            
        resonance_input = {}
        for shard in shards:
            resonance_input[shard] = request.intensity
            
        if request.dominant_concept:
            resonance_input[request.dominant_concept] = 0.85
            
        # Store the shard decomposition for click-to-expand in the visualizer
        primary_concept = shards[0] if shards else request.text
        if not hasattr(_cognition_state, '_concept_thoughts'):
            _cognition_state._concept_thoughts = {}
        _cognition_state._concept_thoughts[primary_concept] = shards

        ca = _cognition_state.process_tick(resonance_input)
        
        # 2. Run controller pipeline tick
        input_signal = {
            "text": request.text,
            "dominant_concept": request.dominant_concept or request.text[:30]
        }
        if request.goal:
            input_signal["goal"] = request.goal
        if request.command:
            input_signal["command"] = request.command
            
        pipeline_result = _cognition_pipeline.step_tick(input_signal)
        
        # Save state right after evolution
        save_brain_file()
        
        # 3. Project concepts to 3D via PCA
        coords, lf_proj, wl_proj = project_concepts_to_3d(_cognition_state.concept_coords)
        
        # 4. Gather active assemblies
        assemblies = []
        for assy in _cognition_state.active_assemblies:
            assemblies.append({
                "id": assy.assembly_id,
                "dominant": assy.dominant_concept,
                "activations": assy.node_activations,
                "confidence": assy.confidence,
                "stability": assy.stability,
                "basin_strength": assy.meta_data.get("basin_strength", 0.0)
            })
            
        # 5. Gather working memory graph
        wm_nodes = []
        wm_edges = []
        for node, data in _cognition_state.memory.graph.nodes(data=True):
            wm_nodes.append({
                "id": node,
                "activation": float(data.get("activation", 0.0)),
                "stability": float(data.get("stability", 0.0)),
                "temporal_depth": int(data.get("temporal_depth", 0))
            })
        for u, v, data in _cognition_state.memory.graph.edges(data=True):
            wm_edges.append({
                "from": u,
                "to": v,
                "weight": float(data.get("weight", 0.0))
            })
            
        # 6. Gather meta shards
        meta_shards = []
        for meta_id, shard in _cognition_state.meta_shards.meta_store.items():
            meta_shards.append({
                "id": meta_id,
                "label": shard.emergent_label,
                "children": shard.children,
                "stability": float(shard.stability),
                "abstraction_level": int(shard.abstraction_level)
            })
            
        # 7. Gather causal paths
        causal_links = []
        for u, v, data in _cognition_state.causal.causal_matrix.edges(data=True):
            causal_links.append({
                "from": u,
                "to": v,
                "weight": float(data.get("weight", 0.0)),
                "frequency": int(data.get("frequency", 1))
            })
            
        # After each user tick, re-cluster to absorb the new thought into the right group
        _shard_groups = auto_cluster_shard_groups(
            n_clusters=min(8, max(2, len(_cognition_state.semantic_field.concept_embeddings) // 5))
        )

        return {
            "success": True,
            "tick": _cognition_state.tick,
            "metrics": {
                "entropy": float(_cognition_state.entropy),
                "coherence": float(_cognition_state.coherence),
                "surprise": float(_cognition_state.surprise),
                "energy": float(_cognition_state.energy_budget),
                "pressure": float(_cognition_state.pressure),
                "mode": str(_cognition_state.reflection.cognitive_mode)
            },
            "concepts_3d": coords,
            "latent_field_3d": lf_proj,
            "working_latent_3d": wl_proj,
            "active_assemblies": assemblies,
            "working_memory": {
                "nodes": wm_nodes,
                "edges": wm_edges
            },
            "attention_focus": list(_cognition_state.attention_focus),
            "meta_shards": meta_shards,
            "causal_links": causal_links,
            "shard_groups": _shard_groups,
            "concept_thoughts": getattr(_cognition_state, '_concept_thoughts', {}),
            "physics_tensors": _cognition_state.physics_tensors,
            "pipeline": {
                "status": pipeline_result["status"],
                "reason": pipeline_result.get("reason"),
                "decision": pipeline_result.get("decision", {})
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Finalize App
app.include_router(router)

LIBS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "utils", "webVisualizer", "libs"))
if os.path.exists(LIBS_DIR):
    app.mount("/libs", StaticFiles(directory=LIBS_DIR), name="libs")

if __name__ == "__main__":
    import uvicorn
    print(f"\nLoom Visualizer available at: http://localhost:8000/loom/visualizer")
    uvicorn.run(app, host="0.0.0.0", port=8000)
