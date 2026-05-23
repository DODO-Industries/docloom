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

        # Serialize causal links
        causal_edges = []
        for u, v, data in _cognition_state.causal.causal_matrix.edges(data=True):
            causal_edges.append({
                "from": u,
                "to": v,
                "weight": float(data.get("weight", 0.0)),
                "frequency": int(data.get("frequency", 1))
            })

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

        state_dict = {
            "tick": _cognition_state.tick,
            "entropy": _cognition_state.entropy,
            "coherence": _cognition_state.coherence,
            "pressure": _cognition_state.pressure,
            "surprise": _cognition_state.surprise,
            "energy_budget": _cognition_state.energy_budget,
            "latent_field": _cognition_state.latent_field.tolist() if _cognition_state.latent_field is not None else None,
            "working_latent": _cognition_state.working_latent.tolist() if _cognition_state.working_latent is not None else None,
            "concept_embeddings": emb_dict,
            "active_assemblies": assemblies,
            "working_memory": {
                "nodes": wm_nodes,
                "edges": wm_edges
            },
            "attention_focus": list(_cognition_state.attention_focus),
            "meta_shards": meta_shards,
            "causal_links": causal_edges,
            "shard_groups": _shard_groups,
            "concept_thoughts": concept_thoughts
        }

        file_path = os.path.join(RESULTS_DIR, "brain_state.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=4)
    except Exception as e:
        print("Error saving brain state file:", e)

def load_brain_file():
    global _cognition_state, _cognition_pipeline, _cognition_controller, _shard_groups
    file_path = os.path.join(RESULTS_DIR, "brain_state.json")
    if not os.path.exists(file_path):
        return False
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            state_dict = json.load(f)

        init_cognition_engine()

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

        # Load embeddings
        _cognition_state.semantic_field.concept_embeddings.clear()
        for k, v in state_dict.get("concept_embeddings", {}).items():
            _cognition_state.semantic_field.concept_embeddings[k] = np.array(v)

        # Load active assemblies
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

        # Load working memory graph
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

        # Load causal graph
        _cognition_state.causal.causal_matrix.clear()
        for l in state_dict.get("causal_links", []):
            _cognition_state.causal.causal_matrix.add_edge(
                l["from"],
                l["to"],
                weight=l["weight"],
                frequency=l["frequency"],
                last_seen=time.time()
            )

        # Load meta shards
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

        # Restore shard groups and per-concept thoughts
        _shard_groups = state_dict.get("shard_groups", {})
        _cognition_state._concept_thoughts = state_dict.get("concept_thoughts", {})
        return True
    except Exception as e:
        print("Failed to load brain state file:", e)
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
        file_path = os.path.join(RESULTS_DIR, "brain_state.json")
        if os.path.exists(file_path):
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
    Auto-derive shard groups from concept embeddings using greedy cosine-similarity clustering.
    No hard-coded labels — clusters are named Cluster_1, Cluster_2, etc.
    New concepts injected by the user are assigned to the nearest existing cluster.
    """
    global _cognition_state
    if _cognition_state is None:
        return {}
    emb_map = _cognition_state.semantic_field.concept_embeddings
    if not emb_map:
        return {}

    concepts = list(emb_map.keys())
    if len(concepts) <= n_clusters:
        # Too few concepts — each is its own cluster
        return {f"Cluster_{i+1}": [c] for i, c in enumerate(concepts)}

    vecs = np.array([emb_map[c] for c in concepts], dtype=np.float32)

    # Simple greedy K-means-like seeding (no sklearn dependency)
    rng = np.random.RandomState(42)
    # Pick n_clusters seeds spread apart using max-distance seeding
    centroids_idx = [rng.randint(len(concepts))]
    for _ in range(n_clusters - 1):
        dists = np.array([
            min(1.0 - float(np.dot(vecs[i], vecs[c_idx]))
                for c_idx in centroids_idx)
            for i in range(len(concepts))
        ])
        next_idx = int(np.argmax(dists))
        centroids_idx.append(next_idx)

    centroids = vecs[centroids_idx]

    # Assign each concept to nearest centroid
    assignments = {}
    for i, (concept, vec) in enumerate(zip(concepts, vecs)):
        sims = centroids @ vec  # dot product = cosine sim since normalized
        nearest = int(np.argmax(sims))
        assignments.setdefault(nearest, []).append(concept)

    return {f"Cluster_{k+1}": v for k, v in assignments.items()}


@router.post("/api/cognition/stress_test")
async def run_cognition_stress_test():
    global _cognition_state, _shard_groups
    try:
        # Re-initialize clean engine with empty state
        init_cognition_engine()
        _cognition_state._concept_thoughts = {}

        # Load test shards — supports generate_test_data.py format: [{"text": ..., "thoughts": [...]}]
        # Also supports simple label format: [{"label": ..., "thoughts": [...]}]
        test_path = os.path.join(RESULTS_DIR, "test_shards_100.json")
        if not os.path.exists(test_path):
            raise HTTPException(status_code=404, detail=f"Test shards file not found at: {test_path}")

        with open(test_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not raw:
            raise HTTPException(status_code=400, detail="Test shards file is empty.")

        # Determine format:
        # Format A (generate_test_data.py): [{"text": concept_title, "thoughts": [...clause strings]}]
        # Format B (label format):          [{"label": cluster_name, "thoughts": [...full sentences]}]
        first = raw[0]
        is_format_a = "text" in first and "thoughts" in first
        is_format_b = "label" in first and "thoughts" in first

        all_concept_names = []  # ordered list of concept node names

        if is_format_a:
            # Each entry: concept sphere = entry["text"], sub-thoughts = entry["thoughts"]
            # Register the concept title as the node; store thoughts for click-expand
            for entry in raw:
                concept = entry["text"].strip()
                thoughts = [t.strip() for t in entry.get("thoughts", []) if t.strip()]
                if not concept:
                    continue
                _cognition_state.semantic_field.register_concept(concept)
                _cognition_state._concept_thoughts[concept] = thoughts
                _cognition_state.memory.graph.add_node(
                    concept, activation=0.9, stability=0.9, temporal_depth=0
                )
                all_concept_names.append(concept)

            # Causal chain across all concepts
            for i in range(len(all_concept_names) - 1):
                _cognition_state.causal.causal_matrix.add_edge(
                    all_concept_names[i], all_concept_names[i + 1],
                    weight=0.6, frequency=3, last_seen=time.time()
                )

            # Auto-cluster after all embeddings are registered
            _shard_groups = auto_cluster_shard_groups(n_clusters=min(8, max(2, len(all_concept_names) // 5)))

        elif is_format_b:
            # Each entry is a pre-labelled cluster of thoughts
            for entry in raw:
                # Do NOT store the label itself as a concept — only store the thoughts
                thoughts = [t.strip() for t in entry.get("thoughts", []) if t.strip()]
                for thought in thoughts:
                    _cognition_state.semantic_field.register_concept(thought)
                    _cognition_state.memory.graph.add_node(
                        thought, activation=0.9, stability=0.9, temporal_depth=0
                    )
                    all_concept_names.append(thought)
                # Causal chain within the cluster
                for i in range(len(thoughts) - 1):
                    _cognition_state.causal.causal_matrix.add_edge(
                        thoughts[i], thoughts[i + 1],
                        weight=0.85, frequency=5, last_seen=time.time()
                    )

            # Auto-cluster: ignore the explicit labels, discover clusters from embeddings
            _shard_groups = auto_cluster_shard_groups(n_clusters=min(8, len(raw)))

        else:
            # Old flat format: [{"text": sentence}]
            for s in raw:
                txt = s.get("text", "").strip()
                if not txt:
                    continue
                _cognition_state.semantic_field.register_concept(txt)
                _cognition_state.memory.graph.add_node(
                    txt, activation=0.9, stability=0.9, temporal_depth=0
                )
                all_concept_names.append(txt)
            _shard_groups = auto_cluster_shard_groups(n_clusters=min(8, max(2, len(all_concept_names) // 12)))

        # Sparse inter-cluster links
        cluster_labels = list(_shard_groups.keys())
        for i in range(len(cluster_labels) - 1):
            c1 = _shard_groups[cluster_labels[i]]
            c2 = _shard_groups[cluster_labels[i + 1]]
            if c1 and c2:
                _cognition_state.causal.causal_matrix.add_edge(
                    c1[0], c2[0], weight=0.25, frequency=1, last_seen=time.time()
                )

        # Init cognitive field with the first concept
        if all_concept_names:
            _cognition_state.process_tick({all_concept_names[0]: 0.9})

        save_brain_file()

        state = await get_cognition_state()
        return state

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Stress test failed: {str(e)}")

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
        
        # Compute cosine similarity with all registered concept embeddings
        results = []
        for concept, vec in _cognition_state.semantic_field.concept_embeddings.items():
            sim = float(np.dot(q_vec, vec))
            results.append({
                "concept": concept,
                "similarity": sim
            })
            
        # Sort by similarity descending
        results.sort(key=lambda x: x["similarity"], reverse=True)
        top_matches = results[:request.top_k]
        
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
        coords, lf_proj, wl_proj = project_concepts_to_3d(_cognition_state.semantic_field.concept_embeddings)
        
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
        coords, lf_proj, wl_proj = project_concepts_to_3d(_cognition_state.semantic_field.concept_embeddings)
        
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
