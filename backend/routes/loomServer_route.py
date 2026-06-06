import os
import sys
import shutil
import json
import time
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

# Service Imports
from backend.services.loom_service.orchestration.loomServer_Service import LoomServerService
from backend.services.loom_service.cortex.neural_viewer import NeuralViewer
from backend.config.envConfig import setup_logger, log_service

# Modular Imports
from backend.routes.loomServerRoute.state_manager import (
    COGNITION_STATE_DIR,
    get_cognition_state,
    set_cognition_state,
    get_cognition_pipeline,
    get_cognition_controller,
    get_shard_groups,
    set_shard_groups,
    init_cognition_engine,
    load_brain_file,
    save_brain_file,
    project_concepts_to_3d,
    auto_cluster_shard_groups
)
from backend.routes.loomServerRoute.algorithms import (
    query_cognition_memory_algorithm,
    get_serialized_cognition_state_helper,
    inject_single_shard_algorithm,
    run_cognition_stress_test_algorithm
)
from backend.routes.loomServerRoute.document_processor_task import (
    process_pdf_task,
    get_process_status
)

logger = setup_logger("LoomServerRoute")
router = APIRouter(prefix="/loom", tags=["Loom"])

# Paths
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

class CognitionTickRequest(BaseModel):
    text: str
    intensity: float = 0.9
    goal: Optional[str] = None
    command: Optional[str] = None
    dominant_concept: Optional[str] = None

class CognitionQueryRequest(BaseModel):
    query: str
    top_k: int = 5

# --- Routes ---

@router.get("/api/status")
async def get_task_status(job_id: str = Query(...)):
    """Returns the current progress and status of a background job."""
    return get_process_status(job_id)

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
                
            state = get_cognition_state()
            if state is None:
                if not load_brain_file():
                    init_cognition_engine()
                    state = get_cognition_state()
                    
            from backend.routes.loomServerRoute.weaver_helper import RealtimeWeaver
            pages = [{"page_number": 1, "content": [{"type": "paragraph", "text": text}]}]
            real_weaver = RealtimeWeaver()
            shards = real_weaver.weave_into_state(pages, state, save_callback=save_brain_file)
            
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
    path = path.strip("\"'")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Atlas file not found")
        
    is_binary_atlas = False
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic == b"ATLS":
                is_binary_atlas = True
    except Exception:
        pass
        
    if is_binary_atlas:
        try:
            import msgpack
            from fastapi import Response
            from backend.services.loom_service.weaver.atlas_router import GlobalAtlasRouter
            from backend.services.loom_service.weaver.substrate_layout import LoomSubstrate
            import hashlib
            
            router_obj = GlobalAtlasRouter(atlas_path=path)
            
            nodes = {
                "root": {
                    "t": "root",
                    "c": f"Universe Spacetime Core (Seed: {router_obj.seed})",
                    "m": {"seed": router_obj.seed, "truth_score": 1.0}
                }
            }
            edges = []
            shard_to_crystal_map = {}
            concept_bridge = {}
            
            for crystal_path, info in router_obj.crystals.items():
                crystal_name = os.path.basename(crystal_path)
                crystal_node_id = f"crystal_{crystal_name}"
                
                nodes[crystal_node_id] = {
                    "t": "atlas",
                    "c": f"Crystal Region: {crystal_name} ({info['state'].upper()})",
                    "m": {
                        "state": info["state"],
                        "num_shards": info["num_shards"],
                        "centroid": info["centroid"].tolist(),
                        "truth_score": 0.95
                    }
                }
                
                edges.append({
                    "f": "root",
                    "t": crystal_node_id,
                    "r": "contains",
                    "s": 1.0
                })
                
                actual_crystal_path = crystal_path
                if not os.path.isabs(actual_crystal_path):
                    actual_crystal_path = os.path.join(os.path.dirname(path), crystal_name)
                elif not os.path.exists(actual_crystal_path):
                    actual_crystal_path = os.path.join(os.path.dirname(path), crystal_name)
                    
                if os.path.exists(actual_crystal_path):
                    try:
                        journal = LoomSubstrate.get_journal(actual_crystal_path)
                        for shard in journal:
                            sid = shard["shard_id"]
                            shard_to_crystal_map[sid] = crystal_name
                            
                            meta = shard.get("meta", {})
                            concepts = meta.get("concepts", []) + meta.get("entities", [])
                            for concept in concepts:
                                c_hash = hashlib.md5(concept.lower().strip().encode('utf-8')).hexdigest()[:16]
                                if c_hash not in concept_bridge:
                                    concept_bridge[c_hash] = []
                                concept_bridge[c_hash].append({"n": sid, "s": 1.0})
                    except Exception as e:
                        log_service(logger, f"Failed to parse crystal {crystal_name} journal: {e}", "warning")
            
            bundle = {
                "v": "1.3",
                "type": "substrate_master",
                "map": shard_to_crystal_map,
                "bridge": concept_bridge,
                "g": {
                    "n": nodes,
                    "e": edges
                }
            }
            packed = msgpack.packb(bundle, use_bin_type=True)
            return Response(content=packed, media_type="application/x-msgpack")
            
        except Exception as e:
            log_service(logger, f"Failed to serve binary atlas: {e}", "error")
            raise HTTPException(status_code=500, detail=f"Failed to serve binary atlas: {str(e)}")
            
    return FileResponse(path)

@router.get("/api/load_shard")
async def load_shard(master_path: str = Query(...), shard_id: str = Query(...)):
    """Loads a specific binary shard file from the atlas directory."""
    try:
        master_path = master_path.strip("\"'")
        base_dir = os.path.dirname(master_path)
        
        is_binary_atlas = master_path.endswith("atlas.bin")
        if not is_binary_atlas and os.path.exists(master_path):
            try:
                with open(master_path, "rb") as f:
                    if f.read(4) == b"ATLS":
                        is_binary_atlas = True
            except Exception:
                pass
                
        if is_binary_atlas:
            crystal_path = os.path.join(base_dir, shard_id)
            if not os.path.exists(crystal_path):
                raise HTTPException(status_code=404, detail=f"Crystal file {shard_id} not found at {crystal_path}")
                
            import msgpack
            from fastapi import Response
            from backend.services.loom_service.weaver.substrate_layout import LoomSubstrate
            
            journal = LoomSubstrate.get_journal(crystal_path)
            
            shard_nodes = {}
            shard_edges = []
            crystal_node_id = f"crystal_{shard_id}"
            
            for idx, entry in enumerate(journal):
                sid = entry["shard_id"]
                meta_info = entry.get("meta", {})
                
                shard_nodes[sid] = {
                    "t": "shard",
                    "c": meta_info.get("text", ""),
                    "m": {
                        "activation_score": meta_info.get("activation", 1.0),
                        "truth_score": meta_info.get("mass", 1.0) / 10.0,
                        "mass": meta_info.get("mass", 1.0),
                        "hits": meta_info.get("hits", 0),
                        "concepts": meta_info.get("concepts", []),
                        "entities": meta_info.get("entities", []),
                        "page": meta_info.get("page", "GLOBAL")
                    }
                }
                
                shard_edges.append({
                    "f": crystal_node_id,
                    "t": sid,
                    "r": "indexes",
                    "s": 1.0
                })
                
            bundle = {
                "v": "1.3",
                "n": shard_nodes,
                "e": shard_edges
            }
            packed = msgpack.packb(bundle, use_bin_type=True)
            return Response(content=packed, media_type="application/x-msgpack")
            
        # Legacy fallback
        atlas_filename = os.path.basename(master_path)
        base_doc_name = atlas_filename.replace("atlas_", "").replace(".loom", "")
        shard_path = os.path.join(base_dir, f"{base_doc_name}_shard_{shard_id}.loom")
        
        if not os.path.exists(shard_path):
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
        for filename in os.listdir(COGNITION_STATE_DIR):
            file_path = os.path.join(COGNITION_STATE_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
        init_cognition_engine()
        save_brain_file()
        return {"success": True, "status": "Cognitive Substrate Reset & Cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/cognition/stress_shards")
async def get_stress_shards():
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
    try:
        body = await request.json()
        shard = body.get("shard")
        prev_concept = body.get("prev_concept")
        return inject_single_shard_algorithm(shard, prev_concept)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/cognition/stress_test")
async def run_cognition_stress_test():
    try:
        return run_cognition_stress_test_algorithm()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/cognition/shards")
async def get_shard_groups_route():
    shard_groups = get_shard_groups()
    return {"success": True, "shard_groups": shard_groups if shard_groups else {}}

@router.post("/api/cognition/query")
async def query_cognition_memory(request: CognitionQueryRequest):
    try:
        return query_cognition_memory_algorithm(request.query, request.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
async def get_cognition_state_route():
    try:
        return get_serialized_cognition_state_helper()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/cognition/tick")
async def tick_cognition_state(request: CognitionTickRequest):
    state = get_cognition_state()
    shard_groups = get_shard_groups()
    if state is None:
        if not load_brain_file():
            init_cognition_engine()
            state = get_cognition_state()
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
        if not hasattr(state, '_concept_thoughts') or state._concept_thoughts is None:
            state._concept_thoughts = {}
        state._concept_thoughts[primary_concept] = shards

        ca = state.process_tick(resonance_input)
        
        # 2. Run controller pipeline tick
        input_signal = {
            "text": request.text,
            "dominant_concept": request.dominant_concept or request.text[:30]
        }
        if request.goal:
            input_signal["goal"] = request.goal
        if request.command:
            input_signal["command"] = request.command
            
        pipeline = get_cognition_pipeline()
        pipeline_result = pipeline.step_tick(input_signal)
        
        # Save state right after evolution
        save_brain_file()
        
        # 3. Project concepts to 3D via PCA
        coords, lf_proj, wl_proj = project_concepts_to_3d(state.concept_coords)
        
        # 4. Gather active assemblies
        assemblies = []
        for assy in state.active_assemblies:
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
        for node, data in state.memory.graph.nodes(data=True):
            wm_nodes.append({
                "id": node,
                "activation": float(data.get("activation", 0.0)),
                "stability": float(data.get("stability", 0.0)),
                "temporal_depth": int(data.get("temporal_depth", 0))
            })
        for u, v, data in state.memory.graph.edges(data=True):
            wm_edges.append({
                "from": u,
                "to": v,
                "weight": float(data.get("weight", 0.0))
            })
            
        # 6. Gather meta shards
        meta_shards = []
        for meta_id, shard in state.meta_shards.meta_store.items():
            meta_shards.append({
                "id": meta_id,
                "label": shard.emergent_label,
                "children": shard.children,
                "stability": float(shard.stability),
                "abstraction_level": int(shard.abstraction_level)
            })
            
        # 7. Gather causal paths
        causal_links = []
        for u, v, data in state.causal.causal_matrix.edges(data=True):
            causal_links.append({
                "from": u,
                "to": v,
                "weight": float(data.get("weight", 0.0)),
                "frequency": int(data.get("frequency", 1))
            })
            
        # After each user tick, re-cluster to absorb the new thought into the right group
        new_groups = auto_cluster_shard_groups(
            n_clusters=min(8, max(2, len(state.semantic_field.concept_embeddings) // 5))
        )
        set_shard_groups(new_groups)

        return {
            "success": True,
            "tick": state.tick,
            "metrics": {
                "entropy": float(state.entropy),
                "coherence": float(state.coherence),
                "surprise": float(state.surprise),
                "energy": float(state.energy_budget),
                "pressure": float(state.pressure),
                "mode": str(state.reflection.cognitive_mode)
            },
            "concepts_3d": coords,
            "latent_field_3d": lf_proj,
            "working_latent_3d": wl_proj,
            "active_assemblies": assemblies,
            "working_memory": {
                "nodes": wm_nodes,
                "edges": wm_edges
            },
            "attention_focus": list(state.attention_focus),
            "meta_shards": meta_shards,
            "causal_links": causal_links,
            "shard_groups": get_shard_groups(),
            "concept_thoughts": getattr(state, '_concept_thoughts', {}),
            "physics_tensors": state.physics_tensors,
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
