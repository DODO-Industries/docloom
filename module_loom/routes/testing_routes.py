import os
import sys
import time
import math
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from module_loom.config.env_config import (
    setup_logger, log_service, BRAIN_STORAGE_DIR,
    RAW_EMBEDDING_DIMENSION, EMBEDDING_DIMENSION
)
from module_loom.services.embedding.embedding_manager import get_embedding_model
from module_loom.services.weaver.atlas_router import GlobalAtlasRouter

logger = setup_logger("TestingVisualizer")
router = APIRouter(prefix="/loom/testing", tags=["TestingVisualizer"])

TESTING_HTML_PATH = os.path.join(BASE_DIR, "..", "test", "testing_viz.html")

# In-memory test session state
_coordinator = None
_session_shards: Dict[str, Dict[str, Any]] = {}
_session_vectors: Dict[str, np.ndarray] = {}        # 128-D aligned (for coordinator)
_session_raw_vectors: Dict[str, np.ndarray] = {}     # 384-D raw (for 3D projection)
_projection_matrix_384: Optional[np.ndarray] = None  # Fixed projection for raw 384-D


def _brain_dir() -> str:
    if os.path.isabs(BRAIN_STORAGE_DIR) and os.path.isdir(BRAIN_STORAGE_DIR):
        return BRAIN_STORAGE_DIR
    for c in (os.path.join(PROJECT_ROOT, BRAIN_STORAGE_DIR), os.path.join(PROJECT_ROOT, "assets", ".brain_data"), os.path.join(PROJECT_ROOT, ".brain_data")):
        if os.path.isdir(c):
            return c
    return os.path.join(PROJECT_ROOT, BRAIN_STORAGE_DIR)


def _get_coordinator():
    global _coordinator
    if _coordinator is None:
        from module_loom.services.weaver.weaver_coordinator import WeaveBrainCoordinator
        from module_loom.config.tuning_config import tuning_manager
        brain_dir = _brain_dir()
        _coordinator = WeaveBrainCoordinator(storage_dir=brain_dir)
        split_size = tuning_manager.get_int("TESTING_CRYSTAL_SPLIT_SIZE", 80)
        if split_size > 0:
            _coordinator.max_shards_per_crystal = split_size
        log_service(logger, f"TestingSandbox bound to brain at {brain_dir} (split_cap={_coordinator.max_shards_per_crystal})", "info")
    return _coordinator


def _align_vector(vec: np.ndarray, target_dim: int) -> np.ndarray:
    """Safely aligns vector to target_dim (e.g. 128) matching coordinator dimension."""
    v = np.asarray(vec, dtype=np.float32)
    if len(v) > target_dim:
        v = v[:target_dim]
    elif len(v) < target_dim:
        v = np.pad(v, (0, target_dim - len(v)), constant_values=0.0)
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    return v


def _get_crystal_anchor(coord, crystal_path: Optional[str]) -> np.ndarray:
    """Gets 3D galaxy anchor dynamically from crystal centroid."""
    if coord and hasattr(coord, "atlas") and crystal_path:
        return coord.atlas.get_crystal_anchor_3d(crystal_path)
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)


def _get_crystal_style(crystal_path: str) -> Tuple[str, str]:
    """Generates dynamic display label and color from crystal path/name."""
    name = os.path.basename(crystal_path).replace(".loom", "")
    h = abs(int(hashlib.md5(name.encode("utf-8")).hexdigest()[:6], 16))
    hue = (h % 360) / 360.0
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return name.capitalize(), hex_color


def _evaluate_hybrid_routing(raw_vec: np.ndarray, model=None) -> dict:
    """Evaluates semantic gravity against crystal centroids using AtlasRouter."""
    coord = _get_coordinator()
    aligned = _align_vector(raw_vec, coord.dimension) if coord else raw_vec
    if coord and hasattr(coord, "atlas") and coord.atlas.crystals:
        emb = coord.atlas.route_vector_with_embassy(aligned)
        p_name = os.path.basename(emb["primary_crystal"]).replace(".loom", "").replace("_", " ").title() if emb.get("primary_crystal") else "Crystal 1"
        s_name = os.path.basename(emb["secondary_crystal"]).replace(".loom", "").replace("_", " ").title() if emb.get("secondary_crystal") else None
        return {
            "primary_domain": p_name, "primary_crystal": emb["primary_crystal"],
            "primary_similarity": emb["primary_sim"], "secondary_domain": s_name,
            "secondary_crystal": emb["secondary_crystal"], "secondary_similarity": emb["secondary_sim"],
            "is_embassy": emb["is_embassy"],
        }
    return {
        "primary_domain": "Crystal 1", "primary_crystal": "crystal_1.loom",
        "primary_similarity": 1.0, "secondary_domain": None, "secondary_crystal": None,
        "secondary_similarity": 0.0, "is_embassy": False,
    }


def _project_to_3d(raw_vector: np.ndarray, crystal_path: Optional[str] = None, hybrid_info: Optional[dict] = None, topic: Optional[str] = None) -> List[float]:
    """
    Projects vector into 3D space with Emergent Crystal Anchoring and
    Spatial Overlap Protection (Bubble clearance physics).
    Zero hardcoded domains or coordinates!
    """
    global _projection_matrix_384
    dim = len(raw_vector)
    coord = _get_coordinator()
    seed = coord.seed if coord else 42

    if _projection_matrix_384 is None or _projection_matrix_384.shape[0] != dim:
        rng = np.random.RandomState(int(seed % 2147483647))
        P = rng.randn(dim, 3)
        q, _ = np.linalg.qr(P)
        _projection_matrix_384 = q.astype(np.float32)

    vec = np.asarray(raw_vector, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    from module_loom.config.tuning_config import tuning_manager
    cluster_radius = float(tuning_manager.get_float("SPATIAL_CLUSTER_RADIUS", 38.0))
    # Local micro-offset within galaxy (expanded radius ~38 units)
    local_p3 = (vec @ _projection_matrix_384) * cluster_radius

    # Determine Dynamic Anchor Center
    if hybrid_info and hybrid_info.get("is_embassy") and hybrid_info.get("secondary_crystal"):
        p1 = hybrid_info["primary_crystal"]
        p2 = hybrid_info["secondary_crystal"]
        a1 = _get_crystal_anchor(coord, p1)
        a2 = _get_crystal_anchor(coord, p2)
        anchor = 0.5 * a1 + 0.5 * a2
    elif crystal_path:
        anchor = _get_crystal_anchor(coord, crystal_path)
    elif hybrid_info and hybrid_info.get("primary_crystal"):
        anchor = _get_crystal_anchor(coord, hybrid_info["primary_crystal"])
    else:
        # Fallback: project raw vector directly to global macro coordinates
        anchor = (vec @ _projection_matrix_384) * cluster_radius

    pos = (anchor + local_p3).copy()

    # Bubble Overlap Protection: prevent clipping with existing session nodes
    clearance_radius = float(tuning_manager.get_float("SPATIAL_CLEARANCE_RADIUS", 2.0))
    clearance_sq = clearance_radius * clearance_radius
    for sid, rec in _session_shards.items():
        other_coords = rec.get("coords")
        if other_coords and len(other_coords) == 3:
            ox, oy, oz = other_coords
            dist_sq = (pos[0]-ox)**2 + (pos[1]-oy)**2 + (pos[2]-oz)**2
            if dist_sq < clearance_sq:
                dist = math.sqrt(dist_sq) if dist_sq > 0.0001 else 0.0001
                nudge = (clearance_radius * 1.1 - dist) * 0.5
                pos[0] += ((pos[0] - ox) / dist) * nudge
                pos[1] += ((pos[1] - oy) / dist) * nudge
                pos[2] += ((pos[2] - oz) / dist) * nudge

    return [round(float(pos[0]), 2), round(float(pos[1]), 2), round(float(pos[2]), 2)]


def _compute_shard_physics(vec: np.ndarray, text: str) -> dict:
    """Computes mass, resting base energy, and Kuramoto phase."""
    words = len(text.split())
    mass = max(0.5, float(math.log(words + 1) * 3.5))
    energy = max(0.04, float(np.mean(np.abs(vec))))  # resting potential > 0
    half = len(vec) // 2
    phase_rad = float(
        (np.arctan2(np.sum(vec[:half]), np.sum(vec[half:])) + 2 * math.pi) % (2 * math.pi)
    )
    return {
        "mass": round(mass, 2),
        "energy": round(energy, 4),
        "phase_rad": round(phase_rad, 3),
        "phase_deg": round(math.degrees(phase_rad), 1),
    }


# ---------- Pydantic models ----------
class IngestRequest(BaseModel):
    text: str
    custom_mass: Optional[float] = None
    topic: Optional[str] = None
    topic_label: Optional[str] = None
    color: Optional[str] = None

class BatchDatasetRequest(BaseModel):
    per_topic: int = 20

class ChunkIngestRequest(BaseModel):
    offset: int = 0
    count: int = 8
    total_per_topic: int = 20

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

class TextBatchIngestRequest(BaseModel):
    texts: List[str]

class DeleteShardRequest(BaseModel):
    shard_id: str


# ---------- Routes ----------

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def serve_testing_ui():
    """Serves the Testing Sandbox visualizer UI."""
    if not os.path.exists(TESTING_HTML_PATH):
        raise HTTPException(status_code=404, detail="testing_viz.html not found")
    with open(TESTING_HTML_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@router.post("/ingest")
def ingest_testing_shard(req: IngestRequest):
    """Ingests text into DocLoom, timing every micro-step and returning 3D coords & attributes."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    coord = _get_coordinator()
    t_start = time.perf_counter()

    # Step 1: Embedding & Physics
    t0 = time.perf_counter()
    model = get_embedding_model()
    raw_vec = np.asarray(model.encode(text), dtype=np.float32)
    aligned_vec = _align_vector(raw_vec, coord.dimension)
    phys = _compute_shard_physics(aligned_vec, text)
    mass = req.custom_mass if req.custom_mass is not None else phys["mass"]
    t_step1 = (time.perf_counter() - t0) * 1000

    # Step 2: Seed Scaffold & Momentum
    t0 = time.perf_counter()
    shard_id = f"shard_{hashlib.blake2b(text.encode('utf-8'), digest_size=5).hexdigest()}"
    coord.physics.calculate_momentum_vector(aligned_vec, coord.seed_core.get_micro_socket(shard_id), mass)
    t_step2 = (time.perf_counter() - t0) * 1000

    # Step 3: Atlas Routing
    t0 = time.perf_counter()
    target_crystal = coord._route_for_ingest(aligned_vec)
    t_step3 = (time.perf_counter() - t0) * 1000

    # Step 4: LSH Bucket
    t0 = time.perf_counter()
    bucket_id = coord._resolve_bucket(coord._get_store(target_crystal), aligned_vec)
    t_step4 = (time.perf_counter() - t0) * 1000

    # Step 5: Substrate Append
    t0 = time.perf_counter()
    coord.ingest_shard(shard_id, aligned_vec, text, mass=mass)
    t_step5 = (time.perf_counter() - t0) * 1000

    t_total = (time.perf_counter() - t_start) * 1000
    hybrid = _evaluate_hybrid_routing(raw_vec, model)
    coords_3d = _project_to_3d(raw_vec, crystal_path=target_crystal, hybrid_info=hybrid)

    dyn_label, dyn_color = _get_crystal_style(target_crystal)
    crystal_base = os.path.basename(target_crystal)

    record = {
        "shard_id": shard_id, "text": text, "coords": coords_3d,
        **phys, "mass": round(mass, 2), "activation": 1.0,
        "bucket_id": bucket_id,
        "crystal": crystal_base,
        "crystal_label": dyn_label,
        "topic": crystal_base,
        "topic_label": dyn_label,
        "color": dyn_color, "hits": 0, "timestamp": time.time(),
        "is_embassy": hybrid.get("is_embassy", False),
        "secondary_crystal": hybrid.get("secondary_crystal"),
        "secondary_topic": hybrid.get("secondary_domain"),
        "timings": {
            "total_ms": round(t_total, 2),
            "step1_embed_ms": round(t_step1, 2), "step2_warp_ms": round(t_step2, 2),
            "step3_route_ms": round(t_step3, 2), "step4_lsh_ms": round(t_step4, 2),
            "step5_append_ms": round(t_step5, 2),
        }
    }
    _session_shards[shard_id] = record
    _session_vectors[shard_id] = aligned_vec
    _session_raw_vectors[shard_id] = raw_vec
    return record


@router.post("/ingest_text_batch")
def ingest_text_batch(req: TextBatchIngestRequest):
    """
    Feeds arbitrary real text (e.g. a downloaded corpus) into THIS server's own
    testing coordinator/session — so an external bulk-ingest client can stream
    real data while watching it plot live in this same page, with exactly one
    writer (this process) touching the brain on disk. Each item is routed
    individually via coord.ingest_shard (correct per-item routing for diverse,
    multi-topic text — unlike /ingest_chunk's single-topic canned dataset,
    which routes a whole chunk off one item's vector).
    """
    texts = [t.strip() for t in req.texts if t and t.strip()]
    if not texts:
        return {"status": "empty", "count": 0}

    coord = _get_coordinator()
    model = get_embedding_model()
    raw_vecs = model.encode(texts, batch_size=len(texts))

    added = []
    for text, raw_vec in zip(texts, raw_vecs):
        raw_vec = np.asarray(raw_vec, dtype=np.float32)
        vec = _align_vector(raw_vec, coord.dimension)
        phys = _compute_shard_physics(vec, text)
        shard_id = f"shard_{hashlib.blake2b(text.encode('utf-8'), digest_size=5).hexdigest()}"

        hybrid = _evaluate_hybrid_routing(raw_vec, model)
        target_crystal = coord.ingest_shard(shard_id, vec, text, mass=phys["mass"])
        coords_3d = _project_to_3d(raw_vec, crystal_path=target_crystal, hybrid_info=hybrid)
        dyn_label, dyn_color = _get_crystal_style(target_crystal)
        cbase = os.path.basename(target_crystal)

        rec = {
            "shard_id": shard_id, "text": text, "coords": coords_3d,
            **phys, "activation": 1.0, "bucket_id": 0, "crystal": cbase,
            "crystal_label": dyn_label, "topic": cbase, "topic_label": dyn_label,
            "color": dyn_color, "hits": 0, "timestamp": time.time(),
            "is_embassy": hybrid.get("is_embassy", False),
        }
        _session_shards[shard_id] = rec
        _session_vectors[shard_id] = vec
        _session_raw_vectors[shard_id] = raw_vec
        added.append(shard_id)

    return {"status": "ok", "count": len(added), "total_shards": len(_session_shards)}


@router.post("/ingest_chunk")
def ingest_dataset_chunk(req: ChunkIngestRequest):
    """Ingests a small chunk (e.g. 8 shards) for live streaming into 3D space."""
    from module_loom.services.testing_dataset import generate_5_topic_dataset

    all_sentences = generate_5_topic_dataset(sentences_per_topic=req.total_per_topic)
    total = len(all_sentences)
    chunk = all_sentences[req.offset : req.offset + req.count]
    if not chunk:
        return {"status": "complete", "offset": req.offset, "count": 0, "total": total, "shards": []}

    coord = _get_coordinator()
    model = get_embedding_model()
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    texts = [item["text"] for item in chunk]
    raw_vecs = model.encode(texts, batch_size=len(texts))
    t_embed = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    coord_items, chunk_meta = [], []
    for item, raw_vec in zip(chunk, raw_vecs):
        raw_vec = np.asarray(raw_vec, dtype=np.float32)
        vec = _align_vector(raw_vec, coord.dimension)
        text = item["text"]
        shard_id = f"shard_{hashlib.blake2b(text.encode('utf-8'), digest_size=5).hexdigest()}"
        phys = _compute_shard_physics(vec, text)
        coord_items.append({
            "shard_id": shard_id, "vector": vec, "text": text, "mass": phys["mass"],
            "metadata": {"topic": item.get("topic", "custom"), "topic_label": item.get("topic_label", "Shard")}
        })
        chunk_meta.append((shard_id, text, vec, raw_vec, phys, item))
    t_warp = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    target_crystal = coord._route_for_ingest(coord_items[0]["vector"]) if coord_items else "crystal_1.loom"
    t_route = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    store = coord._get_store(target_crystal)
    coord._resolve_bucket(store, coord_items[0]["vector"]) if coord_items else 0
    t_lsh = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    try:
        coord.ingest_batch(coord_items)
    except Exception as e:
        log_service(logger, f"Chunk batch append note: {e}", "info")
    t_append = (time.perf_counter() - t0) * 1000

    chunk_shards = []
    for sid, text, vec, raw_vec, phys, item in chunk_meta:
        entry = coord.ram_ledger.get(sid, {})
        actual_c = entry.get("crystal_path") or target_crystal
        coords_3d = _project_to_3d(raw_vec, crystal_path=actual_c)
        dyn_label, dyn_color = _get_crystal_style(actual_c)
        cbase = os.path.basename(actual_c)
        rec = {
            "shard_id": sid, "text": text, "coords": coords_3d,
            **phys, "activation": 1.0, "bucket_id": 0, "crystal": cbase,
            "crystal_label": dyn_label, "topic": cbase, "topic_label": dyn_label,
            "color": dyn_color, "timestamp": time.time(),
        }
        _session_shards[sid] = rec
        _session_vectors[sid] = vec
        _session_raw_vectors[sid] = raw_vec
        chunk_shards.append(rec)

    t_elapsed = (time.perf_counter() - t_start) * 1000
    next_off = req.offset + len(chunk)
    n = max(1, len(chunk_shards))
    return {
        "status": "complete" if next_off >= total else "in_progress",
        "offset": req.offset, "count": len(chunk_shards), "total": total,
        "next_offset": next_off,
        "elapsed_ms": round(t_elapsed, 2),
        "avg_ms": round(t_elapsed / n, 2),
        "shards": chunk_shards,
        "timings": {
            "step1_embed_ms": round(t_embed / n, 2),
            "step2_warp_ms": round(t_warp / n, 2),
            "step3_route_ms": round(t_route / n, 2),
            "step4_lsh_ms": round(t_lsh / n, 2),
            "step5_append_ms": round(t_append / n, 2),
            "total_ms": round(t_elapsed / n, 2),
        }
    }


@router.post("/ingest_dataset")
def ingest_batch_dataset(req: BatchDatasetRequest):
    """Ingests bulk sentences across 5 distinct domains."""
    from module_loom.services.testing_dataset import generate_5_topic_dataset

    per_topic = max(5, min(100, req.per_topic))
    dataset = generate_5_topic_dataset(sentences_per_topic=per_topic)
    coord = _get_coordinator()
    model = get_embedding_model()
    t_start = time.perf_counter()

    texts = [item["text"] for item in dataset]
    raw_vecs = model.encode(texts, batch_size=64)

    coord_items, ds_meta = [], []
    for item, raw_vec in zip(dataset, raw_vecs):
        raw_vec = np.asarray(raw_vec, dtype=np.float32)
        vec = _align_vector(raw_vec, coord.dimension)
        text = item["text"]
        shard_id = f"shard_{hashlib.blake2b(text.encode('utf-8'), digest_size=5).hexdigest()}"
        phys = _compute_shard_physics(vec, text)
        coord_items.append({
            "shard_id": shard_id, "vector": vec, "text": text, "mass": phys["mass"],
            "metadata": {"topic": item.get("topic", "custom"), "topic_label": item.get("topic_label", "Shard")}
        })
        ds_meta.append((shard_id, text, vec, raw_vec, phys, item))

    try:
        coord.ingest_batch(coord_items)
    except Exception as e:
        log_service(logger, f"Dataset batch append note: {e}", "info")

    added_shards = []
    for sid, text, vec, raw_vec, phys, item in ds_meta:
        entry = coord.ram_ledger.get(sid, {})
        actual_c = entry.get("crystal_path") or (coord.atlas.route_vector(vec) if coord.atlas.crystals else "crystal_1.loom")
        coords_3d = _project_to_3d(raw_vec, crystal_path=actual_c)
        dyn_label, dyn_color = _get_crystal_style(actual_c)
        cbase = os.path.basename(actual_c)
        rec = {
            "shard_id": sid, "text": text, "coords": coords_3d,
            **phys, "activation": 1.0, "bucket_id": 0, "crystal": cbase,
            "crystal_label": dyn_label, "topic": cbase, "topic_label": dyn_label,
            "color": dyn_color, "timestamp": time.time(),
        }
        _session_shards[sid] = rec
        _session_vectors[sid] = vec
        _session_raw_vectors[sid] = raw_vec
        added_shards.append(rec)

    total_elapsed = (time.perf_counter() - t_start) * 1000
    return {
        "status": "success", "count": len(added_shards),
        "total_shards": len(_session_shards),
        "elapsed_ms": round(total_elapsed, 2),
        "avg_ms_per_shard": round(total_elapsed / max(1, len(added_shards)), 2),
        "shards": added_shards,
    }


@router.post("/query")
def query_testing_shards(req: QueryRequest):
    """
    Full recall with energy disturbance: computes similarity for ALL session shards,
    returns activated shards, ghost shards from substrate, query probe coords,
    and energy line endpoints for 3D visualization.
    """
    q_text = req.query.strip()
    if not q_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    coord = _get_coordinator()
    t_start = time.perf_counter()

    model = get_embedding_model()
    t0 = time.perf_counter()
    raw_q_vec = np.asarray(model.encode(q_text), dtype=np.float32)
    q_vec = _align_vector(raw_q_vec, coord.dimension)
    q_coords = _project_to_3d(raw_q_vec)
    t_embed = (time.perf_counter() - t0) * 1000

    # Score ALL canvas shards — the recall energy disturbs the entire field
    t0 = time.perf_counter()
    all_activated = []
    for sid, vec in _session_vectors.items():
        sim = float(np.dot(q_vec, vec))
        rec = _session_shards.get(sid)
        if rec:
            all_activated.append({
                "shard_id": sid, "text": rec["text"], "coords": rec["coords"],
                "similarity": round(sim, 4), "similarity_pct": round(max(0.0, sim) * 100, 1),
                "activation_level": round(min(1.0, max(0.0, sim * 2.5)), 3),
                "crystal": rec.get("crystal", "crystal_1.loom"),
                "crystal_label": rec.get("crystal_label", "Crystal 1"),
                "topic": rec.get("topic", "crystal_1.loom"),
                "topic_label": rec.get("crystal_label", "Crystal 1"),
                "color": rec.get("color"),
            })

    all_activated.sort(key=lambda x: x["similarity"], reverse=True)
    top_matches = all_activated[:req.top_k]

    from module_loom.config.tuning_config import tuning_manager
    ltp_boost = float(tuning_manager.get_float("LTP_ENERGY_BOOST", 0.015))
    for m in top_matches:
        sid = m["shard_id"]
        if sid in _session_shards:
            rec = _session_shards[sid]
            rec["hits"] = rec.get("hits", 0) + 1
            rec["energy"] = round(min(0.95, rec.get("energy", 0.05) + ltp_boost), 4)
            m["hits"], m["energy"], m["is_embassy"] = rec["hits"], rec["energy"], rec.get("is_embassy", False)
    t_dot = (time.perf_counter() - t0) * 1000

    # Ghost shards from substrate (disk crystals) NOT in session
    t0 = time.perf_counter()
    substrate_ghosts = []
    try:
        results = coord.recall(q_vec, top_k=req.top_k, learn=True)
        for r in (results or []):
            sid = r.get("shard_id", "")
            if sid and sid not in _session_shards:
                ghost_vec = r.get("vector")
                if ghost_vec is not None:
                    ghost_coords = _project_to_3d(np.asarray(ghost_vec, dtype=np.float32))
                else:
                    ghost_coords = q_coords
                substrate_ghosts.append({
                    "shard_id": sid, "text": r.get("text", sid),
                    "coords": ghost_coords,
                    "similarity_pct": round(r.get("similarity", 0) * 100, 1),
                    "source": "dead_memory",
                })
    except Exception as e:
        log_service(logger, f"Substrate ghost recall note: {e}", "debug")
    t_ghost = (time.perf_counter() - t0) * 1000

    # Energy lines: from query probe to each top-K match
    t0 = time.perf_counter()
    energy_lines = [
        {"from": q_coords, "to": m["coords"], "similarity": m["similarity_pct"]}
        for m in top_matches if m["similarity_pct"] > 10
    ]
    t_field = (time.perf_counter() - t0) * 1000

    t_total = (time.perf_counter() - t_start) * 1000

    return {
        "query": q_text, "query_coords": q_coords,
        "latency_ms": round(t_total, 2),
        "top_match": top_matches[0] if top_matches else None,
        "top_matches": top_matches,
        "all_activated": all_activated,
        "energy_lines": energy_lines,
        "substrate_ghosts": substrate_ghosts,
        "total_activated": len([a for a in all_activated if a["similarity_pct"] > 15]),
        "timings": {
            "step1_embed_ms": round(t_embed, 2),
            "step2_dot_ms": round(t_dot, 2),
            "step3_ghost_ms": round(t_ghost, 2),
            "step4_field_ms": round(t_field, 2),
            "total_ms": round(t_total, 2),
        }
    }


def _restore_from_disk(coord):
    """Restores session cache from disk crystals with dynamic crystal anchors."""
    for cpath in coord.atlas.crystals:
        try:
            store = coord._get_store(cpath)
            dyn_label, dyn_color = _get_crystal_style(cpath)
            cbase = os.path.basename(cpath)
            for idx in range(store.n):
                sid = store.get_id(idx)
                vec = store.get_vector(idx)
                meta = store.get_meta(idx) or {}
                text = str(meta.get("text", sid))
                coords_3d = _project_to_3d(vec, crystal_path=cpath)
                phys = _compute_shard_physics(vec, text)
                rec = {
                    "shard_id": sid, "text": text, "coords": coords_3d,
                    **phys, "activation": 1.0, "bucket_id": 0, "crystal": cbase,
                    "crystal_label": dyn_label, "topic": cbase, "topic_label": dyn_label,
                    "color": dyn_color, "timestamp": time.time(),
                }
                _session_shards[sid] = rec
                _session_vectors[sid] = vec
                _session_raw_vectors[sid] = vec
        except Exception as e:
            log_service(logger, f"Disk restore skipped for {cpath}: {e}", "warning")


@router.get("/atlas")
def get_macro_atlas():
    """Returns lightweight macro universe atlas of all crystals and anchors (O(K), ~1 KB)."""
    coord = _get_coordinator()
    crystals = []
    total = 0
    for cpath, info in coord.atlas.crystals.items():
        anchor = coord.atlas.get_crystal_anchor_3d(cpath).tolist()
        dyn_label, dyn_color = _get_crystal_style(cpath)
        try:
            n = coord._get_store(cpath).n if os.path.exists(cpath) else int(info.get("num_shards", 0))
        except Exception:
            n = int(info.get("num_shards", 0))
        total += n
        crystals.append({
            "name": os.path.basename(cpath), "label": dyn_label, "path": cpath,
            "state": info.get("state", "warm"), "num_shards": n,
            "anchor_3d": [round(float(v), 2) for v in anchor],
            "cohesion": round(GlobalAtlasRouter.avg_cohesion(info), 3),
            "color": dyn_color,
        })
    return {"crystals": crystals, "total_shards": total, "total_crystals": len(crystals), "dimension": coord.dimension}


@router.get("/shards")
def get_session_shards(crystal: Optional[str] = None, limit: int = 150, offset: int = 0, lod: str = "overview"):
    """Progressive LOD shard streaming. lod='overview' returns stratified landmark nodes."""
    coord = _get_coordinator()
    if not _session_shards:
        _restore_from_disk(coord)
    items = list(_session_shards.values())
    if crystal:
        cbase = os.path.basename(crystal)
        items = [s for s in items if s.get("crystal") == cbase]
    if lod == "overview":
        by_c = {}
        for s in items:
            by_c.setdefault(s.get("crystal", "default"), []).append(s)
        res = []
        cap = max(10, limit // max(1, len(by_c)))
        for c, clist in by_c.items():
            if len(clist) <= cap:
                res.extend(clist)
            else:
                third = max(1, cap // 3)
                recent = clist[-third:]
                by_m = sorted(clist[:-third], key=lambda x: x.get("mass", 1.0), reverse=True)[:third]
                chosen = {s["shard_id"] for s in recent + by_m}
                rem = [s for s in clist if s["shard_id"] not in chosen]
                step = max(1, len(rem) // max(1, cap - len(chosen)))
                res.extend(recent + by_m + rem[::step][:cap - len(chosen)])
        return res
    return items[offset:offset + limit]


@router.post("/clear")
def clear_session_space():
    """Clears the session space visualizer nodes."""
    _session_shards.clear()
    _session_vectors.clear()
    _session_raw_vectors.clear()
    return {"status": "cleared", "remaining": 0}


@router.post("/delete_shard")
def delete_shard_endpoint(req: DeleteShardRequest):
    """Deletes a single shard from memory, RAM ledger, and appends tombstone to disk."""
    sid = req.shard_id.strip()
    if not sid:
        raise HTTPException(status_code=400, detail="Shard ID cannot be empty")
    for d in (_session_shards, _session_vectors, _session_raw_vectors):
        d.pop(sid, None)
    coord = _get_coordinator()
    coord.ram_ledger.pop(sid, None)
    try:
        import json
        with open(os.path.join(_brain_dir(), "tombstones.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"shard_id": sid, "deleted_at": time.time()}) + "\n")
    except Exception as e:
        log_service(logger, f"Failed to append tombstone: {e}", "warning")
    return {"status": "deleted", "shard_id": sid, "remaining": len(_session_shards)}


@router.post("/purge_disk")
def purge_disk_storage():
    """Permanently purges all .loom memory crystals and resets the brain on disk."""
    global _coordinator
    import glob
    for d in (_session_shards, _session_vectors, _session_raw_vectors):
        d.clear()
    if _coordinator is not None:
        try:
            _coordinator.close()
        except Exception:
            pass
        _coordinator = None
    brain_dir = _brain_dir()
    removed_files = []
    if os.path.exists(brain_dir):
        for pat in ["*.loom", "*.idx", "tombstones.jsonl", "living_state.bin"]:
            for f in glob.glob(os.path.join(brain_dir, pat)):
                try:
                    os.remove(f)
                    removed_files.append(os.path.basename(f))
                except Exception as e:
                    log_service(logger, f"Could not remove {f}: {e}", "warning")
    try:
        _get_coordinator()
    except Exception as e:
        log_service(logger, f"Coordinator re-init notice: {e}", "info")
    return {"status": "purged", "removed_files": removed_files, "shards_remaining": 0}
