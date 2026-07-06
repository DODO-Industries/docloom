import os
import sys
import json
import time
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# =============================================================================
# DOCLOOM — NEUROMORPHIC VISUALIZATION PIPELINE (WebSocket streamer)
# =============================================================================
# Streams the live Weave Brain state to the WebGL front end (neuro_viz.html):
#   Component A — 128-bit signature matrix + Hamming sweep (assimilation vs
#                 accommodation events on ingest)
#   Component B — spatiotemporal wave propagation on recall (per-node ΔA)
#   Component C — gravitational drift + Kuramoto phase coloring + deep sleep
#   Component D — universe.loom single-file commit map (segments + journal)
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("NeuroVisualizer")
router = APIRouter(prefix="/loom/neuro", tags=["NeuroVisualizer"])

VIZ_HTML_PATH = os.path.join(PROJECT_ROOT, "Vizualization", "neuro_viz.html")
MAX_VIZ_NODES = 1500

_coordinator = None
_projection: Optional[np.ndarray] = None


def _brain_dir() -> str:
    """Prefer assets/.brain_data (current home), fall back to repo-root/.brain_data."""
    for candidate in (
        os.path.join(PROJECT_ROOT, "assets", ".brain_data"),
        os.path.join(PROJECT_ROOT, ".brain_data"),
    ):
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(PROJECT_ROOT, "assets", ".brain_data")


def get_coordinator():
    global _coordinator
    if _coordinator is None:
        from backend.services.loom_service.weaver.weaver_coordinator import WeaveBrainCoordinator
        _coordinator = WeaveBrainCoordinator(storage_dir=_brain_dir())
        log_service(logger, f"NeuroViz bound to brain at {_brain_dir()} "
                            f"(dim={_coordinator.dimension}, seed={_coordinator.seed})", "info")
    return _coordinator


def embed_text(text: str, dimension: int) -> np.ndarray:
    """Real embeddings when the LLM service is available; deterministic
    hash-seeded fallback otherwise (keeps the visualizer alive offline)."""
    try:
        from backend.services.LLM_service.embedding.embedding_manager import get_embeddings
        vec = np.asarray(get_embeddings([text])[0], dtype=np.float32)
    except Exception:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        vec = np.random.default_rng(seed).normal(0, 1, dimension).astype(np.float32)
    if len(vec) > dimension:
        vec = vec[:dimension]
    elif len(vec) < dimension:
        vec = np.pad(vec, (0, dimension - len(vec)), constant_values=0.0)
    return vec


def _get_projection(dimension: int) -> np.ndarray:
    """Fixed seeded D->3 projection so node positions are stable across sessions."""
    global _projection
    if _projection is None or _projection.shape[1] != dimension:
        rng = np.random.default_rng(1618)
        _projection = rng.normal(0, 1.0 / np.sqrt(dimension), size=(3, dimension)).astype(np.float32)
    return _projection


def project_3d(vec: np.ndarray, radius: float = 42.0) -> List[float]:
    p = _get_projection(len(vec)) @ vec
    norm = float(np.linalg.norm(p))
    if norm < 1e-9:
        return [0.0, 0.0, 0.0]
    unit = p / norm
    # Radial spread by vector magnitude keeps clusters volumetric, not shell-bound
    r = radius * (0.45 + 0.55 * float(np.tanh(np.linalg.norm(vec) / np.sqrt(len(vec)))))
    return [float(unit[0] * r), float(unit[1] * r), float(unit[2] * r)]


def signature_bits(vec: np.ndarray) -> List[int]:
    sig = np.sign(vec[:128])
    sig[sig == 0] = 1
    if len(sig) < 128:
        sig = np.pad(sig, (0, 128 - len(sig)), constant_values=1)
    return ((sig + 1) // 2).astype(int).tolist()


def _universe_stats(coord) -> Dict[str, Any]:
    if coord.universe is None:
        return {}
    try:
        return coord.universe.stats()
    except Exception:
        return {}


def _node_payload(coord, store, idx: int, crystal: str) -> Dict[str, Any]:
    vec = store.get_vector(idx)
    meta = store.get_meta(idx)
    sid = store.get_id(idx)
    ledger = coord.ram_ledger.get(sid, {})
    return {
        "id": sid,
        "pos": project_3d(vec),
        "activation": float(ledger.get("activation", 0.0)),
        "phase": float(ledger.get("phase_angle", 0.0)),
        "mass": float(meta.get("mass", 1.0)),
        "text": str(meta.get("text", ""))[:140],
        "crystal": os.path.basename(crystal),
    }


def _snapshot(coord) -> Dict[str, Any]:
    """Initial world state: up to MAX_VIZ_NODES most-recent shards across crystals."""
    nodes: List[Dict[str, Any]] = []
    crystals_info = []
    n_crystals = max(1, len(coord.atlas.crystals))
    per_crystal = max(1, MAX_VIZ_NODES // n_crystals)
    for cpath, info in coord.atlas.crystals.items():
        crystals_info.append({
            "name": os.path.basename(cpath),
            "state": info["state"],
            "num_shards": int(info["num_shards"]),
        })
        if not os.path.exists(cpath):
            continue
        try:
            store = coord._get_store(cpath)
            take = min(store.n, per_crystal)
            for idx in range(store.n - take, store.n):
                nodes.append(_node_payload(coord, store, idx, cpath))
        except Exception as e:
            log_service(logger, f"snapshot skip {cpath}: {e}", "warning")
    return {
        "type": "init",
        "dimension": coord.dimension,
        "seed": coord.seed,
        "nodes": nodes,
        "crystals": crystals_info,
        "universe": _universe_stats(coord),
        "ledger_size": len(coord.ram_ledger),
    }


@router.get("/", response_class=HTMLResponse)
async def get_neuro_visualizer():
    if not os.path.exists(VIZ_HTML_PATH):
        return HTMLResponse("<h1>neuro_viz.html not found</h1>", status_code=404)
    with open(VIZ_HTML_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@router.websocket("/ws")
async def neuro_ws(ws: WebSocket):
    await ws.accept()
    coord = get_coordinator()
    await ws.send_text(json.dumps(_snapshot(coord)))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            mtype = msg.get("type")

            # ---------------- Component A + ingest ----------------
            if mtype == "ingest":
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                vec = embed_text(text, coord.dimension)
                sid = f"viz_{hashlib.md5(text.encode('utf-8')).hexdigest()[:10]}"

                crystals_before = set(coord.atlas.crystals.keys())
                leaders_before = 0
                target_probe = None
                try:
                    momentum = coord.physics.calculate_momentum_vector(
                        vec, coord.seed_core.get_micro_socket(sid), 1.0)
                    target_probe = coord._route_for_ingest(momentum)
                    leaders_before = coord._get_store(target_probe).num_leaders
                except Exception:
                    pass

                target = coord.ingest_shard(sid, vec, text, mass=1.0)
                store = coord._get_store(target)
                assimilated = store.num_leaders == leaders_before  # no new leader -> joined a cluster
                new_crystals = set(coord.atlas.crystals.keys()) - crystals_before

                await ws.send_text(json.dumps({
                    "type": "ingest_event",
                    "node": _node_payload(coord, store, store.n - 1, target),
                    "signature": signature_bits(vec),
                    "assimilated": bool(assimilated),
                    "division": [os.path.basename(p) for p in new_crystals],
                    "universe": _universe_stats(coord),
                    "ledger_size": len(coord.ram_ledger),
                }))

            # ---------------- Component B + recall ----------------
            elif mtype == "recall":
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                q = embed_text(text, coord.dimension)
                results = coord.recall(q, top_k=int(msg.get("top_k", 8)))

                # Per-node wave gains for the nodes on screen (viz-only compute)
                wave: Dict[str, float] = {}
                try:
                    target = coord.atlas.route_vector(q)
                    if os.path.exists(target):
                        store = coord._get_store(target)
                        norms = store.norms_all()
                        qn = float(np.linalg.norm(q))
                        denom = np.maximum(norms * qn, 1e-12)
                        cos = store.dot_all(q) / denom
                        take = min(store.n, MAX_VIZ_NODES)
                        for idx in range(store.n - take, store.n):
                            wave[store.get_id(idx)] = round(float(max(0.0, cos[idx])), 4)
                except Exception:
                    pass

                phases = {sid: round(float(s.get("phase_angle", 0.0)), 4)
                          for sid, s in coord.ram_ledger.items()}
                activations = {sid: round(float(s.get("activation", 0.0)), 4)
                               for sid, s in coord.ram_ledger.items()}

                await ws.send_text(json.dumps({
                    "type": "recall_event",
                    "query_pos": project_3d(q),
                    "top": results,
                    "wave": wave,
                    "phases": phases,
                    "activations": activations,
                    "universe": _universe_stats(coord),
                    "ledger_size": len(coord.ram_ledger),
                }))

            # ---------------- Component C: decay tick ----------------
            elif mtype == "decay_tick":
                coord.enforce_spatiotemporal_decay(time.time())
                await ws.send_text(json.dumps({
                    "type": "decay_event",
                    "activations": {sid: round(float(s.get("activation", 0.0)), 4)
                                    for sid, s in coord.ram_ledger.items()},
                    "ledger_size": len(coord.ram_ledger),
                }))

            elif mtype == "refresh":
                await ws.send_text(json.dumps(_snapshot(coord)))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log_service(logger, f"NeuroViz WS error: {e}", "error")
        try:
            await ws.close()
        except Exception:
            pass
