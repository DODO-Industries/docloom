import os
import sys
import json
import time
import math
import asyncio
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set

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
#   Component E — per-crystal color/offset + causal-graph edges + smart
#                 overview/expand sampling + a batched bulk-ingest feeder
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.config.envConfig import setup_logger, log_service
from backend.config.tunningManagment import tuning_manager

logger = setup_logger("NeuroVisualizer")
router = APIRouter(prefix="/loom/neuro", tags=["NeuroVisualizer"])

VIZ_HTML_PATH = os.path.join(PROJECT_ROOT, "Vizualization", "neuro_viz.html")
MAX_VIZ_NODES = 1500
# Default overview is intentionally small — "don't render the whole brain by
# default." expand_crystal (manual click or auto-triggered by a recall hit)
# loads a fuller sample for one specific crystal on demand.
OVERVIEW_PER_CRYSTAL = 40
EXPAND_MAX_NODES = 2000
MAX_VIZ_EDGES = 800
# Golden-angle stepping: each crystal's hue/offset only depends on its own
# permanent index, never on how many crystals currently exist — so an
# already-placed crystal never visually jumps when a new one is discovered.
GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))
CRYSTAL_OFFSET_RADIUS = 52.0
# "Seed 5 crystals" feeder — a temporary, per-call crystal-cap override (see
# _run_feeder) makes 5+ crystals reachable with a few hundred shards instead
# of the real MAX_CRYSTAL_SIZE default (500,000).
FEEDER_TOTAL_SHARDS = 800
FEEDER_CRYSTAL_CAP = 150
FEEDER_CHUNK_SIZE = 150

_coordinator = None
_projection: Optional[np.ndarray] = None
_crystal_registry: Dict[str, int] = {}
_last_autosave_check = 0.0


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


def close_coordinator() -> None:
    """
    Explicit shutdown hook for app.py's lifespan — flushes whatever changed
    since the last autosave (checkpoint + full living_state.bin save) via
    WeaveBrainCoordinator.close(). Relying only on __del__ at interpreter exit
    is not reliable (skipped on some shutdown paths); this makes a clean
    server stop actually durable instead of losing up to the autosave
    interval's worth of recent activity.
    """
    global _coordinator
    if _coordinator is not None:
        try:
            _coordinator.close()
        except Exception as e:
            log_service(logger, f"NeuroViz coordinator close failed: {e}", "warning")
        finally:
            # Reset the singleton so a subsequent get_coordinator() (e.g. a
            # fresh boot cycle, or a test simulating one) reopens from disk
            # instead of reusing this now-closed instance's file handles.
            _coordinator = None


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


def _crystal_index(path: str) -> int:
    """Permanent index assigned the first time a crystal path is seen — stable
    for the life of the process, independent of how many crystals exist now."""
    key = os.path.abspath(path)
    idx = _crystal_registry.get(key)
    if idx is None:
        idx = len(_crystal_registry)
        _crystal_registry[key] = idx
    return idx


def crystal_hue(path: str) -> float:
    return (_crystal_index(path) * 0.61803398875) % 1.0


def crystal_offset(path: str) -> List[float]:
    """Deterministic per-crystal spatial offset so multiple crystals render as
    visually distinct regions instead of overlapping in the shared projection.
    Golden-angle stepped, so it never changes for a crystal once assigned."""
    angle = _crystal_index(path) * GOLDEN_ANGLE
    return [CRYSTAL_OFFSET_RADIUS * math.cos(angle), 0.0, CRYSTAL_OFFSET_RADIUS * math.sin(angle)]


def project_3d_for_crystal(vec: np.ndarray, crystal_path: str, radius: float = 42.0) -> List[float]:
    base = project_3d(vec, radius=radius)
    off = crystal_offset(crystal_path)
    return [base[0] + off[0], base[1] + off[1], base[2] + off[2]]


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
    payload = {
        "id": sid,
        "pos": project_3d_for_crystal(vec, crystal),
        "activation": float(ledger.get("activation", 0.0)),
        "phase": float(ledger.get("phase_angle", 0.0)),
        "mass": float(meta.get("mass", 1.0)),
        "text": str(meta.get("text", ""))[:140],
        "crystal": os.path.basename(crystal),
    }
    if meta.get("semantic_crystal"):
        payload["semantic_crystal"] = True
    superseded_by = coord.superseded_by_crystal.get(os.path.abspath(crystal), {}).get(sid)
    if superseded_by:
        payload["superseded_by"] = superseded_by
    return payload


def _edges_for_nodes(coord, node_ids: Set[str], max_edges: int = MAX_VIZ_EDGES) -> List[List[Any]]:
    """Causal-graph edges (coord.causal_graph, grown by process_cognitive_tick)
    restricted to pairs where BOTH endpoints are already in `node_ids` — the
    frontend only ever needs to draw a line between two points it actually has
    positions for. Capped by weight so a densely-connected graph can't blow up
    the payload."""
    if not node_ids or coord.causal_graph.number_of_edges() == 0:
        return []
    edges: List[List[Any]] = []
    for u, v, data in coord.causal_graph.edges(data=True):
        if u in node_ids and v in node_ids:
            edges.append([u, v, round(float(data.get("weight", 0.5)), 3)])
    if len(edges) > max_edges:
        edges.sort(key=lambda e: -e[2])
        edges = edges[:max_edges]
    return edges


def _overview_sample_indices(store, centroid: Optional[np.ndarray], per_crystal: int) -> List[int]:
    """
    A small, representative sample of one crystal instead of "the whole
    thing" — mixes most-recent, highest-mass, and centroid-nearest shards
    (deduped) so the default view is cheap and the expanded view is still
    meaningful, not just an arbitrary prefix/suffix slice.
    """
    n = store.n
    if n <= per_crystal:
        return list(range(n))

    third = max(1, per_crystal // 3)
    picks: Set[int] = set()

    picks.update(range(max(0, n - third), n))  # most-recent

    mass = store.mass_all()
    if len(mass):
        for i in np.argsort(-mass)[:third]:
            picks.add(int(i))

    remaining = per_crystal - len(picks)
    if remaining > 0 and centroid is not None and float(np.linalg.norm(centroid)) > 1e-9:
        norms = store.norms_all()
        c_norm = float(np.linalg.norm(centroid))
        denom = np.maximum(norms * c_norm, 1e-12)
        cos = store.dot_all(centroid) / denom
        for i in np.argsort(-cos):
            i = int(i)
            if i not in picks:
                picks.add(i)
                remaining -= 1
                if remaining <= 0:
                    break

    ordered = sorted(picks)
    return ordered[:per_crystal] if len(ordered) > per_crystal else ordered


def _snapshot(coord) -> Dict[str, Any]:
    """
    Initial/refresh world state — a SMALL overview, not the whole brain: up to
    OVERVIEW_PER_CRYSTAL representative shards per crystal (see
    _overview_sample_indices), further capped by MAX_VIZ_NODES total. Use
    expand_crystal (manual click or auto-triggered on recall) to load a
    fuller sample of one specific crystal on demand.
    """
    nodes: List[Dict[str, Any]] = []
    crystals_info = []
    n_crystals = max(1, len(coord.atlas.crystals))
    per_crystal = max(1, min(OVERVIEW_PER_CRYSTAL, MAX_VIZ_NODES // n_crystals))
    for cpath, info in coord.atlas.crystals.items():
        crystals_info.append({
            "name": os.path.basename(cpath),
            "state": info["state"],
            "num_shards": int(info["num_shards"]),
            "hue": crystal_hue(cpath),
        })
        if not os.path.exists(cpath):
            continue
        try:
            store = coord._get_store(cpath)
            centroid = info.get("centroid")
            for idx in _overview_sample_indices(store, centroid, per_crystal):
                nodes.append(_node_payload(coord, store, idx, cpath))
        except Exception as e:
            log_service(logger, f"snapshot skip {cpath}: {e}", "warning")
    edges = _edges_for_nodes(coord, {n["id"] for n in nodes})
    return {
        "type": "init",
        "dimension": coord.dimension,
        "seed": coord.seed,
        "nodes": nodes,
        "edges": edges,
        "crystals": crystals_info,
        "universe": _universe_stats(coord),
        "ledger_size": len(coord.ram_ledger),
    }


def _crystal_path_by_name(coord, name: str) -> Optional[str]:
    for cpath in coord.atlas.crystals.keys():
        if os.path.basename(cpath) == name:
            return cpath
    return None


def _expand_payload(coord, crystal_path: str) -> Dict[str, Any]:
    """Fuller sample (up to EXPAND_MAX_NODES) of ONE crystal — "as we search
    more, show more of that part/crystal." Sent on a manual crystal-legend
    click or automatically the first time a recall lands in a crystal that
    hasn't been expanded yet on this connection (see neuro_ws)."""
    name = os.path.basename(crystal_path)
    if not os.path.exists(crystal_path):
        return {"type": "expand_event", "crystal": name, "nodes": [], "edges": []}
    store = coord._get_store(crystal_path)
    centroid = coord.atlas.crystals.get(crystal_path, {}).get("centroid")
    nodes = [
        _node_payload(coord, store, idx, crystal_path)
        for idx in _overview_sample_indices(store, centroid, EXPAND_MAX_NODES)
    ]
    edges = _edges_for_nodes(coord, {n["id"] for n in nodes})
    return {"type": "expand_event", "crystal": name, "nodes": nodes, "edges": edges}


def _maybe_autosave(coord) -> None:
    """Cheap time-gated trigger for coord.autosave() — most WS messages skip
    this entirely (a single float compare), so it never adds latency to the
    hot ingest/recall path; see WeaveBrainCoordinator.autosave()."""
    global _last_autosave_check
    now = time.time()
    interval = tuning_manager.get_int("VIZ_AUTOSAVE_INTERVAL_SEC", 45)
    if now - _last_autosave_check >= interval:
        _last_autosave_check = now
        coord.autosave()


def _sleep_event_payload(coord, result: Dict[str, Any]) -> Dict[str, Any]:
    consolidated = []
    for c in result.get("consolidated", []):
        entry = coord.ram_ledger.get(c["new_shard"], {})
        cpath, cidx = entry.get("crystal_path"), entry.get("crystal_idx")
        node = None
        if cpath and cidx is not None:
            try:
                node = _node_payload(coord, coord._get_store(cpath), cidx, cpath)
            except Exception:
                pass
        consolidated.append({
            "concept": c["concept"], "new_shard": c["new_shard"],
            "source_shard_ids": c["source_shard_ids"], "node": node,
        })
    return {"type": "sleep_event", "consolidated": consolidated, "ledger_size": len(coord.ram_ledger)}


def _maybe_sleep(coord) -> Optional[Dict[str, Any]]:
    """
    Opportunistic idle-trigger for coord.maybe_sleep_cycle(force=False) — no
    background task/timer exists in this app, so this piggybacks on the
    frontend's existing decay_tick heartbeat (sent every 5s while connected)
    to get a real, if coarse, periodic idle check for free. If nobody's
    connected, sleep simply doesn't run until someone reconnects — the same
    limitation autosave already has.
    """
    try:
        return coord.maybe_sleep_cycle(force=False)
    except Exception as e:
        log_service(logger, f"NeuroViz sleep cycle check failed: {e}", "warning")
        return None


# =============================================================================
# "SEED 5x" FEEDER — bulk test data across 5 real topic domains, real
# embeddings (batch-called, not one API round trip per shard), a temporary
# lowered crystal cap so cellular division actually triggers, and ONE
# chunked-progress protocol instead of one WS message per shard (the direct
# fix for the "blockage" reported when scaling up test data by hand).
# =============================================================================

_FEEDER_TOPICS: Dict[str, List[str]] = {
    "marine_biology": [
        "Coral reefs are built over centuries by colonies of tiny calcium-carbonate-secreting polyps.",
        "Bioluminescent plankton produce flashes of light through a luciferin-luciferase chemical reaction.",
        "The deep ocean's midnight zone receives no sunlight and hosts pressure-adapted, often blind species.",
        "Whale falls create temporary deep-sea ecosystems that can sustain scavenger life for decades.",
        "Octopuses have three hearts and blue, copper-based blood instead of iron-based hemoglobin.",
        "Kelp forests along cold coastlines provide shelter and food for sea otters, fish, and urchins.",
        "Hydrothermal vent communities derive energy from chemosynthesis rather than sunlight.",
        "Sea turtles navigate thousands of miles using Earth's magnetic field as an internal compass.",
        "Mangrove root systems stabilize coastlines and serve as nurseries for juvenile fish.",
        "The ocean's twilight zone hosts the largest daily animal migration on the planet.",
        "Coral bleaching occurs when rising water temperatures cause polyps to expel their symbiotic algae.",
        "Sharks have electroreceptor organs called ampullae of Lorenzini that detect faint electric fields.",
        "Tidal pools create harsh, fluctuating micro-environments that shape highly adaptable species.",
        "Plankton blooms can be tracked from space by the color they impart to surface waters.",
        "Deep-sea anglerfish use a bioluminescent lure to attract prey in total darkness.",
        "Ocean currents distribute heat globally and strongly influence regional climate patterns.",
        "Sea sponges lack true tissues or organs yet filter enormous volumes of water for nutrients.",
        "Barnacles permanently cement themselves to a surface after a brief free-swimming larval stage.",
        "Squid can change skin color and texture almost instantly using chromatophores and papillae.",
        "Coastal upwelling brings nutrient-rich deep water to the surface, fueling productive fisheries.",
    ],
    "renaissance_art": [
        "Linear perspective, formalized by Brunelleschi, gave Renaissance painting a convincing sense of depth.",
        "Leonardo da Vinci's sfumato technique blends tones so gradually that outlines seem to dissolve.",
        "Michelangelo carved the Pietà from a single block of Carrara marble in his early twenties.",
        "Fresco painting required artists to work quickly onto wet plaster before it dried.",
        "Botticelli's Primavera weaves classical mythology into an allegory of springtime abundance.",
        "The Medici family's patronage financed much of Florence's early Renaissance artistic output.",
        "Raphael's School of Athens places ancient philosophers within a grand classical architecture.",
        "Oil paint, refined in the Renaissance, allowed slower work and richer, layered color.",
        "Donatello's bronze David was the first freestanding nude sculpture since antiquity.",
        "Renaissance workshops trained apprentices through years of grinding pigments before ever painting.",
        "The Sistine Chapel ceiling took Michelangelo roughly four years to complete while lying on scaffolding.",
        "Vasari's Lives of the Artists is one of the earliest works of art history and biography.",
        "Venetian painters like Titian became known for vivid color achieved through oil glazing.",
        "Humanist scholarship encouraged Renaissance artists to study anatomy and classical proportion.",
        "Chiaroscuro uses strong contrasts between light and dark to model three-dimensional form.",
        "Patrons commissioned portraits partly to display wealth, status, and dynastic continuity.",
        "Brunelleschi's dome for Florence Cathedral solved an engineering problem thought unsolvable.",
        "Northern Renaissance painters like van Eyck achieved minute detail through thin oil glazes.",
        "The rediscovery of classical texts reshaped how Renaissance artists depicted the human body.",
        "Drawing, or disegno, was considered the intellectual foundation underlying all Renaissance art forms.",
    ],
    "computer_networking": [
        "TCP establishes a reliable connection through a three-way handshake before data transfer begins.",
        "DNS translates human-readable domain names into the IP addresses machines actually route traffic to.",
        "A subnet mask divides an IP address into network and host portions for routing decisions.",
        "Switches operate at the data link layer, forwarding frames based on MAC addresses.",
        "BGP is the routing protocol that determines how traffic flows between autonomous systems on the internet.",
        "Firewalls inspect and filter packets according to rules defined by source, destination, and port.",
        "NAT lets many devices on a private network share a single public IP address.",
        "Latency, jitter, and packet loss are the three metrics that most affect real-time network quality.",
        "A VPN tunnels traffic through an encrypted connection over an otherwise untrusted network.",
        "DHCP automatically assigns IP addresses to devices joining a network without manual configuration.",
        "Load balancers distribute incoming requests across multiple servers to avoid any single bottleneck.",
        "The OSI model describes networking in seven layers, from physical cabling up to the application.",
        "TLS handshakes negotiate encryption keys so that HTTPS traffic can't be read in transit.",
        "Routers forward packets between different networks based on entries in a routing table.",
        "Content delivery networks cache data geographically closer to users to reduce round-trip latency.",
        "A collision domain is a network segment where data packets can interfere with one another.",
        "MTU defines the largest packet size a network link can carry without fragmentation.",
        "Wi-Fi networks use CSMA/CA to avoid multiple devices transmitting over each other at once.",
        "Network address translation and port forwarding both help route traffic into private networks.",
        "Traceroute reveals the sequence of routers a packet passes through to reach its destination.",
    ],
    "culinary_spices": [
        "Saffron, the world's most expensive spice by weight, is hand-harvested from crocus flower stigmas.",
        "Toasting whole cumin seeds before grinding releases significantly more of their aromatic oils.",
        "Black pepper's pungency comes primarily from the alkaloid compound piperine.",
        "Vanilla pods are cured for months through a labor-intensive process of sweating and drying.",
        "Cinnamon sold in most supermarkets is usually cassia, a close but distinct relative of true cinnamon.",
        "Star anise contains anethole, the same aromatic compound responsible for the smell of fennel.",
        "Turmeric's yellow color comes from curcumin, a compound also studied for its anti-inflammatory properties.",
        "Cardamom pods can be green, black, or white depending on species and processing method.",
        "Smoked paprika gets its flavor from peppers slowly dried over oak wood fires in Spain.",
        "Cloves contain eugenol, a compound that gives them their distinctive numbing, medicinal aroma.",
        "Garam masala blends vary widely by region, family, and even by individual household.",
        "Nutmeg and mace both come from the same fruit — nutmeg is the seed, mace is its lacy covering.",
        "Chili heat is measured in Scoville units, based on the concentration of capsaicin present.",
        "Fenugreek seeds taste bitter raw but mellow considerably once toasted or slow-cooked.",
        "Za'atar is a Middle Eastern spice blend typically combining thyme, sumac, and toasted sesame.",
        "Sumac's tart, lemony flavor comes from drying and grinding the berries of the sumac shrub.",
        "Five-spice powder balances sweet, sour, bitter, salty, and umami in one blend.",
        "Whole spices generally stay potent far longer than pre-ground versions of the same spice.",
        "Curry leaves, distinct from curry powder, are used fresh or fried to release their aroma.",
        "Asafoetida is used in tiny amounts, mellowing into a savory, onion-like flavor once cooked.",
    ],
    "mountain_geology": [
        "The Himalayas continue to rise as the Indian and Eurasian tectonic plates keep colliding.",
        "Fold mountains form when compressive tectonic forces buckle layers of rock upward.",
        "Glacial erosion carves distinctive U-shaped valleys, unlike the V-shaped valleys rivers cut.",
        "Volcanic mountains like Mount Fuji build up over time from repeated layers of lava and ash.",
        "Cirques are bowl-shaped hollows carved into mountainsides by the head of a glacier.",
        "The tree line on a mountain marks the elevation above which temperatures are too low for tree growth.",
        "Isostatic rebound causes land to slowly rise after the weight of ancient ice sheets melts away.",
        "Scree slopes form from freeze-thaw weathering that fractures exposed rock over time.",
        "Fault-block mountains, like the Sierra Nevada, form when large crustal blocks tilt along faults.",
        "Alpine environments experience dramatic temperature swings between day and night at high altitude.",
        "Moraines are ridges of rock and sediment deposited at the edges of a moving glacier.",
        "Orogeny is the geological term for the process of mountain formation through crustal deformation.",
        "Permafrost beneath some high-altitude soils remains frozen year-round despite surface thaw in summer.",
        "Hanging valleys form where a smaller glacier's valley floor sits higher than the main valley it joins.",
        "Mountain rain shadows create sharply drier climates on the leeward side of a range.",
        "Metamorphic rock like schist and gneiss often forms deep within mountain-building collision zones.",
        "Avalanches are most likely when a weak snow layer lies buried beneath heavier recent snowfall.",
        "Karst topography can form in mountainous limestone regions, riddled with caves and sinkholes.",
        "Seismic activity along mountain fault lines is a direct signature of ongoing tectonic uplift.",
        "Alpine glaciers are retreating at accelerating rates as global average temperatures rise.",
    ],
}


def _feeder_texts(total: int) -> List[Tuple[str, str]]:
    """Cycles through all 5 topic domains, padding each domain's real base
    sentences with light numbered variants once exhausted, to reach `total`
    (topic, text) pairs — real, semantically distinct content per domain
    rather than synthetic placeholder vectors."""
    topics = list(_FEEDER_TOPICS.keys())
    per_topic = max(1, total // len(topics))
    out: List[Tuple[str, str]] = []
    for topic in topics:
        base = _FEEDER_TOPICS[topic]
        for i in range(per_topic):
            sentence = base[i % len(base)]
            if i >= len(base):
                sentence = f"{sentence} (further notes, batch {i // len(base)})"
            out.append((topic, sentence))
    return out[:total]


async def _run_feeder(ws: WebSocket, coord) -> None:
    """
    Seeds FEEDER_TOTAL_SHARDS shards across 5 topic domains, batch-embedding
    each chunk through the real embedding pipeline (not one call per shard),
    ingesting via the EXISTING coord.ingest_batch() (one batched append per
    crystal, one atlas save, one journal write per chunk). Temporarily lowers
    coord.max_shards_per_crystal so cellular division actually triggers —
    _route_for_ingest() reads that attribute fresh on every call, so a
    try/finally override is sufficient; no other code path needs to change.
    Reports progress in chunks and finishes with ONE full snapshot rebuild
    (bulk_done), instead of hundreds of incremental per-shard messages.
    """
    from backend.services.LLM_service.embedding.embedding_manager import get_embeddings

    indexed_pairs = list(enumerate(_feeder_texts(FEEDER_TOTAL_SHARDS)))
    original_cap = coord.max_shards_per_crystal
    coord.max_shards_per_crystal = FEEDER_CRYSTAL_CAP
    done = 0
    try:
        for start in range(0, len(indexed_pairs), FEEDER_CHUNK_SIZE):
            chunk = indexed_pairs[start:start + FEEDER_CHUNK_SIZE]
            texts = [text for _, (_, text) in chunk]
            try:
                vectors = get_embeddings(texts)
            except Exception:
                vectors = [embed_text(t, coord.dimension) for t in texts]

            items = []
            for (gi, (topic, text)), vec in zip(chunk, vectors):
                vec = np.asarray(vec, dtype=np.float32)
                if len(vec) > coord.dimension:
                    vec = vec[:coord.dimension]
                elif len(vec) < coord.dimension:
                    vec = np.pad(vec, (0, coord.dimension - len(vec)), constant_values=0.0)
                items.append({
                    "shard_id": f"seed_{topic}_{gi}",
                    "vector": vec,
                    "text": text,
                    "mass": 1.0,
                    "metadata": {"topic": topic, "seeded": True},
                })

            coord.ingest_batch(items)
            done += len(items)
            await ws.send_text(json.dumps({
                "type": "bulk_progress",
                "done": done,
                "total": len(indexed_pairs),
                "crystals": len(coord.atlas.crystals),
            }))
            await asyncio.sleep(0)  # yield control between chunks
    finally:
        coord.max_shards_per_crystal = original_cap

    payload = _snapshot(coord)
    payload["type"] = "bulk_done"
    await ws.send_text(json.dumps(payload))


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

    # Per-connection only (not global) — tracks which crystals this client has
    # already been sent a full expand_event for, so a recall auto-expands a
    # crystal's detail exactly once, not on every subsequent hit.
    expanded_crystals: Set[str] = set()

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
                prev_sid = coord._last_ingested_sid

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

                # Best-effort single edge from whatever was ingested right
                # before this — the frontend silently skips it if it doesn't
                # already have a node for prev_sid, so no extra lookup needed here.
                edges = [[prev_sid, sid, 1.0]] if prev_sid and prev_sid != sid else []

                await ws.send_text(json.dumps({
                    "type": "ingest_event",
                    "node": _node_payload(coord, store, store.n - 1, target),
                    "edges": edges,
                    "signature": signature_bits(vec),
                    "assimilated": bool(assimilated),
                    "division": [os.path.basename(p) for p in new_crystals],
                    "universe": _universe_stats(coord),
                    "ledger_size": len(coord.ram_ledger),
                }))
                _maybe_autosave(coord)
                sleep_result = _maybe_sleep(coord)
                if sleep_result:
                    await ws.send_text(json.dumps(_sleep_event_payload(coord, sleep_result)))

            # ---------------- Component B + recall ----------------
            elif mtype == "recall":
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                q = embed_text(text, coord.dimension)
                results = coord.recall(q, top_k=int(msg.get("top_k", 8)))

                # Per-node wave gains for the nodes on screen (viz-only compute)
                wave: Dict[str, float] = {}
                target = None
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
                edges = _edges_for_nodes(coord, set(wave.keys()))

                await ws.send_text(json.dumps({
                    "type": "recall_event",
                    "query_pos": project_3d_for_crystal(q, target) if target else project_3d(q),
                    "top": results,
                    "wave": wave,
                    "edges": edges,
                    "phases": phases,
                    "activations": activations,
                    "universe": _universe_stats(coord),
                    "ledger_size": len(coord.ram_ledger),
                }))
                _maybe_autosave(coord)
                sleep_result = _maybe_sleep(coord)
                if sleep_result:
                    await ws.send_text(json.dumps(_sleep_event_payload(coord, sleep_result)))

                # "As we search more, show more of that part/crystal" — the
                # first time a recall lands in a crystal this connection
                # hasn't seen expanded yet, follow up with its fuller sample.
                if target and os.path.basename(target) not in expanded_crystals:
                    expanded_crystals.add(os.path.basename(target))
                    await ws.send_text(json.dumps(_expand_payload(coord, target)))

            # ---------------- Component C: decay tick ----------------
            elif mtype == "decay_tick":
                coord.enforce_spatiotemporal_decay(time.time())
                await ws.send_text(json.dumps({
                    "type": "decay_event",
                    "activations": {sid: round(float(s.get("activation", 0.0)), 4)
                                    for sid, s in coord.ram_ledger.items()},
                    "ledger_size": len(coord.ram_ledger),
                }))
                _maybe_autosave(coord)
                # decay_tick is the frontend's existing 5s heartbeat — the only
                # recurring signal available without adding a background task,
                # so this is where the idle-triggered (non-forced) sleep check
                # actually gets a chance to fire while a client stays connected.
                sleep_result = _maybe_sleep(coord)
                if sleep_result:
                    await ws.send_text(json.dumps(_sleep_event_payload(coord, sleep_result)))

            elif mtype == "refresh":
                await ws.send_text(json.dumps(_snapshot(coord)))

            # ---------------- Component E: crystal expand + feeder ----------------
            elif mtype == "expand_crystal":
                name = (msg.get("crystal") or "").strip()
                cpath = _crystal_path_by_name(coord, name)
                if cpath:
                    expanded_crystals.add(name)
                    await ws.send_text(json.dumps(_expand_payload(coord, cpath)))

            elif mtype == "bulk_ingest":
                await _run_feeder(ws, coord)
                expanded_crystals.clear()  # crystal set likely changed shape entirely
                # Force the full living-state save (not just the gated lightweight
                # snapshot) right after a bulk seed — a significant, deliberate
                # state change worth an immediate durable checkpoint rather than
                # waiting out the normal autosave interval.
                coord.autosave(force_full=True)

            # ---------------- Sleep cycle: consolidation over the store ----------------
            elif mtype == "sleep_now":
                result = coord.maybe_sleep_cycle(force=True)
                payload = _sleep_event_payload(coord, result) if result else {
                    "type": "sleep_event", "consolidated": [], "ledger_size": len(coord.ram_ledger),
                }
                await ws.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log_service(logger, f"NeuroViz WS error: {e}", "error")
        try:
            await ws.close()
        except Exception:
            pass
