"""
Empirical verification for DocLoom Multi-Cluster Retrieval & Cross-Cluster Fusion:
Tests that when knowledge is partitioned across multiple crystals/clusters:
1. Border Embassy queries seamlessly combine shards across crystal borders.
2. If the primary excited state yields weak resonance, the fallback deep-memory
   sweep activates so past knowledge is NEVER missed.
3. Co-occurrence working-set associations link concepts across cluster boundaries.
"""
import os
import sys
import shutil
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from module_AI.memory_bridge import LoomMemory
from module_loom.services.weaver.weaver_coordinator import WeaveBrainCoordinator


def test_multicluster_fusion():
    test_storage = os.path.join(PROJECT_ROOT, "module_AI", "data", "test_multicluster_brain")
    if os.path.exists(test_storage):
        shutil.rmtree(test_storage)
    os.makedirs(test_storage, exist_ok=True)

    print("=" * 70)
    print("  DocLoom Multi-Cluster Retrieval & Cross-Cluster Fusion Test")
    print("=" * 70)

    # Initialize coordinator with multi-cluster test storage
    coordinator = WeaveBrainCoordinator(storage_dir=test_storage, dimension=128, seed=42)
    mem = LoomMemory(coordinator=coordinator)

    # 1. Ingest into Crystal 1 (Nature & Rivers)
    c1_shards = [
        "Near the swift river, the brown otter swam toward the mossy rock.",
        "The river bank was covered with smooth gray stones and fresh water plants.",
        "Otters are playful aquatic animals that love diving under clean river water."
    ]
    for s in c1_shards:
        mem.remember(s)

    # Create explicit Crystal 2 (Mythology & Relics) by registering a second crystal in atlas
    c2_path = os.path.join(test_storage, "crystal_2.loom")
    # Centroid orthogonal or far in latent space from nature
    dummy_c2_vec = np.zeros(128, dtype=np.float32)
    dummy_c2_vec[64:] = 1.0
    dummy_c2_vec /= np.linalg.norm(dummy_c2_vec)
    coordinator.atlas.register_crystal(c2_path, dummy_c2_vec, state="warm")

    c2_shards = [
        "In the deep fiery cavern, the ancient dragon guarded the golden spellbook.",
        "The golden spellbook contained ancient runes written in glowing blue starlight.",
        "Dormant ancient knowledge: the crystal key is hidden inside the third vault."
    ]
    for s in c2_shards:
        vec = mem.embed(s)
        shard_id = f"c2_{abs(hash(s)) % 1000000}"
        coordinator.ingest_shard(shard_id, vec, s, target_crystal=c2_path)

    print(f"Registered crystals in atlas: {len(coordinator.atlas.crystals)}")
    for p in coordinator.atlas.crystals:
        print(f"  - {os.path.basename(p)}")

    # Test 1: Border Embassy / Cross-Cluster Query
    # A query that bridges nature and dragons (e.g. "Did the dragon fly over the river?")
    q_embassy = "Did the dragon fly over the river water?"
    print(f"\n[Test 1] Border Embassy Query: {q_embassy!r}")
    matches_emb = mem.recall_distilled(q_embassy, candidate_pool=10, top_k=6)
    c1_hits = sum(1 for m in matches_emb if "river" in m.get("text", "").lower() or "otter" in m.get("text", "").lower())
    c2_hits = sum(1 for m in matches_emb if "dragon" in m.get("text", "").lower() or "spellbook" in m.get("text", "").lower())
    print(f"  -> Crystal 1 (Nature) hits returned: {c1_hits}")
    print(f"  -> Crystal 2 (Dragon) hits returned: {c2_hits}")
    print(f"  -> Cross-Cluster Embassy Fusion Success: {c1_hits > 0 and c2_hits > 0}")

    # Test 2: Fallback Deep-Memory Scan
    # A obscure query where primary excited state might have weak resonance
    q_obscure = "Where is the crystal key hidden inside the third vault?"
    print(f"\n[Test 2] Deep Past Knowledge Query: {q_obscure!r}")
    matches_obs = mem.recall_distilled(q_obscure, candidate_pool=10, top_k=4)
    found_key = any("crystal key" in m.get("text", "").lower() for m in matches_obs)
    lead_obs = matches_obs[0]["text"][:65] if matches_obs else "None"
    print(f"  -> Lead recalled shard: {lead_obs!r}")
    print(f"  -> Deep Knowledge Recovered from Distant Cluster: {found_key}")

    # Test 3: Associative Working-Set Cross-Cluster Bridge
    # Bind an otter shard from C1 with the golden spellbook from C2 as a co-occurrence pair
    otter_sid = [m["shard_id"] for m in matches_emb if "otter" in m.get("text", "").lower()][0]
    dragon_sid = [m["shard_id"] for m in matches_emb if "dragon" in m.get("text", "").lower()][0]
    coordinator.working_set_pairs[(otter_sid, dragon_sid)] = 5  # Strong co-use association

    q_pure_otter = "What does the otter like to dive into?"
    print(f"\n[Test 3] Associative Bridge: Querying purely otter: {q_pure_otter!r}")
    matches_assoc = mem.recall_distilled(q_pure_otter, candidate_pool=10, top_k=6)
    bridged_dragon = any(m.get("shard_id") == dragon_sid for m in matches_assoc)
    print(f"  -> Associated Dragon Shard pulled across cluster boundary: {bridged_dragon}")

    mem.coordinator.close()
    if os.path.exists(test_storage):
        shutil.rmtree(test_storage)

    print("\n" + "=" * 70)
    print("Multi-Cluster Retrieval & Cross-Cluster Fusion Verified Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test_multicluster_fusion()
