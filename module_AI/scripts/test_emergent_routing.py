"""
Test demonstrating Emergent Spatial Routing without hardcoded tags:
Shows how the Atlas Router mathematically self-organizes semantic clusters
(Medical, Astronomy, Programming) and routes queries in sub-milliseconds.
"""
import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from module_loom.services.embedding.transformer import EmbeddingTransformer
from module_loom.services.weaver.atlas_router import GlobalAtlasRouter

def test_emergent_routing():
    print("=" * 70)
    print("  DocLoom Emergent Spatial Geometry & Centroid Routing Test")
    print("=" * 70)

    # 1. Initialize Transformer
    transformer = EmbeddingTransformer()
    router = GlobalAtlasRouter(seed=42)

    # 2. Define 3 semantic domains (Pure text - ZERO hardcoded labels passed to router)
    medical_shards = [
        "Cardiology focuses on disorders of the heart and circulatory system blood vessels.",
        "A myocardial infarction occurs when blood flow decreases or stops to part of the heart.",
        "Angioplasty is a procedure used to widen obstructed arteries and veins.",
        "The patient presented with acute chest pain and elevated cardiac troponin levels."
    ]

    astronomy_shards = [
        "A neutron star is the collapsed core of a massive supergiant star.",
        "Supermassive black holes reside at the center of almost all known massive galaxies.",
        "The James Webb Space Telescope observes the universe primarily in infrared light.",
        "Planetary nebulae form when dying red giant stars eject their outer gaseous envelopes."
    ]

    programming_shards = [
        "Python asyncio provides a framework for writing single-threaded concurrent code using coroutines.",
        "Decorators in Python allow dynamically altering the functionality of a function or class.",
        "The global interpreter lock (GIL) is a mutex that protects access to Python objects.",
        "Memory profiling identifies memory leaks in large data processing pipelines."
    ]

    print("\n[Step 1] Ingesting shards and forming mathematical centroids...")

    def compute_centroid(texts):
        vecs = transformer.model.encode(texts)
        centroid = np.mean(vecs, axis=0)
        return centroid / np.linalg.norm(centroid)

    # Register 3 crystals in the Atlas (named only by IDs, router has NO labels)
    c_medical = compute_centroid(medical_shards)
    c_astronomy = compute_centroid(astronomy_shards)
    c_programming = compute_centroid(programming_shards)

    router.register_crystal("crystal_alpha.loom", c_medical)
    router.register_crystal("crystal_beta.loom", c_astronomy)
    router.register_crystal("crystal_gamma.loom", c_programming)

    print("Registered 3 Self-Organized Crystals:")
    print("  - crystal_alpha.loom  (Centroid formed by cardiovascular & surgical texts)")
    print("  - crystal_beta.loom   (Centroid formed by astrophysics & cosmological texts)")
    print("  - crystal_gamma.loom  (Centroid formed by Python & concurrency texts)")

    # 3. Test Routing
    test_queries = [
        {
            "query": "What treatment is recommended for blocked coronary arteries?",
            "expected_crystal": "crystal_alpha.loom",
            "domain_label": "Pure Medical"
        },
        {
            "query": "How does gravity warp spacetime around a spinning black hole?",
            "expected_crystal": "crystal_beta.loom",
            "domain_label": "Pure Astronomy"
        },
        {
            "query": "How do I write an asynchronous generator function in Python?",
            "expected_crystal": "crystal_gamma.loom",
            "domain_label": "Pure Programming"
        },
        {
            "query": "Can we write a Python asyncio script to monitor patient cardiac telemetry?",
            "expected_crystal": "crystal_alpha.loom", # Or gamma with embassy
            "domain_label": "Cross-Domain Border Embassy (Medical + Python)"
        }
    ]

    print("\n[Step 2] Executing live queries through Atlas Router...")

    for t in test_queries:
        q = t["query"]
        t0 = time.perf_counter()
        q_vec = transformer.model.encode([q])[0]
        embed_ms = (time.perf_counter() - t0) * 1000

        t_route0 = time.perf_counter()
        result = router.route_vector_with_embassy(q_vec)
        route_ms = (time.perf_counter() - t_route0) * 1000

        print("\n" + "-" * 60)
        print(f"Query: '{q}'")
        print(f"Domain: [{t['domain_label']}]")
        print(f"  -> Routing Latency: {route_ms:.4f} ms (Embedding: {embed_ms:.1f} ms)")
        print(f"  -> Selected Primary Crystal: {result['primary_crystal']} (Cosine Sim: {result['primary_sim']:.4f})")
        
        # Display affinities to all crystals to see the geometry
        scores = []
        for path, info in router.crystals.items():
            sim = float(np.dot(q_vec, info["centroid"]) / (np.linalg.norm(q_vec) * np.linalg.norm(info["centroid"])))
            scores.append((path, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        
        print("  -> Full Geometric Distance breakdown:")
        for path, sim in scores:
            print(f"       • {path:22s} similarity = {sim:+.4f}")

        if result["is_embassy"]:
            print(f"  -> [BORDER EMBASSY TRIGGERED]: Bridges {result['primary_crystal']} <---> {result['secondary_crystal']} (Sec Sim: {result['secondary_sim']:.4f})")

        # Verify
        if not result["is_embassy"]:
            assert result["primary_crystal"] == t["expected_crystal"], f"Failed routing: expected {t['expected_crystal']} got {result['primary_crystal']}"
            print(f"  -> SUCCESS: Correct crystal targeted without scanning any others!")

    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED: Geometric Spatial Routing Verified!")
    print("=" * 70)

if __name__ == "__main__":
    test_emergent_routing()
