import os
import time
import shutil
import unittest
try:
    import psutil
except ImportError:
    psutil = None
import numpy as np

# Add project root to sys.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.services.loom_service.cortex.latent_field_cognition.dynamic_field_substrate import (
    DynamicFieldSubstrateEngine,
    CONCEPT_DICT_PACKED,
    ROOT_DOMAINS
)

class TestDynamicFieldSubstrate(unittest.TestCase):
    def setUp(self):
        # Setup temporary testing storage
        self.test_dir = os.path.abspath(os.path.join(current_dir, "test_brain_data_temp"))
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception:
                pass
        self.engine = DynamicFieldSubstrateEngine(
            storage_dir=self.test_dir, 
            max_nodes=1100, 
            tau_collapse=2.0
        )

    def tearDown(self):
        self.engine.close()
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception:
                pass

    def test_ingestion(self):
        print("\n--> Testing Real-Time Ingestion Space Matrix Routing...")
        emb1 = np.random.uniform(-1, 1, 128)
        emb2 = np.random.uniform(-1, 1, 128)

        idx1 = self.engine.ingest("concept_1", emb1)
        idx2 = self.engine.ingest("concept_2", emb2)

        self.assertEqual(idx1, 0)
        self.assertEqual(idx2, 1)
        self.assertEqual(self.engine.current_node_count, 2)

        # Confirm coordinates initialization
        self.assertTrue(np.allclose(self.engine.coords_map[0], [0.0, 0.0, 0.0]))
        self.assertFalse(np.allclose(self.engine.coords_map[1], [0.0, 0.0, 0.0]))

        # Confirm physics properties
        self.assertEqual(self.engine.physics_map[0, 0], 1.0)  # Mass = 1.0
        self.assertEqual(self.engine.physics_map[0, 2], 1.0)  # Activation = 1.0

        # Confirm packed hdc signatures are not all zero
        self.assertFalse(np.all(self.engine.hdc_map[0] == 0))

    def test_physics_and_gravitational_collapse(self):
        print("\n--> Testing Vectorized Physics & Gravitational Collapse Spawning...")
        
        # Ingest 4 nodes
        idx1 = self.engine.ingest("c1", np.ones(128))
        idx2 = self.engine.ingest("c2", np.ones(128))
        idx3 = self.engine.ingest("c3", np.ones(128))
        idx4 = self.engine.ingest("c4", np.ones(128))

        # Manually force all coordinates to be exactly identical to trigger collapse
        self.engine.coords_map[:4] = np.array([[2.0, 2.0, 2.0]] * 4)

        # Start physics loop and wait briefly
        self.engine.start_physics_loop()
        time.sleep(0.5)
        self.engine.stop_physics_loop()

        # A new consolidation node (macro node) should have been spawned (n becomes 5)
        self.assertEqual(self.engine.current_node_count, 5)
        self.assertEqual(self.engine.concept_ids[4], "macro_4")
        
        # Consolidation node must have Mass = 100.0
        self.assertEqual(self.engine.physics_map[4, 0], 100.0)
        # Coordinates should match center of mass (which is [2.0, 2.0, 2.0])
        self.assertTrue(np.allclose(self.engine.coords_map[4], [2.0, 2.0, 2.0]))

    def test_causal_links_and_torque(self):
        print("\n--> Testing Causal Entanglement Stitching & Spatial Attraction Forces...")
        emb1 = np.ones(128)
        emb2 = -np.ones(128)

        idx1 = self.engine.ingest("concept_1", emb1)
        idx2 = self.engine.ingest("concept_2", emb2)

        # Position them apart
        self.engine.coords_map[idx1] = [1.0, 1.0, 1.0]
        self.engine.coords_map[idx2] = [3.0, 3.0, 3.0]

        # Record sequence transition concept_1 -> concept_2
        self.engine.record_transition("concept_1", "concept_2")

        # Verify links file initialization
        target = int(self.engine.causal_map[idx1, 0, 0])
        strength = self.engine.causal_map[idx1, 0, 1]
        self.assertEqual(target, idx2)
        self.assertEqual(strength, 1.0)

        # In physics iteration, phase alignment attracts coordinates
        # Force phases to be synchronized (so cos(diff) == 1)
        self.engine.physics_map[idx1, 1] = 0.0
        self.engine.physics_map[idx2, 1] = 0.0

        # Run physics thread briefly and check that coords are closer
        dist_before = np.linalg.norm(self.engine.coords_map[idx1] - self.engine.coords_map[idx2])
        
        self.engine.start_physics_loop()
        time.sleep(0.4)
        self.engine.stop_physics_loop()

        dist_after = np.linalg.norm(self.engine.coords_map[idx1] - self.engine.coords_map[idx2])
        self.assertLess(dist_after, dist_before, "Causal linkage failed to pull coordinates closer.")

    def test_autonomous_associative_probing(self):
        print("\n--> Testing Autonomous Associative Probing & Hamming Naming...")
        
        # Retrieve packed signature for Quantum Physics
        qp_sig = CONCEPT_DICT_PACKED[0].copy()
        
        # Convert packed signature to embedding representation (0 -> -1, 1 -> 1)
        unpacked_bits = np.unpackbits(qp_sig)[:128]
        bipolar = (unpacked_bits.astype(np.float32) * 2.0) - 1.0

        # Ingest
        idx = self.engine.ingest("quantum_node", bipolar)

        # Naming probe should match Quantum Physics with 100% similarity
        label = self.engine.probe_node_label(idx)
        self.assertTrue("100% Quantum Physics" in label)
        print(f"  Probing matched label output: '{label}'")

    def test_memory_leak(self):
        print("\n--> Testing Zero-Memory-Creep Verification via psutil...")
        if psutil is None:
            print("  [!] psutil is not installed. Skipping memory creep check.")
            self.skipTest("psutil is not installed")
            return
        process = psutil.Process(os.getpid())
        
        # Warmup
        self.engine.ingest("warmup_c", np.random.uniform(-1, 1, 128))
        
        mem_before = process.memory_info().rss / (1024 * 1024)  # in MB
        
        # Ingest 1000 nodes
        for i in range(1000):
            self.engine.ingest(f"many_{i}", np.random.uniform(-1, 1, 128))
            
        mem_after = process.memory_info().rss / (1024 * 1024)  # in MB
        diff = mem_after - mem_before
        
        print(f"  Memory footprint before: {mem_before:.2f} MB")
        print(f"  Memory footprint after: {mem_after:.2f} MB")
        print(f"  Memory increase: {diff:.2f} MB")
        
        # Ingestion of 1000 mapped nodes should not cause significant RAM creep (< 5 MB headroom)
        self.assertLess(diff, 5.0, "Zero-memory-creep verification failed. Memory leaked during ingestion.")

if __name__ == "__main__":
    unittest.main()
