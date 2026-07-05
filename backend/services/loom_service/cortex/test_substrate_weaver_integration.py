import os
import time
import shutil
import unittest
import msgpack
import numpy as np

# Add project root to sys.path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.services.loom_service.cortex.SubstrateWeaver import SubstrateWeaver

class TestSubstrateWeaverIntegration(unittest.TestCase):
    def setUp(self):
        # Setup temporary testing storage
        self.test_dir = os.path.abspath(os.path.join(current_dir, "test_brain_data_weaver_temp"))
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception:
                pass
        
        # We initialize SubstrateWeaver with a small node limit and the test storage directory
        self.weaver = SubstrateWeaver(node_limit=10, storage_dir=self.test_dir)
        # Nudge collapse threshold lower for testing with small dataset
        self.weaver.engine.tau_collapse = 2.0

    def tearDown(self):
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception:
                pass

    def test_weave_cycle_integration(self):
        print("\n--> Testing WeaveBrainCoordinator.orchestrate_weave Integration...")
        
        # Conceptual Dataset: two distinct topics to trigger two clusters
        data = [
            "Quantum physics is the study of matter and energy at the nanometer scale.",
            "Quantum gates are building blocks for quantum computers.",
            "Qubits are quantum mechanical systems that represent data.",
            "Quantum computing is a rapidly expanding field of computer science.",
            "Pineapples are tropical fruits that grow in warm climates.",
            "A pineapple plant can take up to three years to flower and fruit.",
            "Pineapple growth requires highly acidic sandy loam soils."
        ]
        
        # 1. Initialize WeaveBrainCoordinator
        from backend.services.loom_service.weaver.weaver_coordinator import WeaveBrainCoordinator
        from backend.services.loom_service.weaver.decode.brain_decoder import decode_loom
        
        coordinator = WeaveBrainCoordinator(
            seed=42,
            storage_dir=self.test_dir,
            dimension=384
        )
        
        # 2. Execute orchestrate_weave
        coordinator.orchestrate_weave(data)
        
        # 3. Verify that output files were successfully created by the weaver coordinator
        crystal_path = os.path.join(self.test_dir, "crystal_1.loom")
        self.assertTrue(os.path.exists(crystal_path), "Crystal loom file was not created by WeaveBrainCoordinator.")
        
        coords_path = os.path.join(self.test_dir, "coordinates.bin")
        physics_path = os.path.join(self.test_dir, "physics_tensors.bin")
        hdc_path = os.path.join(self.test_dir, "hdc_signatures.bin")
        causal_path = os.path.join(self.test_dir, "causal_links.bin")
        
        self.assertTrue(os.path.exists(coords_path), "Coordinates binary tensor was not saved.")
        self.assertTrue(os.path.exists(physics_path), "Physics binary tensor was not saved.")
        self.assertTrue(os.path.exists(hdc_path), "HDC signatures binary was not saved.")
        self.assertTrue(os.path.exists(causal_path), "Causal links binary was not saved.")
        
        # 4. Decode the crystal file and verify contents
        decoded = decode_loom(crystal_path)
        self.assertEqual(decoded["file_type"], "PHYSICAL_SUBSTRATE_CONTAINER")
        self.assertEqual(decoded["universe_seed_key"], 42)
        self.assertGreater(decoded["num_shards"], 0)
        
        texts = [s["text"] for s in decoded["shards"]]
        self.assertIn(data[0], texts)
        
        print("[+] WeaveBrainCoordinator integration test completed successfully.")

if __name__ == "__main__":
    unittest.main()
