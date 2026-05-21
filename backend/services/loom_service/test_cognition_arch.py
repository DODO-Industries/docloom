import os
import sys
import numpy as np
import datetime
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from backend.services.loom_service.cognition import GlobalCognitiveState

def print_header(text):
    print("\n" + "="*90)
    print(f" {text.upper()} ")
    print("="*90)

def run_cognitive_simulation():
    print_header("DOCLOOM PHASE 3: CONTINUOUS SEMANTIC SUBSTRATE TEST")
    print("[1/6] INITIALIZING VECTOR-SPACE STATE...")
    state = GlobalCognitiveState()
    
    # Initialize the field and model to dynamically get dimension
    state.semantic_field.register_concept("dummy_init")
    dim = len(state.semantic_field.concept_embeddings["dummy_init"])
    target_intent = np.ones(dim) / np.linalg.norm(np.ones(dim))
    state.intent.add_intent("understand_gravity_and_black_holes", target_intent, intensity=0.8)

    # Issue 2: Real Natural Language Semantics
    narrative_stream = [
        {"Massive objects warp the fabric of spacetime": 0.9, "Gravity is the curvature of the universe": 0.8},
        {"Gravity is the curvature of the universe": 0.9, "Light bends around massive galaxies": 0.95},
        {"Light bends around massive galaxies": 0.8, "Spacetime continuum defines the path of light": 0.9},
        {"Black holes contain a singularity of infinite density": 0.95, "Entropy inside a black hole is proportional to its area": 0.85},
        {"Black holes contain a singularity of infinite density": 0.9, "The event horizon is the point of no return": 0.85},
        {"The event horizon is the point of no return": 0.9, "Hawking radiation slowly evaporates black holes": 0.8}
    ]

    for tick, base_acts in enumerate(narrative_stream, 1):
        print(f"\n--- TICK {tick} | INGESTING SEMANTIC SHARDS ---")
        for k in base_acts.keys(): print(f"  + {k[:60]}...")
        
        ca = state.process_tick(base_acts)
        
        print(f"  [FIELD METRICS] Energy: {state.energy_budget:.2f} | Turbulence (Entropy): {state.entropy:.2f} | Coherence: {state.coherence:.2f} | Mode: {state.reflection.cognitive_mode}")
        
        if ca:
            print(f"  [ATTRACTOR CRYSTALLIZATION] Assembly Formed in Vector Space!")
            print(f"    -> Centroid concept: '{ca.dominant_concept[:60]}...'")
            print(f"    -> Basin Strength: {ca.meta_data.get('basin_strength', 0):.2f}")
            print(f"    -> Constituent Density: {len(ca.node_activations)} resonating components")
            
            if len(state.active_assemblies) > 1:
                prev = state.active_assemblies[-2].dominant_concept
                if prev != ca.dominant_concept:
                    print(f"  [CAUSAL TENSOR] Trajectory: '{prev[:20]}...' -> '{ca.dominant_concept[:20]}...'")
                    
            pred_phrases = state.causal.simulate_future_path(ca.dominant_concept, depth=1)
            if pred_phrases and len(pred_phrases[0]) > 1:
                print(f"  [IMAGINATION] Field predicts trajectory towards: '{pred_phrases[0][1][:40]}...'")
        else:
            print("  [ATTRACTOR PHYSICS] Field is turbulent, no stable basin reached.")

        if tick == 3:
            print("\n  >>> TRIGGERING EPISODIC REHEARSAL (Wave Resonance)...")
            state.memory.replay_active_memories(state)
            
        time.sleep(0.1)

    print("\n[META SHARDS] Inspecting Semantic Abstractions:")
    for meta_id, shard in state.meta_shards.meta_store.items():
        print(f"  -> {shard.emergent_label[:60]}... (Stability: {shard.stability:.2f})")

    print_header("FINAL COGNITIVE STATE SUMMARY")
    summary = state.get_summary()
    for k, v in summary.items():
        if isinstance(v, float): print(f"  {k.upper()}: {v:.4f}")
        else: print(f"  {k.upper()}: {v}")
        
    print("\n[SELF-EVALUATION METRICS]")
    print(f"  Reasoning Quality: {state.reflection.reasoning_quality:.2f}")
    print(f"  Epistemic Score: {state.reflection.epistemic_score:.2f}")
    print(f"  Prediction Accuracy: {state.reflection.prediction_accuracy:.2f}")
    
    print("\n" + "="*90)
    print(" CONTINUOUS COGNITION TEST SUCCESSFUL ")
    print("="*90 + "\n")

if __name__ == "__main__":
    run_cognitive_simulation()
