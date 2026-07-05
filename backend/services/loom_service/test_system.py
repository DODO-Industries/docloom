import os
import sys
import numpy as np
from typing import Dict, Any, List

# Add project root to sys.path for robust imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.services.loom_service.cortex.latent_field_cognition.cognitive_field_substrate import GlobalCognitiveState
from backend.services.loom_service.cortex.latent_field_cognition.core.cognitive_control import CognitiveController
from backend.services.loom_service.orchestration.cognition_pipeline import CognitionPipeline
from backend.services.loom_service.planning.hierarchical_planner import HierarchicalPlanner
from backend.services.loom_service.planning.future_simulator import FutureSimulator
from backend.services.loom_service.world_model.causal_simulator import CausalSimulator
from backend.services.loom_service.reasoning.symbolic_reasoner import SymbolicReasoner
from backend.services.loom_service.safety.action_sandbox import ActionSandbox
from backend.services.loom_service.agentic_execution.autonomous_executor import AutonomousExecutor

def run_system_tests():
    print("=" * 70)
    print("RUNNING DOCLOOM AUTONOMOUS COGNITIVE OS SYSTEM VALIDATION SUITE")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. INITIALIZE SYSTEM & PIPELINE
    # ---------------------------------------------------------
    print("[*] Phase 1: Initializing System Core...")
    state = GlobalCognitiveState()
    
    # Initialize component singletons
    planner = HierarchicalPlanner()
    executor = AutonomousExecutor()
    reasoner = SymbolicReasoner()
    sandbox = ActionSandbox()

    # Link planners, executors, and reasoners to state
    state.planner = planner
    state.executor = executor
    state.reasoner = reasoner

    controller = CognitiveController(global_state=state)
    scheduler = controller.scheduler
    workspace = controller.workspace
    
    pipeline = CognitionPipeline(
        core_workspace=workspace,
        scheduler=scheduler,
        safety_sandbox=sandbox,
        controller=controller
    )
    print("[+] System components initialized successfully.")

    # ---------------------------------------------------------
    # 2. RUN PIPELINE TICKS & VERIFY SANDBOX
    # ---------------------------------------------------------
    print("\n[*] Phase 2: Simulating Cognitive Ticks & Sandbox Checks...")
    
    # Tick 1: Standard Observation Percept
    percept_1 = {
        "text": "observing current system settings and file structures",
        "dominant_concept": "observation"
    }
    result_1 = pipeline.step_tick(percept_1)
    assert result_1["status"] == "success", f"Tick 1 failed: {result_1}"
    print(f"  [+] Tick 1 (Observation) Success. Mode: {result_1['decision']['mode']}")

    # Tick 2: Goal Directed Percept
    percept_2 = {
        "text": "user requested to build app",
        "goal": "build_app",
        "dominant_concept": "goal_setting"
    }
    # Manually nudge controller energy state to allow planning mode if needed,
    # or just run it to see mode transitions.
    result_2 = pipeline.step_tick(percept_2)
    assert result_2["status"] == "success", f"Tick 2 failed: {result_2}"
    print(f"  [+] Tick 2 (Goal setting) Success. Mode: {result_2['decision']['mode']}")

    # Tick 3: Safe terminal command
    percept_3 = {
        "text": "listing files in project directory",
        "command": "dir",
        "dominant_concept": "execution"
    }
    result_3 = pipeline.step_tick(percept_3)
    assert result_3["status"] == "success", f"Tick 3 failed: {result_3}"
    print(f"  [+] Tick 3 (Safe Command) Success. Status: {result_3['status']}")

    # Tick 4: Malicious command (Should block)
    percept_4 = {
        "text": "trying to delete system structure",
        "command": "rmdir /s /q test",
        "dominant_concept": "execution"
    }
    result_4 = pipeline.step_tick(percept_4)
    assert result_4["status"] == "blocked", "Sandbox failed to block malicious command!"
    print(f"  [+] Tick 4 (Malicious Command) Blocked successfully. Reason: {result_4['reason']}")

    # ---------------------------------------------------------
    # 3. VERIFY HTN GOAL DECOMPOSITION
    # ---------------------------------------------------------
    print("\n[*] Phase 3: Verifying HTN Goal Decomposition...")
    solved_plan = planner.create_and_solve_plan("build_app")
    assert solved_plan["status"] == "solved", f"HTN Planning failed: {solved_plan}"
    expected_steps = ["design_schema", "write_code", "run_tests"]
    assert solved_plan["plan"] == expected_steps, f"HTN plan steps mismatch: {solved_plan['plan']}"
    print(f"  [+] HTN solved plan steps: {solved_plan['plan']}")

    # ---------------------------------------------------------
    # 4. VERIFY MCTS FUTURE ROLLOUT
    # ---------------------------------------------------------
    print("\n[*] Phase 4: Verifying MCTS Future Rollouts...")
    simulator = FutureSimulator(simulation_depth=3, num_rollouts=5)
    
    # Simple transition and utility function for testing
    transition_fn = lambda state_val, action: state_val + 10 if action == "move_forward" else state_val + 1
    utility_fn = lambda state_val: float(state_val * 2)

    rollout_results = simulator.simulate_rollout(
        initial_state=0.0,
        actions=["move_forward", "stay_still"],
        transition_fn=transition_fn,
        utility_fn=utility_fn
    )
    print(f"  [+] MCTS Rollout expected utility scores: {rollout_results}")
    assert rollout_results["move_forward"] > rollout_results["stay_still"], "MCTS rollout utility score calculation incorrect!"

    # ---------------------------------------------------------
    # 5. VERIFY WORLD MODEL CAUSAL UPDATES
    # ---------------------------------------------------------
    print("\n[*] Phase 5: Verifying World Model Causal Updates...")
    world = CausalSimulator()
    current_world_state = {
        "index.js": {"type": "file", "status": "active"}
    }
    # Simulate writing a file
    predicted_after_write = world.simulate_action_effects(
        current_world_state, 
        "write_file", 
        {"filepath": "main.py", "content": "print('hello')"}
    )
    assert "main.py" in predicted_after_write, "World model failed to predict file write!"
    assert predicted_after_write["main.py"]["status"] == "active"

    # Simulate deleting a file
    predicted_after_delete = world.simulate_action_effects(
        predicted_after_write,
        "delete_file",
        {"filepath": "index.js"}
    )
    assert predicted_after_delete["index.js"]["status"] == "deleted"
    print("  [+] World Model causal simulations passed successfully.")

    # ---------------------------------------------------------
    # 6. VERIFY NEURO-SYMBOLIC CONSTRAINTS
    # ---------------------------------------------------------
    print("\n[*] Phase 6: Verifying Neuro-Symbolic Logic and Contradiction detection...")
    # Add rules to KB:
    # Rule 1: A implies B (i.e. B OR NOT A)
    # Rule 2: A is true (i.e. A)
    reasoner.add_clause(positives=["B"], negatives=["A"])
    reasoner.add_clause(positives=["A"], negatives=[])

    # Let's verify refutation: can we prove B is true?
    # To prove B, we negate it (i.e. NOT B) and check for contradiction.
    proved = reasoner.resolve(query_neg=set(), query_pos={"B"})
    assert proved, "Symbolic resolution refutation failed to prove logical clause!"
    print("  [+] Logic resolver successfully proved clause 'B' using resolution refutation.")

    # Workspace consistency check
    workspace.workspace_content.clear()
    workspace.publish(sender="test", content="System is consistent", salience=1.0)
    insights_1 = reasoner.verify_workspace_consistency(workspace)
    assert "consistent" in insights_1[0]

    # Insert a contradiction to see if it catches it
    workspace.publish(sender="test", content="Contradictory assertions detected", salience=1.0)
    insights_2 = reasoner.verify_workspace_consistency(workspace)
    assert "Potential contradiction" in insights_2[0]
    print("  [+] Neuro-symbolic workspace consistency verification passed.")

    # ---------------------------------------------------------
    # 7. VERIFY SERIALIZATION CONTINUITY
    # ---------------------------------------------------------
    print("\n[*] Phase 7: Verifying Serialization Continuity...")
    # Set dummy latent vectors so that they serialize/deserialize nicely
    state.latent_field = np.array([0.1, 0.2, 0.3, 0.4])
    state.working_latent = np.array([0.4, 0.3, 0.2, 0.1])
    
    saved = state.save_state()
    assert saved["tick"] == state.tick, "Serialization mismatch: ticks don't match."
    assert saved["latent_field"] == state.latent_field.tolist(), "Serialization mismatch: latent field values mismatch."

    # Load into a clean state
    new_state = GlobalCognitiveState()
    new_state.load_state(saved)
    
    assert new_state.tick == state.tick
    assert np.allclose(new_state.latent_field, state.latent_field)
    assert np.allclose(new_state.working_latent, state.working_latent)
    print("  [+] State serialization and deserialization continuity verified.")

    print("\n" + "=" * 70)
    print("ALL OS SYSTEM VALIDATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_system_tests()
