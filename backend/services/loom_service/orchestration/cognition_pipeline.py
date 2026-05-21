from typing import Dict, Any, List

class CognitionPipeline:
    """
    Orchestrates the sequential flow of inputs and outputs through all modules in a single cognitive tick.
    """
    def __init__(self, core_workspace: Any, scheduler: Any, safety_sandbox: Any, controller: Any):
        self.workspace = core_workspace
        self.scheduler = scheduler
        self.sandbox = safety_sandbox
        self.controller = controller
        self.tick_count = 0

    def step_tick(self, input_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes one full cognitive loop tick:
        1. Ingest perception inputs
        2. Verify sandbox constraints
        3. Trigger planning & action choices
        4. Apply self-improvement updates
        """
        self.tick_count += 1
        
        # Step 1: Sandbox checks
        cmd = input_signal.get("command", "")
        if cmd and not self.sandbox.is_command_safe(cmd):
            return {
                "tick": self.tick_count,
                "status": "blocked",
                "reason": f"Command '{cmd}' blocked by sandbox constraints."
            }

        # Step 2: Route task scheduling weights
        self.scheduler.register_task(f"tick_{self.tick_count}", 1.0)
        self.scheduler.tick_scheduler()

        # Step 3: Run tick controller logic
        decision = self.controller.process_percept(input_signal)

        # Update workspace properties
        self.workspace.publish_event("tick_completed", {
            "tick": self.tick_count,
            "decision": decision
        })

        return {
            "tick": self.tick_count,
            "status": "success",
            "decision": decision
        }
