from typing import List, Dict, Any, Optional

class HierarchicalPlanner:
    """
    Implements a Hierarchical Task Network (HTN) planner.
    Tracks compound and primitive tasks to construct a sequential execution plan.
    """
    def __init__(self):
        # HTN Methods (decompositions) and Operators (primitives)
        self.methods = {
            "build_app": ["design_schema", "write_code", "run_tests"],
            "research_topic": ["retrieve_docs", "synthesize_context", "check_contradictions"]
        }
        self.operators = {
            "design_schema": {"pre": [], "post": ["schema_ready"]},
            "write_code": {"pre": ["schema_ready"], "post": ["code_ready"]},
            "run_tests": {"pre": ["code_ready"], "post": ["system_verified"]},
            "retrieve_docs": {"pre": [], "post": ["docs_loaded"]},
            "synthesize_context": {"pre": ["docs_loaded"], "post": ["context_synthesized"]},
            "check_contradictions": {"pre": ["context_synthesized"], "post": ["contradictions_resolved"]}
        }
        
    def solve(self, initial_tasks: List[str], state: List[str]) -> List[str]:
        plan = []
        task_stack = list(initial_tasks)
        current_state = set(state)
        
        while task_stack:
            task = task_stack.pop(0)
            
            # If compound task, expand it using methods
            if task in self.methods:
                subtasks = self.methods[task]
                # Prepend subtasks to stack
                task_stack = subtasks + task_stack
            # If primitive operator, check preconditions and execute/add to plan
            elif task in self.operators:
                op = self.operators[task]
                preconditions = op["pre"]
                if all(p in current_state for p in preconditions):
                    plan.append(task)
                    current_state.update(op["post"])
                else:
                    # Precondition failed, planning failure!
                    # In a real HTN, we would backtrack; here we record a failure step
                    plan.append(f"fail_preconditions_{task}")
            else:
                # Assume it's a primitive action with no preconditions
                plan.append(task)
                
        return plan

    def create_and_solve_plan(self, goal: str) -> Dict[str, Any]:
        """Solves the HTN plan for a given goal text/symbol."""
        tasks = []
        if "build" in goal.lower():
            tasks = ["build_app"]
        elif "research" in goal.lower():
            tasks = ["research_topic"]
        else:
            tasks = [goal]
        
        plan = self.solve(tasks, [])
        return {
            "goal": goal,
            "tasks": tasks,
            "plan": plan,
            "status": "solved" if not any("fail" in t for t in plan) else "failed"
        }
