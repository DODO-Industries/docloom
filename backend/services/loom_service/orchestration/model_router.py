from typing import Dict, Any

class ModelRouter:
    """
    Routes requests to local vs remote API models based on prompt complexity or task types.
    """
    def __init__(self):
        pass

    def route_prompt(self, prompt: str, task_type: str) -> Dict[str, Any]:
        """
        Determines the optimal LLM size/provider to handle the prompt query.
        """
        prompt_len = len(prompt)
        
        # Heuristics based on length and task complexity
        if task_type in ("planning", "symbolic_theorem_proving") or prompt_len > 1500:
            return {
                "selected_model": "gemini-pro-api",
                "max_tokens": 4096,
                "reason": "Complex task type or long context window."
            }
        else:
            return {
                "selected_model": "local-llama-3-8b",
                "max_tokens": 1024,
                "reason": "Simple summarization, routing or formatting task."
            }
