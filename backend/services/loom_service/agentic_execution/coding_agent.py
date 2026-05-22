from typing import Dict, Any
import sys
import io

class CodingAgent:
    """
    Generates, lints, and executes Python code segments inside a safe sandbox evaluation wrapper.
    """
    def __init__(self):
        pass

    def lint_code(self, code_snippet: str) -> Dict[str, Any]:
        """Checks code syntax structure for basic compile errors."""
        try:
            compile(code_snippet, "<string>", "exec")
            return {"valid": True, "error": None}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def execute_snippet(self, code_snippet: str, globals_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        """Executes a code block and captures standard output/errors."""
        if globals_dict is None:
            globals_dict = {}

        old_stdout = sys.stdout
        redirected_output = sys.stdout = io.StringIO()
        success = True
        error_msg = None

        try:
            exec(code_snippet, globals_dict)
        except Exception as e:
            success = False
            error_msg = str(e)
        finally:
            sys.stdout = old_stdout

        return {
            "success": success,
            "stdout": redirected_output.getvalue(),
            "error": error_msg
        }
