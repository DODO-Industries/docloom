from typing import Dict, Any

class ScenarioGenerator:
    """
    Generates test environmental states and files mock setups to dry-run plans.
    """
    def __init__(self):
        pass

    def generate_mock_environment(self, scenario_type: str) -> Dict[str, Any]:
        """Returns standard state setups based on target scenario tags."""
        if scenario_type == "empty_workspace":
            return {
                "workspace_path": "d:\\persnol\\DocLoom\\sandbox",
                "files": [],
                "permissions": "user"
            }
        elif scenario_type == "standard_docloom":
            return {
                "workspace_path": "d:\\persnol\\DocLoom\\backend",
                "files": ["main.py", "weaver.py", "viewer.py"],
                "permissions": "admin"
            }
        return {
            "workspace_path": "unknown",
            "files": []
        }
