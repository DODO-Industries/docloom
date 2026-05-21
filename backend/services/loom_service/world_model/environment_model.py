import os
import sys
from typing import Dict, Any

class EnvironmentModel:
    """
    Maintains awareness of the operating environment: OS platforms,
    workspace directory structures, config variables, and sandbox limits.
    """
    def __init__(self, workspace_path: str = "d:/persnol/DocLoom"):
        self.workspace_path = os.path.abspath(workspace_path)
        self.os_platform = sys.platform
        self.environment_vars = dict(os.environ)
        
    def scan_workspace_structure(self) -> Dict[str, Any]:
        """Maps directory folders and checks file existence in workspace."""
        structure = {}
        if os.path.exists(self.workspace_path):
            for root, dirs, files in os.walk(self.workspace_path):
                # Only scan first level for performance
                structure["folders"] = dirs[:15]
                structure["files"] = files[:30]
                break
        return structure
        
    def check_file_status(self, relative_path: str) -> Dict[str, Any]:
        full_path = os.path.join(self.workspace_path, relative_path)
        exists = os.path.exists(full_path)
        return {
            "path": full_path,
            "exists": exists,
            "size": os.path.getsize(full_path) if exists and os.path.isfile(full_path) else 0,
            "is_dir": os.path.isdir(full_path) if exists else False
        }
