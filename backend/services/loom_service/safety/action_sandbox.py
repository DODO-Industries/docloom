from typing import Dict, Any, List
import os

class ActionSandbox:
    """
    Enforces restrictions on file paths, directory write zones, and executed terminal scripts.
    """
    def __init__(self, allowed_directory_prefixes: List[str] = None):
        self.allowed_prefixes = allowed_directory_prefixes if allowed_directory_prefixes is not None else [
            "d:\\persnol\\docloom",
            "d:/persnol/docloom",
            "c:\\users\\atrira\\.gemini"
        ]

    def is_file_path_safe(self, absolute_path: str) -> bool:
        """Determines if a target file path resides inside allowed root paths."""
        try:
            norm_path = os.path.normpath(absolute_path).lower()
            for prefix in self.allowed_prefixes:
                norm_prefix = os.path.normpath(prefix).lower()
                if norm_path.startswith(norm_prefix):
                    return True
            return False
        except Exception:
            return False

    def is_command_safe(self, cmd_string: str) -> bool:
        """Scans terminal scripts for blocked/harmful shell inputs."""
        cmd_lower = cmd_string.lower()
        blocked_keywords = ["rmdir /s", "del /q", "format", "shutdown", "wget ", "curl "]
        return not any(keyword in cmd_lower for keyword in blocked_keywords)
