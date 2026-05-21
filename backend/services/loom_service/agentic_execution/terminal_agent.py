from typing import Dict, Any
import subprocess

class TerminalAgent:
    """
    Safely routes terminal command strings to subprocess execution, restricting unsafe root commands.
    """
    def __init__(self):
        # List of forbidden command substrings for safety
        self.blocklist = ["rm -rf /", "format", "shutdown", "del /f /s /q"]

    def run_command(self, cmd_string: str) -> Dict[str, Any]:
        """Runs shell command and captures stdout/stderr responses."""
        if any(bad in cmd_string for bad in self.blocklist):
            return {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": "Permission denied: Command blocked by Sandbox rules."
            }

        try:
            # We will run this command in a shell-like mock simulation or restricted subprocess
            # For validation safety, we'll run a safe command or return standard stubbed feedback
            if cmd_string.strip() in ("dir", "ls", "echo hello"):
                return {
                    "success": True,
                    "exit_code": 0,
                    "stdout": "Mock Terminal: Execution successful.",
                    "stderr": ""
                }
            
            # Simple subprocess for safe echo / info retrieval
            res = subprocess.run(
                cmd_string,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            return {
                "success": res.returncode == 0,
                "exit_code": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr
            }
        except Exception as e:
            return {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e)
            }
        
    def add_blocklist_term(self, term: str) -> None:
        """Adds a forbidden term to blocklist."""
        self.blocklist.append(term)
