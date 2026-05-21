from typing import Dict, Any, List

class PermissionSystem:
    """
    Manages active permissions and routes request intercepts for critical operations requiring approvals.
    """
    def __init__(self):
        # Maps action -> permission status
        self.grants: Dict[str, bool] = {
            "read_file": True,
            "write_file": False,
            "run_command": False
        }

    def grant_permission(self, action: str) -> None:
        """Grants permission override."""
        self.grants[action] = True

    def revoke_permission(self, action: str) -> None:
        """Revokes permission override."""
        self.grants[action] = False

    def request_execution_permission(self, action: str, target: str) -> bool:
        """Returns True if pre-approved, otherwise returns False (requires prompt intervention)."""
        # If pre-granted, approve automatically
        if self.grants.get(action, False):
            return True
        return False
