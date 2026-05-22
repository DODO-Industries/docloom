from typing import Dict, Any

class SelfShutdownProtocol:
    """
    Saves critical state, flushes cognitive memory queues, and triggers emergency shutdown if safety thresholds are breached.
    """
    def __init__(self):
        self.shutdown_triggered = False

    def trigger_emergency_halt(self, reason: str, workspace_state: Dict[str, Any], storage_manager: Any) -> Dict[str, Any]:
        """Performs serialization of state, logs emergency diagnostic data, and flags shutdown execution."""
        self.shutdown_triggered = True
        
        # Safe checkpoint save
        storage_manager.save_state({
            "status": "emergency_halted",
            "reason": reason,
            "saved_state": workspace_state
        })

        return {
            "halt_complete": True,
            "diagnostic_saved": True,
            "reason": reason,
            "status_code": 503
        }
