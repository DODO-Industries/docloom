"""
Unified server entry point: delegates to backend.app.app.
Eliminates server code duplication across backend and module_loom.
"""
from backend.app import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
