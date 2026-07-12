import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("DocLoomApp")

# Routers are mounted best-effort: the legacy loom/AI routes depend on modules
# still living in test/work_on_architecture (unfinished refactor) — a broken
# legacy import must not take down the core services.
_routers = []
for _name, _path in [
    ("loom", "backend.test.work_on_architecture.loomServer_route"),
    ("ai", "backend.test.work_on_architecture.AIServer_route"),
    ("embedding", "backend.routes.embedding.embedding_route"),
    ("neuro_visualizer", "backend.routes.visualizer.neuro_visualizer_route"),
]:
    try:
        _mod = __import__(_path, fromlist=["router"])
        _routers.append((_name, _mod.router))
    except Exception as _e:
        log_service(logger, f"Router '{_name}' unavailable ({_path}): {_e}", "warning")

@asynccontextmanager
async def lifespan(app: FastAPI):
    log_service(logger, "Initializing DocLoom Integrated Services...", "info")
    yield
    log_service(logger, "Shutting down services...", "info")
    # Best-effort, matching the router-loading style above: flush the brain
    # coordinator's final state on a clean shutdown rather than relying on
    # __del__ at interpreter exit, which is not guaranteed to run.
    try:
        from backend.routes.visualizer.neuro_visualizer_route import close_coordinator
        close_coordinator()
    except Exception as _e:
        log_service(logger, f"Coordinator shutdown skipped: {_e}", "warning")

app = FastAPI(
    title="DocLoom Integrated Neural Engine",
    description="Full-spectrum document intelligence: Weaving, Navigation, and AI Reasoning.",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routes (best-effort — see loader above)
for _name, _router in _routers:
    app.include_router(_router)
    log_service(logger, f"Mounted router: {_name}", "info")

# Static Files for Visualizer
LIBS_DIR = os.path.abspath(os.path.join(BASE_DIR, "utils", "webVisualizer", "libs"))
if os.path.exists(LIBS_DIR):
    app.mount("/libs", StaticFiles(directory=LIBS_DIR), name="libs")

@app.get("/")
async def root():
    return {
        "app": "DocLoom",
        "status": "Running",
        "endpoints": {
            "visualizer": "/loom/visualizer",
            "neuro_visualizer": "/loom/neuro",
            "loom_api": "/loom",
            "ai_api": "/ai"
        }
    }

if __name__ == "__main__":
    import uvicorn
    log_service(logger, "Starting Streamlined DocLoom Server on http://localhost:8000", "info")
    uvicorn.run(app, host="0.0.0.0", port=8000)
