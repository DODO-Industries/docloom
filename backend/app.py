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

# Mount active core routes
_routers = []
for _name, _path in [
    ("embedding", "module_loom.routes.embedding_routes"),
    ("neuro_visualizer", "module_loom.routes.visualizer_routes"),
    ("testing_visualizer", "module_loom.routes.testing_routes"),
    ("ai", "module_AI.routes"),
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
        from module_loom.routes.visualizer_routes import close_coordinator
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
LIBS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "module_loom", "utils", "webVisualizer", "libs"))
if os.path.exists(LIBS_DIR):
    app.mount("/libs", StaticFiles(directory=LIBS_DIR), name="libs")

# Static assets for the Testing Sandbox (CSS, JS engine)
TESTING_STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "module_loom", "test", "static"))
if os.path.exists(TESTING_STATIC_DIR):
    app.mount("/loom/testing/static", StaticFiles(directory=TESTING_STATIC_DIR), name="testing-static")

@app.get("/")
async def root():
    return {
        "app": "DocLoom",
        "status": "Running",
        "endpoints": {
            "testing_visualizer": "/loom/testing",
            "neuro_visualizer": "/loom/neuro",
            "visualizer": "/loom/visualizer",
            "loom_api": "/loom",
            "ai_api": "/ai"
        }
    }

if __name__ == "__main__":
    import uvicorn
    log_service(logger, "Starting Streamlined DocLoom Server on http://localhost:8000", "info")
    uvicorn.run(app, host="0.0.0.0", port=8000)
