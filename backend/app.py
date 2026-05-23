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

from backend.routes.loomServer_route import router as loom_router
from backend.routes.AIServer_route import router as ai_router
from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("DocLoomApp")

@asynccontextmanager
async def lifespan(app: FastAPI):
    log_service(logger, "Initializing DocLoom Integrated Services...", "info")
    yield
    log_service(logger, "Shutting down services...", "info")

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

# Mount Routes
app.include_router(loom_router)
app.include_router(ai_router)

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
            "loom_api": "/loom",
            "ai_api": "/ai"
        }
    }

if __name__ == "__main__":
    import uvicorn
    log_service(logger, "Starting Streamlined DocLoom Server on http://localhost:8000", "info")
    uvicorn.run(app, host="0.0.0.0", port=8000)
