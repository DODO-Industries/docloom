import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from module_loom.config.env_config import (
    setup_logger, log_service, LOOM_HOST, LOOM_PORT, EMBEDDING_MODEL_NAME
)
from module_loom.routes.embedding_routes import router as embedding_router
from module_loom.routes.visualizer_routes import router as visualizer_router, close_coordinator
from module_loom.routes.testing_routes import router as testing_router
from module_AI.routes import router as ai_router

logger = setup_logger("DocLoomModule")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_service(logger, "Initializing DocLoom Loom Module...", "info")
    # Start and pre-warm the neural embedding engine on server boot
    try:
        from module_loom.services.embedding.transformer import EmbeddingTransformer
        log_service(logger, f"Starting Neural Embedding Engine ({EMBEDDING_MODEL_NAME})...", "info")
        transformer = EmbeddingTransformer()
        _ = transformer.model.encode(["DocLoom neural warmup on startup"])
        log_service(logger, "Neural Embedding Engine Ready.", "info")
    except Exception as e:
        log_service(logger, f"Embedding Engine startup notice: {e}", "warning")
    yield
    log_service(logger, "Shutting down loom module...", "info")
    try:
        close_coordinator()
    except Exception as e:
        log_service(logger, f"Coordinator shutdown skipped: {e}", "warning")


app = FastAPI(
    title="DocLoom Loom Module",
    description="Self-contained loom research module: weaving, cognition, and live visualization.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(embedding_router)
app.include_router(visualizer_router)
app.include_router(testing_router)
app.include_router(ai_router)

LIBS_DIR = os.path.join(BASE_DIR, "utils", "webVisualizer", "libs")
if os.path.exists(LIBS_DIR):
    app.mount("/libs", StaticFiles(directory=LIBS_DIR), name="libs")

# Static assets for the Testing Sandbox (CSS, JS engine)
TESTING_STATIC_DIR = os.path.join(BASE_DIR, "test", "static")
if os.path.exists(TESTING_STATIC_DIR):
    app.mount("/loom/testing/static", StaticFiles(directory=TESTING_STATIC_DIR), name="testing-static")


@app.get("/")
async def root():
    return {
        "app": "DocLoom Loom Module",
        "status": "Running",
        "endpoints": {
            "testing_sandbox": "/loom/testing",
            "neuro_visualizer": "/loom/neuro",
            "embedding": "/embed",
            "ai_ask": "/ai/ask",
            "ai_remember": "/ai/remember",
            "ai_stats": "/ai/stats",
        },
    }


if __name__ == "__main__":
    import uvicorn
    log_service(logger, f"Starting DocLoom Loom Module on http://{LOOM_HOST}:{LOOM_PORT}", "info")
    uvicorn.run(app, host=LOOM_HOST, port=LOOM_PORT)
