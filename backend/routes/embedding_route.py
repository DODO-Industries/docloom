import os
import sys
from fastapi import APIRouter, HTTPException, FastAPI
from pydantic import BaseModel
from typing import List

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.services.loom_service.cortex.embedding.transformer import EmbeddingTransformer
from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("EmbeddingRoute")
router = APIRouter(tags=["Embedding"])

# Initialize the shared transformer singleton (loaded once)
_transformer = EmbeddingTransformer()

class EmbeddingRequest(BaseModel):
    texts: List[str]

@router.post("/embed")
async def get_embeddings(request: EmbeddingRequest):
    """
    Generate continuous embeddings for the provided texts using SentenceTransformer.
    """
    try:
        embeddings = _transformer.get_embeddings(request.texts)
        # Convert np.ndarray to list for JSON serialization
        return {
            "success": True,
            "embeddings": [emb.tolist() for emb in embeddings]
        }
    except Exception as e:
        log_service(logger, f"Error in /embed: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Start a stand-alone app hosting just this route for testing
    app = FastAPI(title="Stand-alone Embedding Service")
    app.include_router(router)
    log_service(logger, "Starting stand-alone embedding service on http://localhost:8000", "info")
    uvicorn.run(app, host="0.0.0.0", port=8000)
