from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from module_AI.pipeline import CognitivePipeline

router = APIRouter(prefix="/ai", tags=["AI"])
_pipeline: Optional[CognitivePipeline] = None


def _get_pipeline() -> CognitivePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CognitivePipeline()
    return _pipeline


class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    remember: bool = True


class RememberRequest(BaseModel):
    text: str
    metadata: dict = {}


@router.post("/ask")
def ask(req: AskRequest):
    return _get_pipeline().ask(req.query, top_k=req.top_k, remember_interaction=req.remember)


@router.post("/remember")
def remember(req: RememberRequest):
    shard_id = _get_pipeline().remember(req.text, metadata=req.metadata)
    return {"shard_id": shard_id}


@router.get("/stats")
def stats():
    return _get_pipeline().memory.stats()
