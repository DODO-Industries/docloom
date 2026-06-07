import os
import sys
import json
import urllib.request
import numpy as np
from backend.config.envConfig import EMBEDDING_API_URL, setup_logger, log_service

logger = setup_logger("LLMEmbeddingService")

class LLMEmbeddingService:
    """
    Centralized Embedding Service under the LLM_service boundary.
    Coordinates embedding generation using a 4-tier lookup strategy:
    1. In-process lookup: Reuses active EmbeddingTransformer singleton from FastAPI route.
    2. HTTP POST query: Calls the configured EMBEDDING_API_URL endpoint.
    3. Local fallback: Lazily loads local SentenceTransformer model.
    4. Dummy fallback: Generates deterministic random arrays of size 384.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMEmbeddingService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "local_model"):
            return
        self.local_model = None

    def encode(self, text, **kwargs):
        """
        Encode text(s) into embeddings.
        Supports single string or list of strings.
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else list(text)

        # Tier 1: In-process lookup (FastAPI server context to avoid deadlocks)
        if "backend.routes.embedding_route" in sys.modules:
            try:
                from backend.routes.embedding_route import _transformer
                if _transformer is not None and hasattr(_transformer, "model"):
                    embs = _transformer.model.encode(texts, **kwargs)
                    return embs[0] if is_single else embs
            except Exception as e:
                log_service(logger, f"In-process lookup failed: {e}", "debug")

        # Tier 2: HTTP POST query to local /embed API endpoint
        url = EMBEDDING_API_URL
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps({"texts": texts}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=3.0) as r:
                res = json.loads(r.read().decode("utf-8"))
                if res.get("success") and "embeddings" in res:
                    embs = [np.array(emb, dtype=np.float32) for emb in res["embeddings"]]
                    if is_single:
                        return embs[0]
                    return np.array(embs, dtype=np.float32)
        except Exception as e:
            log_service(logger, f"HTTP API call to {url} failed: {e}. Falling back...", "warning")

        # Tier 3: Local Model Fallback (SentenceTransformer)
        if self.local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                class DummyEmbeddingModel:
                    def encode(self, input_texts, **kwargs):
                        single_input = isinstance(input_texts, str)
                        inputs = [input_texts] if single_input else list(input_texts)
                        results = []
                        for txt in inputs:
                            rng = np.random.RandomState(sum(ord(c) for c in txt))
                            v = rng.randn(384)
                            results.append(v / np.linalg.norm(v))
                        if single_input:
                            return results[0]
                        return np.array(results, dtype=np.float32)
                self.local_model = DummyEmbeddingModel()

        return self.local_model.encode(text if is_single else texts, **kwargs)

    def get_embeddings(self, texts):
        """
        Batch generate embeddings (returns a list of numpy arrays or 2D array).
        """
        embs = self.encode(texts)
        if isinstance(embs, np.ndarray) and len(embs.shape) == 2:
            return [row for row in embs]
        return embs

_embedding_service_cache = None

def get_embedding_service() -> LLMEmbeddingService:
    global _embedding_service_cache
    if _embedding_service_cache is None:
        _embedding_service_cache = LLMEmbeddingService()
    return _embedding_service_cache
