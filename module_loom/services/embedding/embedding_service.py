import os
import sys
import json
import socket
import urllib.request
import urllib.parse
import numpy as np
from module_loom.config.env_config import EMBEDDING_API_URL, setup_logger, log_service

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
        # Circuit breaker for the HTTP tier: after this many consecutive
        # connection failures, stop attempting the remote /embed endpoint for
        # COOLDOWN seconds. Without this, a not-running embed server costs a
        # full socket timeout on EVERY encode() call — which turns bulk
        # ingestion (thousands of shards) into hours of dead waiting.
        self._http_consecutive_failures = 0
        self._http_disabled_until = 0.0
        self._HTTP_TRIP = 2
        self._HTTP_COOLDOWN = 120.0
        self._HTTP_TIMEOUT = 2.0
        # Even with the circuit breaker, the FIRST call (and the first call
        # after each cooldown expires) still pays a full request timeout on a
        # dead endpoint before falling back. A cheap TCP connect pre-probe with
        # a short timeout short-circuits that: a not-running server is refused
        # or times out in _HTTP_CONNECT_TIMEOUT, not _HTTP_TIMEOUT. Set
        # EMBEDDING_HTTP_DISABLED=1 to skip the HTTP tier entirely (dev boxes
        # with no embed server — go straight to the local model).
        self._HTTP_CONNECT_TIMEOUT = 0.25
        self._http_hard_disabled = os.getenv("EMBEDDING_HTTP_DISABLED", "0") == "1"
        self._http_ever_ok = False  # set once a POST succeeds -> stop probing a known-up server

    def _endpoint_reachable(self, url):
        """Fast TCP reachability probe so a dead endpoint costs
        ~_HTTP_CONNECT_TIMEOUT, not a full request timeout, on the first call."""
        try:
            parsed = urllib.parse.urlparse(url)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            if not host:
                return False
            with socket.create_connection((host, port), timeout=self._HTTP_CONNECT_TIMEOUT):
                return True
        except Exception:
            return False

    def encode(self, text, **kwargs):
        """
        Encode text(s) into embeddings.
        Supports single string or list of strings.
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else list(text)

        # Tier 1: In-process singleton lookup (Fast, zero-network, prevents self-HTTP deadlocks)
        try:
            from module_loom.services.embedding.transformer import EmbeddingTransformer
            transformer = EmbeddingTransformer()
            embs = transformer.model.encode(texts, **kwargs)
            return embs[0] if is_single else embs
        except Exception as e:
            log_service(logger, f"In-process transformer lookup note: {e}", "debug")

        # Tier 2: HTTP POST query to local /embed API endpoint (circuit-broken)
        import time as _time
        url = EMBEDDING_API_URL
        if not self._http_hard_disabled and _time.time() >= self._http_disabled_until:
            # Fast reachability probe first: a dead endpoint fails here in
            # ~_HTTP_CONNECT_TIMEOUT and counts toward the breaker, so a bulk
            # load doesn't re-probe (or worse, re-timeout) on every item. Skip
            # the probe once a POST has actually succeeded — no point paying an
            # extra TCP connect per call against a server we know is up.
            if not self._http_ever_ok and not self._endpoint_reachable(url):
                self._http_consecutive_failures += 1
                if self._http_consecutive_failures >= self._HTTP_TRIP:
                    self._http_disabled_until = _time.time() + self._HTTP_COOLDOWN
                    log_service(logger, f"HTTP embed endpoint {url} unreachable "
                                        f"({self._http_consecutive_failures}x, TCP probe) — pausing that "
                                        f"tier for {self._HTTP_COOLDOWN:.0f}s, using local model.", "warning")
                url = None  # skip the POST below, fall through to local model
        else:
            url = None
        if url is not None:
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps({"texts": texts}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=self._HTTP_TIMEOUT) as r:
                    res = json.loads(r.read().decode("utf-8"))
                    if res.get("success") and "embeddings" in res:
                        self._http_consecutive_failures = 0
                        self._http_ever_ok = True
                        embs = [np.array(emb, dtype=np.float32) for emb in res["embeddings"]]
                        if is_single:
                            return embs[0]
                        return np.array(embs, dtype=np.float32)
            except Exception as e:
                self._http_consecutive_failures += 1
                if self._http_consecutive_failures >= self._HTTP_TRIP:
                    self._http_disabled_until = _time.time() + self._HTTP_COOLDOWN
                    log_service(logger, f"HTTP embed endpoint {url} unreachable "
                                        f"({self._http_consecutive_failures}x) — pausing that tier "
                                        f"for {self._HTTP_COOLDOWN:.0f}s, using local model.", "warning")
                else:
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
