import json
import time
import threading
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from module_loom.config.env_config import GENAI_API_KEY, GENAI_URL, GENAI_MODEL, setup_logger

logger = setup_logger("LLMCompletionService")

# After this many consecutive transport failures the service disables itself
# for COOLDOWN_SEC so a dead endpoint can't stall every sleep cycle / readout
# behind a full timeout each time. A single success resets the counter.
FAILURE_TRIP_THRESHOLD = 3
COOLDOWN_SEC = 300.0


def _chat_completions_url(base: str) -> str:
    """Normalizes GENAI_URL into a full /v1/chat/completions endpoint."""
    base = (base or "").strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/chat/completions"):
        return base
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base + "/chat/completions"


class LLMCompletionService:
    """
    Minimal OpenAI-compatible chat-completions client for the custom GenAI
    proxy (stdlib-only, mirroring LLMEmbeddingService so the substrate never
    gains a heavyweight dependency on its cognition path).

    complete() NEVER raises into the caller — it returns None on any failure,
    so cognitive machinery (sleep-cycle REM synthesis, benchmark readout) can
    always fall back to its deterministic path. Failures are logged loudly.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "url"):
            return
        self.url = _chat_completions_url(GENAI_URL)
        self.model = GENAI_MODEL
        self.api_key = GENAI_API_KEY
        self.timeout = 120.0
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._disabled_until = 0.0
        self.stats: Dict[str, Any] = {
            "calls": 0, "failures": 0,
            "total_latency_s": 0.0,
            "prompt_tokens": 0, "completion_tokens": 0,
        }

    def available(self) -> bool:
        if not (self.url and self.api_key and self.model):
            return False
        return time.time() >= self._disabled_until

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.4,
        max_retries: int = 4,
    ) -> Optional[str]:
        """
        One chat completion. Returns assistant text, or None on any failure.
        Rate-limit (HTTP 429) and transient 5xx responses are RETRYABLE — they
        mean "slow down / try again," not "endpoint is dead," so they back off
        and retry and do NOT count toward the circuit breaker. Only genuine
        transport death (connection refused, timeout, DNS) trips the breaker.
        """
        if not self.available():
            return None
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "GENAI_KEY": self.api_key,  # older proxy generations authenticate via this header
        }
        start = time.time()
        backoff = 1.0
        for attempt in range(max_retries + 1):
            req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    res = json.loads(r.read().decode("utf-8"))
                text = res["choices"][0]["message"]["content"]
                elapsed = time.time() - start
                usage = res.get("usage") or {}
                with self._lock:
                    self._consecutive_failures = 0
                    self.stats["calls"] += 1
                    self.stats["total_latency_s"] += elapsed
                    self.stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
                    self.stats["completion_tokens"] += int(usage.get("completion_tokens", 0))
                return text
            except urllib.error.HTTPError as e:
                retryable = e.code == 429 or 500 <= e.code < 600
                if retryable and attempt < max_retries:
                    # Honor Retry-After when present, else exponential backoff.
                    ra = e.headers.get("Retry-After") if e.headers else None
                    try:
                        wait = float(ra) if ra else backoff
                    except ValueError:
                        wait = backoff
                    time.sleep(min(wait, 30.0))
                    backoff = min(backoff * 2, 30.0)
                    continue
                self._record_transport_failure(start, f"HTTP {e.code}")
                return None
            except Exception as e:
                self._record_transport_failure(start, f"{type(e).__name__}: {e}")
                return None
        # Exhausted retries on a retryable error — count once, don't trip breaker.
        with self._lock:
            self.stats["calls"] += 1
            self.stats["failures"] += 1
        logger.warning(f"Completion still rate-limited/unavailable after {max_retries} retries.")
        return None

    def _record_transport_failure(self, start: float, why: str) -> None:
        elapsed = time.time() - start
        with self._lock:
            self.stats["calls"] += 1
            self.stats["failures"] += 1
            self._consecutive_failures += 1
            tripped = self._consecutive_failures >= FAILURE_TRIP_THRESHOLD
            if tripped:
                self._disabled_until = time.time() + COOLDOWN_SEC
        logger.warning(
            f"Completion transport failure after {elapsed:.1f}s ({why})"
            + (f" — disabling provider for {COOLDOWN_SEC:.0f}s after "
               f"{self._consecutive_failures} consecutive failures" if tripped else "")
        )


_completion_service_cache = None


def get_completion_service() -> LLMCompletionService:
    global _completion_service_cache
    if _completion_service_cache is None:
        _completion_service_cache = LLMCompletionService()
    return _completion_service_cache
