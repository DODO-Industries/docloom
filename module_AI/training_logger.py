"""
Roadmap Phase 1.3: log (excited shards + physics state, question, answer)
triples on every real interaction — the exact dataset Phase 3's distillation
training needs. Costs almost nothing (one JSON line append) and every call
made without it running is training data lost for good.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional

LOG_DIR = os.path.join(os.path.dirname(__file__), "data", "training_log")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "interactions.jsonl")


def log_interaction(
    query: str,
    excited_shards: List[Dict[str, Any]],
    answer: str,
    used_memory: bool,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    excited_shards: recall() results enriched with physics fields, e.g.
        [{"shard_id": ..., "text": ..., "score": ..., "activation": ...,
          "energy": ..., "phase_angle": ..., "momentum": ..., "stability": ...,
          "resonance": ..., "attention": ..., "hits": ...}, ...]
    """
    record = {
        "ts": time.time(),
        "query": query,
        "excited_shards": excited_shards,
        "answer": answer,
        "used_memory": used_memory,
    }
    if extra:
        record["extra"] = extra
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def count_logged() -> int:
    if not os.path.exists(LOG_PATH):
        return 0
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)
