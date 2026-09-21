import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from module_loom.config.env_config import LOCAL_LM_HOST

# LM Studio (OpenAI-compatible local server)
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", LOCAL_LM_HOST)
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-1.7b")
LM_STUDIO_TIMEOUT = float(os.getenv("LM_STUDIO_TIMEOUT", "120"))

# Retrieval behavior. NOTE: recall()'s "score" is an unnormalized composite of
# cosine relevance + gravity/wave physics terms, not a plain [-1,1] similarity —
# 12.0 is an empirically-picked floor (exact-match ~20, unrelated content ~10-11
# on the all-MiniLM-L6-v2 + 128-d test corpus). Re-tune as the corpus grows.
AI_RECALL_TOP_K = int(os.getenv("AI_RECALL_TOP_K", "5"))
AI_RECALL_MIN_SIMILARITY = float(os.getenv("AI_RECALL_MIN_SIMILARITY", "12.0"))

# Cap per-shard snippet length in the prompt — keeps total prompt size
# predictable regardless of how long a recalled passage happens to be, so it
# fits LM Studio's per-slot context budget (context_length / Parallel slots).
AI_CONTEXT_SNIPPET_CHARS = int(os.getenv("AI_CONTEXT_SNIPPET_CHARS", "220"))
