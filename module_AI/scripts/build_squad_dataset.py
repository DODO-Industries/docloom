"""
Downloads SQuAD (public, ~100k question/answer-from-passage examples) and
converts it into the exact same record schema as module_AI/training_logger.py's
interactions.jsonl — so train.py can read both files together. Teaches the
general "read provided context, answer from it" skill at scale, for free (no
LLM calls) — complements the smaller, slower Qwen-distilled Loom-specific data.

Each SQuAD passage becomes one "excited shard": real embedding (via our own
model, not Qwen), with neutral default physics fields (SQuAD has no Loom
ingestion history, so there's no real activation/energy/etc — defaults keep
the schema valid without fabricating fake dynamics).

Usage:
    python -m module_AI.scripts.build_squad_dataset --limit 2000
"""
import argparse
import json
import os
import sys

import numpy as np
import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from module_AI.model.config import DocLoomModelConfig

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
CACHE_PATH = os.path.join(PROJECT_ROOT, "module_AI", "data", "raw", "squad_train-v1.1.json")
OUT_PATH = os.path.join(PROJECT_ROOT, "module_AI", "data", "training_log", "squad_examples.jsonl")

NEUTRAL_PHYSICS = {
    "activation": 1.0, "energy": 0.5, "phase_angle": 0.0,
    "momentum": 0.0, "stability": 1.0, "resonance": 0.0, "attention": 0.0,
}


def download_squad() -> dict:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    if not os.path.exists(CACHE_PATH):
        print(f"Downloading SQuAD from {SQUAD_URL} ...")
        resp = requests.get(SQUAD_URL, timeout=60)
        resp.raise_for_status()
        with open(CACHE_PATH, "wb") as f:
            f.write(resp.content)
        print(f"Cached to {CACHE_PATH} ({len(resp.content)/1024/1024:.1f} MB)")
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_examples(squad_data: dict, limit: int):
    """One example per (context, question) — first answer only, first
    limit examples across the dataset (already shuffled-ish by topic order,
    good enough for a training set of this nature)."""
    examples = []
    for article in squad_data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if qa.get("is_impossible") or not qa.get("answers"):
                    continue
                examples.append({"context": context, "question": qa["question"], "answer": qa["answers"][0]["text"]})
                if len(examples) >= limit:
                    return examples
    return examples


def _align_vector(vec: np.ndarray, target_dim: int) -> np.ndarray:
    """Same alignment as module_AI/memory_bridge.py — SQuAD's raw sentence-
    transformer output is 384-dim, but the real Loom-derived vectors in
    interactions.jsonl are truncated to the coordinator's 128-dim working
    space. Mixing unaligned dims in one training batch breaks np.stack."""
    v = np.asarray(vec, dtype=np.float32)
    if len(v) > target_dim:
        v = v[:target_dim]
    elif len(v) < target_dim:
        v = np.pad(v, (0, target_dim - len(v)), constant_values=0.0)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--context-chars", type=int, default=220, help="matches AI_CONTEXT_SNIPPET_CHARS")
    ap.add_argument("--embed-batch-size", type=int, default=64)
    args = ap.parse_args()

    from module_loom.services.embedding.embedding_manager import get_embedding_model

    model_cfg = DocLoomModelConfig()
    squad_data = download_squad()
    examples = extract_examples(squad_data, args.limit)
    print(f"Extracted {len(examples)} (context, question, answer) triples")

    model = get_embedding_model()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    written = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i in range(0, len(examples), args.embed_batch_size):
            batch = examples[i: i + args.embed_batch_size]
            vectors = model.encode([ex["context"] for ex in batch])
            for ex, vec in zip(batch, vectors):
                snippet = ex["context"][:args.context_chars]
                aligned_vec = _align_vector(vec, model_cfg.loom_content_dim)
                record = {
                    "ts": 0,
                    "query": ex["question"],
                    "excited_shards": [{
                        "shard_id": "squad", "text": snippet, "score": 20.0,
                        "vector": aligned_vec.tolist(), **NEUTRAL_PHYSICS,
                    }],
                    "answer": ex["answer"],
                    "used_memory": True,
                    "extra": {"source": "squad"},
                }
                f.write(json.dumps(record) + "\n")
                written += 1
            if (i // args.embed_batch_size) % 10 == 0:
                print(f"  embedded {written}/{len(examples)}")

    print(f"Wrote {written} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
