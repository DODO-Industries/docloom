"""
Stage A — self-supervised pretraining data, built entirely from Loom's own
already-ingested shards. Zero LLM calls: reads real stored text + real
physics state via the existing read-only decoder
(module_loom.services.decoder.brain_decoder_service.decode_cluster).

Objective: given a few preceding shards as memory tokens (their content vector
+ physics state, in real ingest order), predict the next shard's text. This
is plain self-supervised language modeling, but exercised through the same
memory-token mechanism Stage B (distillation) uses — so it teaches the model
to actually read memory tokens before it ever sees a question.
"""
import os
import sys
from typing import List, Optional

from torch.utils.data import Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from module_loom.services.decoder.brain_decoder_service import decode_cluster
from module_AI.model.config import DocLoomModelConfig
from module_AI.model.tokenizer import encode

PHYSICS_KEYS = ["activation", "energy", "stability", "resonance", "momentum", "attention"]
# Note: decode_cluster's meta doesn't carry "phase_angle" (that lives in the
# live ram_ledger, not frozen ingest metadata) — pack_shard_features() already
# defaults missing keys to 0.0, so this is a harmless, known gap, not a bug.


def _shard_to_dict(row: dict) -> dict:
    d = {"text": row.get("text", ""), "vector": row.get("vector")}
    for k in PHYSICS_KEYS:
        d[k] = row.get(k, 0.0)
    return d


class PretrainDataset(Dataset):
    def __init__(self, brain_dir: str, cfg: DocLoomModelConfig, cluster_names: Optional[List[str]] = None,
                 limit: int = 20000, min_words: int = 5, exclude_clusters: Optional[List[str]] = None):
        if cluster_names is None:
            # Auto-discover every crystal file in the brain dir — a corpus this
            # size splits into multiple crystals (see WeaveBrainCoordinator's
            # cellular division), and reading only "crystal_1.loom" silently
            # drops everything ingested after the first split.
            cluster_names = sorted(f for f in os.listdir(brain_dir) if f.endswith(".loom") and f != "universe.loom")
        if exclude_clusters:
            cluster_names = [c for c in cluster_names if c not in set(exclude_clusters)]

        shards: List[dict] = []
        per_cluster_limit = max(1, limit // max(1, len(cluster_names)))
        for name in cluster_names:
            decoded = decode_cluster(brain_dir, name, include_vectors=True, limit=per_cluster_limit)
            cluster_shards = [s for s in decoded["shards"] if len(s.get("text", "").split()) >= min_words]
            print(f"Loaded {len(cluster_shards)} shards from {decoded['cluster']} ({decoded['shard_count']} total on disk)")
            shards.extend(cluster_shards)

        self.examples = []
        n_mem = cfg.n_memory_tokens
        for i in range(n_mem, len(shards)):
            memory_shards = [_shard_to_dict(shards[j]) for j in range(i - n_mem, i)]
            target_ids = encode(shards[i]["text"])
            if not target_ids:
                continue
            max_target_len = cfg.block_size - n_mem
            self.examples.append({
                "excited_shards": memory_shards,
                "ids": target_ids[:max_target_len],
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def chat_format_examples(texts: List[str], cfg: DocLoomModelConfig) -> List[dict]:
    """Turns a list of already chat-formatted strings (see
    module_AI.model.chat_format) into the same {excited_shards, ids} shape
    PretrainDataset produces, WITHOUT writing them into Loom -- lets us
    control exactly how much chat-format data Stage A sees, separate from
    whatever's actually stored in the brain."""
    examples = []
    for text in texts:
        ids = encode(text)
        if not ids:
            continue
        examples.append({"excited_shards": [], "ids": ids[: cfg.block_size - cfg.n_memory_tokens]})
    return examples
