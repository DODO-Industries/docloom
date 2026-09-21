"""
Real batching for training — replaces the old batch_size=1 loop. Padding uses
tiktoken's <|endoftext|> id (50256) as the pad token (a standard, common trick
— gpt2's tokenizer has no dedicated pad token). Padded positions are excluded
from the loss via ignore_index=-100, and from attention via a key-padding
mask (module_AI/model/backbone.py), so padding never influences learning.
"""
import random
from typing import List, Optional

import torch

PAD_ID = 50256  # tiktoken gpt2 <|endoftext|>
IGNORE_INDEX = -100


def collate_batch(batch: List[dict], device: str = "cpu"):
    """
    batch: list of {"excited_shards": [...], "ids": [int, ...], "q_len": int (optional)}
    q_len marks where the answer starts (module_AI/model/train.py's Stage B) —
    tokens before it are excluded from the loss. Absent/0 means "loss over the
    whole sequence" (module_AI/model/pretrain.py's Stage A).

    Returns: batch_excited_shards, input_ids, target_ids (with -100 at ignored
    positions), attention_mask (for input_ids' padding).
    """
    max_len = max(len(ex["ids"]) for ex in batch)
    input_ids_list, target_ids_list, attn_list = [], [], []

    for ex in batch:
        ids = ex["ids"]
        q_len = ex.get("q_len", 0)
        real_len = len(ids)
        padded = ids + [PAD_ID] * (max_len - real_len)

        inp = padded[:-1]
        tgt = padded[1:]
        # A target position is valid loss iff: (a) it's not padding, and
        # (b) it's not part of the question (for Stage B; q_len=0 means "no
        # question, everything's valid" for Stage A).
        valid = [
            (1 if (i + 1) < real_len and (i + 1) >= max(0, q_len - 1) else 0)
            for i in range(len(tgt))
        ]
        tgt_masked = [t if v else IGNORE_INDEX for t, v in zip(tgt, valid)]
        attn = [1 if i < real_len - 1 else 0 for i in range(len(inp))]

        input_ids_list.append(inp)
        target_ids_list.append(tgt_masked)
        attn_list.append(attn)

    batch_excited_shards = [ex["excited_shards"] for ex in batch]
    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
    target_ids = torch.tensor(target_ids_list, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attn_list, dtype=torch.bool, device=device)
    return batch_excited_shards, input_ids, target_ids, attention_mask


def make_length_bucketed_batches(examples: List[dict], batch_size: int, seed: int = 0) -> List[List[dict]]:
    """
    Groups examples into batches of SIMILAR length before padding, instead of
    random composition. Random batching mixes short and long sequences, so
    every batch pads up to its longest member — and attention cost scales
    with sequence length squared, so that padding waste is expensive, not
    just wasteful. Sorting by length first keeps each batch's padding close
    to zero. Batch ORDER is still shuffled each call, so training isn't
    length-ordered (short examples first, then long) — only each batch's
    *contents* are length-similar.
    """
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["ids"]))
    batches = [
        [examples[i] for i in order[j: j + batch_size]]
        for j in range(0, len(order), batch_size)
    ]
    random.Random(seed).shuffle(batches)
    return batches


def lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay to ~10% of peak LR — standard recipe,
    replaces the old flat learning rate."""
    import math
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
