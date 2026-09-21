"""
Teaches Stage A the chat format (module_AI/model/chat_format.py) by writing
some question/answer pairs into Loom as plain shards, in our own tag format,
before Stage A pretraining runs. Root-cause fix for the Stage B collapse:
the frozen backbone never saw <|user|>/<|assistant|> structure at all, only
plain story text -- so it needs to see that structure here, in the same
self-supervised pretraining Stage A already does, not bolted on later by a
tiny Stage B adapter.

Reuses data we already collected (SQuAD + real Loom/Qwen interactions) --
no new LLM calls needed. Takes a modest SQuAD sample (not all 87k) so this
doesn't drown out the story-continuation shards already in the corpus.

Usage:
    python -m module_AI.scripts.ingest_chat_examples --brain-dir module_AI/data/stress_brain
"""
import argparse
import json
import os
import random
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from module_AI.model.chat_format import format_turn
from module_AI.training_logger import LOG_PATH

SQUAD_LOG_PATH = os.path.join(os.path.dirname(LOG_PATH), "squad_examples.jsonl")


def _load_examples(path: str, sample_size: int = None) -> list:
    if not os.path.exists(path):
        return []
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            examples.append((rec["query"], rec["answer"]))
    if sample_size is not None and len(examples) > sample_size:
        examples = random.Random(42).sample(examples, sample_size)
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain-dir", type=str,
                     default=os.path.join(PROJECT_ROOT, "module_AI", "data", "stress_brain"))
    ap.add_argument("--squad-sample-size", type=int, default=5000,
                     help="SQuAD has 87k examples -- use a modest sample so it teaches the "
                          "tag structure without drowning out story shards in the corpus")
    ap.add_argument("--ingest-batch-size", type=int, default=64)
    args = ap.parse_args()

    real_examples = _load_examples(LOG_PATH)
    squad_examples = _load_examples(SQUAD_LOG_PATH, sample_size=args.squad_sample_size)
    print(f"Real Loom/Qwen examples: {len(real_examples)}, SQuAD sample: {len(squad_examples)}")

    chat_texts = [format_turn(q, a) for q, a in real_examples + squad_examples]
    random.Random(42).shuffle(chat_texts)
    print(f"Built {len(chat_texts)} chat-formatted shards to ingest")

    from module_loom.services.weaver.weaver_coordinator import WeaveBrainCoordinator
    from module_AI.memory_bridge import LoomMemory

    os.makedirs(args.brain_dir, exist_ok=True)
    coordinator = WeaveBrainCoordinator(storage_dir=args.brain_dir)
    memory = LoomMemory(coordinator=coordinator)

    ingested = 0
    batch = []
    for text in chat_texts:
        batch.append(text)
        if len(batch) >= args.ingest_batch_size:
            vectors = memory.embed_batch(batch)
            for t, v in zip(batch, vectors):
                memory.remember(t, metadata={"kind": "corpus", "source": "chat_format"}, vector=v)
            ingested += len(batch)
            batch = []
            if ingested % (args.ingest_batch_size * 10) == 0:
                print(f"  ingested {ingested}/{len(chat_texts)}")
    if batch:
        vectors = memory.embed_batch(batch)
        for t, v in zip(batch, vectors):
            memory.remember(t, metadata={"kind": "corpus", "source": "chat_format"}, vector=v)
        ingested += len(batch)

    coordinator.checkpoint()
    coordinator.close()
    print(f"Done. Ingested {ingested} chat-formatted shards into {args.brain_dir}.")


if __name__ == "__main__":
    main()
