"""
Generates fluent short stories with TinyStories-8M (a small, already-trained
model — used here purely as a local text generator, NOT for weight transfer;
see module_AI/ROADMAP.md for why we stayed Loom-native architecture instead
of adopting GPT-Neo's design). The generated text is then ingested into Loom
through the REAL ingest_shard() pipeline — same physics simulation as
everything else — so it becomes a genuine Loom memory with real
activation/energy/Kuramoto-phase dynamics, not a synthetic bypass like
SQuAD's neutral placeholder physics fields.

Usage:
    python -m module_AI.scripts.generate_tinystories_data --count 2000 \
        --brain-dir module_AI/data/stress_brain
"""
import argparse
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PROMPTS = [
    "Once upon a time", "One day", "In a small village", "Long ago",
    "There was a little", "It was a sunny day when", "Deep in the forest",
    "Every morning", "On a cold winter night", "Near the river",
]


def generate_stories(count: int, max_new_tokens: int = 80, batch_size: int = 16):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading TinyStories-8M on {device}...")
    tok = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-8M").to(device)
    model.eval()
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only models need left-padding for correct batched generation

    stories = []
    t0 = time.perf_counter()
    while len(stories) < count:
        batch_prompts = [PROMPTS[i % len(PROMPTS)] for i in range(len(stories), len(stories) + batch_size)]
        inputs = tok(batch_prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                  temperature=0.9, top_p=0.95, pad_token_id=tok.eos_token_id)
        for ids in out:
            text = tok.decode(ids, skip_special_tokens=True).strip()
            if len(text.split()) >= 8:
                stories.append(text)
        if len(stories) % 160 == 0:
            print(f"  generated {len(stories)}/{count} ({time.perf_counter()-t0:.0f}s)")
    return stories[:count]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--brain-dir", type=str,
                     default=os.path.join(PROJECT_ROOT, "module_AI", "data", "stress_brain"))
    ap.add_argument("--batch-size", type=int, default=64, help="Loom ingest embedding batch size")
    args = ap.parse_args()

    stories = generate_stories(args.count)
    print(f"Generated {len(stories)} stories, ingesting into Loom (real physics pipeline)...")

    from module_loom.services.weaver.weaver_coordinator import WeaveBrainCoordinator
    from module_AI.memory_bridge import LoomMemory

    os.makedirs(args.brain_dir, exist_ok=True)
    coordinator = WeaveBrainCoordinator(storage_dir=args.brain_dir)
    memory = LoomMemory(coordinator=coordinator)

    t0 = time.perf_counter()
    ingested = 0
    batch = []
    for story in stories:
        batch.append(story)
        if len(batch) >= args.batch_size:
            vectors = memory.embed_batch(batch)
            for text, vec in zip(batch, vectors):
                memory.remember(text, metadata={"kind": "corpus", "source": "tinystories"}, vector=vec)
            ingested += len(batch)
            batch = []
            if ingested % (args.batch_size * 5) == 0:
                elapsed = time.perf_counter() - t0
                print(f"  ingested {ingested}/{len(stories)} ({ingested/elapsed:.1f} shards/s)")
    if batch:
        vectors = memory.embed_batch(batch)
        for text, vec in zip(batch, vectors):
            memory.remember(text, metadata={"kind": "corpus", "source": "tinystories"}, vector=vec)
        ingested += len(batch)

    coordinator.checkpoint()
    coordinator.close()
    print(f"Done. Ingested {ingested} TinyStories-generated shards into {args.brain_dir}.")


if __name__ == "__main__":
    main()
