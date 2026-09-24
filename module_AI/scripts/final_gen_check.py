"""
Final sanity check after the overnight pipeline — loads the latest Stage B
checkpoint, runs a few real questions through real Loom recall (not fake
placeholder memory), and writes results to a plain text file for review.
"""
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from module_AI.model.checkpoint import load_checkpoint
from module_AI.model.docloom_model import DocLoomModel
from module_AI.model.config import DocLoomModelConfig
from module_AI.memory_bridge import LoomMemory
from module_AI.model.chat_format import format_prompt

CKPT = os.path.join(PROJECT_ROOT, "module_AI", "data", "checkpoints", "docloom_stageB.safetensors")
OUT_PATH = os.path.join(PROJECT_ROOT, "module_AI", "benchmarks", "overnight_final_generations.txt")

import torch

QUESTIONS = [
    # Grounded questions directly matching Loom's memory:
    "What did the otter do after seeing the rock?",
    "What did the dragon see when he opened his eyes?",
    "Why did the little bunny stop jumping?",
    "What did Lily do to make the snowman?",
    # General / fallback questions:
    "Where is the Eiffel Tower?",
    "What is gravity?",
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint {CKPT} on {device}...")
    model = load_checkpoint(DocLoomModel, DocLoomModelConfig, CKPT, device=device).to(device)
    model.eval()

    mem = LoomMemory()
    lines = []
    for q in QUESTIONS:
        # Upgrade A: Multi-Pass Physics Distilled Recall
        matches = mem.recall_distilled(q, candidate_pool=16, top_k=8)
        excited_shards = [
            {"text": m.get("text", ""), "score": m.get("score"),
             "vector": mem.embed(m.get("text", "")).tolist(),
             **mem.get_shard_physics(m.get("shard_id"))}
            for m in matches
        ]
        lead_snippet = matches[0].get("text", "") if matches else ""

        # Test Mode 1: Pure Latent Memory Tokens (No text context in prompt)
        prompt_pure = format_prompt(q)
        out_pure = model.generate(
            excited_shards, prompt_pure, max_new_tokens=100, temperature=0.6,
            top_k=40, top_p=0.9, repetition_penalty=1.15, return_full_text=False
        )

        # Test Mode 2: Hybrid Memory Tokens + Full Distilled Context Snippet
        context_snippets = [lead_snippet] if lead_snippet else []
        prompt_hybrid = format_prompt(q, context_snippets=context_snippets)
        out_hybrid = model.generate(
            excited_shards, prompt_hybrid, max_new_tokens=100, temperature=0.5,
            top_k=40, top_p=0.9, repetition_penalty=1.15, return_full_text=False
        )


        line = (
            f"============================================================\n"
            f"Q: {q}\n"
            f"  [Distilled Shards]: {len(matches)} (lead: {lead_snippet[:65]!r}...)\n"
            f"  [Mode 1 - Pure Memory Tokens]:\n    -> {out_pure}\n"
            f"  [Mode 2 - Memory Tokens + Context Snippet]:\n    -> {out_hybrid}\n"
        )
        lines.append(line)
        print(line)
    mem.coordinator.close()


    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote results to {OUT_PATH}")



if __name__ == "__main__":
    main()
