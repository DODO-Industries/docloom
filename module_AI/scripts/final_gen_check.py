"""
Final sanity check after the overnight pipeline — loads the latest Stage B
checkpoint, runs a few real questions through real Loom recall (not fake
placeholder memory), and writes results to a plain text file for review.
"""
import sys
import os

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

QUESTIONS = [
    "Where is the Eiffel Tower?",
    "What is the capital of Australia?",
    "Who wrote War and Peace?",
    "What is gravity?",
    "Who was Napoleon?",
]


def main():
    model = load_checkpoint(DocLoomModel, DocLoomModelConfig, CKPT)
    model.eval()

    mem = LoomMemory()
    lines = []
    for q in QUESTIONS:
        matches = mem.recall(q, top_k=5, learn=False)
        excited_shards = [
            {"text": m.get("text", ""), "score": m.get("score"),
             "vector": mem.embed(m.get("text", "")).tolist(),
             **mem.get_shard_physics(m.get("shard_id"))}
            for m in matches
        ]
        prompt = format_prompt(q)
        out = model.generate(excited_shards, prompt, max_new_tokens=30, temperature=0.6)
        lines.append(f"Q: {q}\n  matches_found: {len(matches)}\n  -> {out.encode('ascii','replace').decode()!r}\n")
        print(lines[-1])
    mem.coordinator.close()

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote results to {OUT_PATH}")


if __name__ == "__main__":
    main()
