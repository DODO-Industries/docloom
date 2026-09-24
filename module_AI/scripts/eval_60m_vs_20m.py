"""
Evaluation script comparing 20M vs 60M DocLoom models on multi-cluster
retrieval and grounded question answering.
"""
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from module_AI.model.checkpoint import load_checkpoint
from module_AI.model.docloom_model import DocLoomModel
from module_AI.model.config import DocLoomModelConfig
from module_AI.memory_bridge import LoomMemory
from module_AI.model.chat_format import format_prompt

CKPT_DIR = os.path.join(PROJECT_ROOT, "module_AI", "data", "checkpoints")
CKPT_20M = os.path.join(CKPT_DIR, "docloom_20m_stageB.safetensors")
CKPT_60M = os.path.join(CKPT_DIR, "docloom_60m_stageB.safetensors")

BENCHMARK_QUERIES = [
    {
        "query": "What did the dragon see when he opened his eyes?",
        "category": "Grounded Recall (Dragon Shard)",
        "cross_cluster": False
    },
    {
        "query": "What did the otter do after seeing the rock?",
        "category": "Grounded Recall (Otter Shard)",
        "cross_cluster": False
    },
    {
        "query": "Did the dragon fly over the river water?",
        "category": "Cross-Cluster Embassy (Dragon + River Nature)",
        "cross_cluster": True
    },
    {
        "query": "Where was the crystal key hidden inside the third vault?",
        "category": "Deep-Memory Fallback Sweep",
        "cross_cluster": True
    },
    {
        "query": "How did the young boy demonstrate his attitude toward people who are different?",
        "category": "Grounded Ethics / Value Alignment",
        "cross_cluster": False
    }
]


def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== DocLoom Benchmark: 20M vs 60M (Device: {device}) ===")

    mem = LoomMemory()
    models = {}

    if os.path.exists(CKPT_20M):
        print(f"Loading 20M model: {CKPT_20M}")
        models["20M"] = load_checkpoint(DocLoomModel, DocLoomModelConfig, CKPT_20M, device=device).to(device)
        models["20M"].eval()
    else:
        print(f"Warning: 20M checkpoint not found at {CKPT_20M}")

    if os.path.exists(CKPT_60M):
        print(f"Loading 60M model: {CKPT_60M}")
        models["60M"] = load_checkpoint(DocLoomModel, DocLoomModelConfig, CKPT_60M, device=device).to(device)
        models["60M"].eval()
    else:
        print(f"Warning: 60M checkpoint not found at {CKPT_60M}")

    results = []

    for item in BENCHMARK_QUERIES:
        q = item["query"]
        cat = item["category"]
        print(f"\nEvaluating: '{q}' [{cat}]")

        t0 = time.time()
        matches = mem.recall_distilled(q, candidate_pool=16, top_k=8)
        recall_time_ms = (time.time() - t0) * 1000

        excited_shards = [
            {"text": m.get("text", ""), "score": m.get("score"),
             "vector": mem.embed(m.get("text", "")).tolist(),
             **mem.get_shard_physics(m.get("shard_id"))}
            for m in matches
        ]
        lead_snippet = matches[0].get("text", "") if matches else ""

        entry = {
            "query": q,
            "category": cat,
            "recall_ms": recall_time_ms,
            "retrieved_count": len(matches),
            "lead_snippet": lead_snippet[:80],
            "outputs": {}
        }

        prompt = format_prompt(q)

        for model_name, model in models.items():
            t_gen0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    excited_shards, prompt, max_new_tokens=90, temperature=0.6,
                    top_k=40, top_p=0.9, repetition_penalty=1.15, return_full_text=False
                )
            gen_time_ms = (time.time() - t_gen0) * 1000
            entry["outputs"][model_name] = {
                "text": out.strip(),
                "time_ms": gen_time_ms
            }
            print(f"  [{model_name}] ({gen_time_ms:.1f}ms): {out.strip()[:100]}...")

        results.append(entry)

    mem.coordinator.close()
    return results


if __name__ == "__main__":
    run_benchmark()
