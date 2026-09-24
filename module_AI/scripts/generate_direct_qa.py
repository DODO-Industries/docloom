"""
Generates direct factual Q&A pairs with explicit Thinking/Evidence steps.
Eliminates meta-disclaimers ("The memory context mentions...") by training
the model to cite the exact fact inside <|think|> and output the crisp answer
inside <|answer|>.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from module_AI.memory_bridge import LoomMemory
from module_loom.services.decoder.brain_decoder_service import decode_cluster

OUT_PATH = os.path.join(PROJECT_ROOT, "module_AI", "data", "training_log", "direct_grounded_qa.jsonl")
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"


def ask_lm_studio(text: str, timeout: int = 20) -> Optional[Dict[str, str]]:
    prompt = (
        f"Story snippet:\n\"\"\"{text.strip()}\"\"\"\n\n"
        "Generate 1 specific factual question directly answered in the snippet, "
        "the exact evidence quote, and a direct factual answer (NO meta-disclaimers, "
        "NO 'the context says', just the clean answer).\n"
        "Respond ONLY with a JSON object in this format:\n"
        "{\"question\": \"...\", \"thought\": \"Evidence: ...\", \"answer\": \"...\"}"
    )
    req_data = json.dumps({
        "model": "qwen/qwen3-1.7b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 450
    }).encode("utf-8")

    req = urllib.request.Request(LM_STUDIO_URL, data=req_data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            reasoning = (msg.get("reasoning_content") or "").strip()
            combined = content if content else reasoning
            # Clean markdown codeblocks
            combined = combined.replace("```json", "").replace("```", "").strip()
            # Extract JSON substring
            start = combined.find("{")
            end = combined.rfind("}")
            if start != -1 and end != -1:
                return json.loads(combined[start:end+1])
    except Exception:
        return None
    return None




def collect_shards_from_brain(brain_dir: str, limit: int = 2000) -> List[Dict[str, Any]]:
    shards = []
    cluster_names = sorted(f for f in os.listdir(brain_dir) if f.endswith(".loom") and f != "universe.loom")
    for name in cluster_names:
        decoded = decode_cluster(brain_dir, name, include_vectors=True, limit=limit)
        for s in decoded["shards"]:
            text = s.get("text", "")
            if text and len(text.split()) >= 15:
                shards.append({"shard_id": s.get("shard_id"), "text": text})
                if len(shards) >= limit:
                    return shards
    return shards



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain-dir", type=str,
                    default=os.path.join(PROJECT_ROOT, "assets", ".brain_data"))
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    print(f"Collecting up to {args.limit} shards from {args.brain_dir}...")
    shards = collect_shards_from_brain(args.brain_dir, limit=args.limit)
    print(f"Loaded {len(shards)} candidates.")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    mem = LoomMemory()

    records_written = 0
    t0 = time.time()

    def process_shard(shard):
        res = ask_lm_studio(shard["text"])
        if not res or "question" not in res or "answer" not in res:
            return None
        q = res["question"].strip()
        a = res["answer"].strip()
        thought = res.get("thought", "").strip()
        if not q or not a:
            return None
        # Retrieve living physics features using distilled recall
        matches = mem.recall_distilled(q, candidate_pool=16, top_k=8, learn=True)
        excited_shards = [
            {"shard_id": m.get("shard_id"), "text": m.get("text"), "score": m.get("score"),
             "vector": mem.embed(m.get("text", "")).tolist(),
             **mem.get_shard_physics(m.get("shard_id"))}
            for m in matches
        ]
        return {
            "ts": time.time(),
            "query": q,
            "thought": thought,
            "answer": a,
            "excited_shards": excited_shards,
            "used_memory": True
        }

    with open(OUT_PATH, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_shard, s): s for s in shards}
            for f in as_completed(futures):
                rec = f.result()
                if rec:
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()
                    records_written += 1
                    if records_written % 20 == 0:
                        elapsed = time.time() - t0
                        rate = records_written / max(1, elapsed)
                        print(f"Generated {records_written} direct QA records ({rate:.1f} rec/s)...")

    mem.coordinator.close()
    print(f"Done! Saved {records_written} direct QA records to {OUT_PATH}")


if __name__ == "__main__":
    main()
