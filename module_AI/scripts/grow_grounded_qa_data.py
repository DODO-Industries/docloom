"""
Generates REAL Loom/Qwen training examples that are actually grounded in
Loom's own content -- fixes a real problem found by testing: the old 30
hardcoded trivia questions (Eiffel Tower, capitals, etc.) have nothing to do
with what's actually stored in Loom (story text), so recall returned
irrelevant matches and the model was training on noise (learning "ignore
memory", the opposite of the goal).

For each real Loom shard sampled: ask Qwen to write ONE question that shard's
text actually answers, then run that question through the real /ai/ask
pipeline (same as module_AI/scripts/grow_stage_b_data.py) -- recall should
now surface that same shard as a top match, and the answer genuinely depends
on it. /ai/ask already logs every call to interactions.jsonl (Stage B's real
training data) for free.

Usage:
    python -m module_AI.scripts.grow_grounded_qa_data --time-limit-s 14400 --workers 8
"""
import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from module_AI.llm_client import LMStudioClient

SERVER = "http://localhost:8001"
QUESTION_PROMPT_SYSTEM = (
    "You write exactly one short, specific factual question that the given "
    "passage directly answers. Output ONLY the question, nothing else."
)


def sample_shards(brain_dir: str, count: int, exclude_clusters=(), min_words: int = 15):
    from module_loom.services.decoder.brain_decoder_service import decode_cluster

    cluster_names = sorted(
        f for f in os.listdir(brain_dir)
        if f.endswith(".loom") and f != "universe.loom" and f not in set(exclude_clusters)
    )
    texts = []
    per_cluster = max(1, count // max(1, len(cluster_names)))
    for name in cluster_names:
        decoded = decode_cluster(brain_dir, name, include_vectors=False, limit=per_cluster * 3)
        long_enough = [s["text"] for s in decoded["shards"] if len(s.get("text", "").split()) >= min_words]
        texts.extend(random.Random(42).sample(long_enough, min(per_cluster, len(long_enough))))
    random.Random(7).shuffle(texts)
    return texts[:count]


def make_question(llm: LMStudioClient, passage: str) -> str:
    q = llm.chat(QUESTION_PROMPT_SYSTEM, passage[:600], max_tokens=64, temperature=0.4)
    return q.strip().strip('"')


def ask_pipeline(question: str) -> None:
    r = requests.post(f"{SERVER}/ai/ask", json={"query": question, "remember": False}, timeout=90)
    r.raise_for_status()


def process_one(llm: LMStudioClient, passage: str) -> str:
    question = make_question(llm, passage)
    ask_pipeline(question)
    return question


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain-dir", type=str,
                     default=os.path.join(PROJECT_ROOT, "module_AI", "data", "stress_brain"))
    ap.add_argument("--exclude-clusters", type=str, default="crystal_1_E.loom")
    ap.add_argument("--time-limit-s", type=float, default=14400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample-size", type=int, default=4000,
                     help="how many shards to draw questions from (time-limit-s stops it earlier if hit first)")
    args = ap.parse_args()

    llm = LMStudioClient()
    if not llm.is_reachable():
        print("LM Studio not reachable at", llm.base_url, "-- start it and retry.")
        return

    passages = sample_shards(args.brain_dir, args.sample_size,
                              exclude_clusters=[c.strip() for c in args.exclude_clusters.split(",") if c.strip()])
    print(f"Sampled {len(passages)} passages to generate grounded questions from.")

    t0 = time.perf_counter()
    done, failed = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, llm, p): p for p in passages}
        for fut in as_completed(futures):
            if time.perf_counter() - t0 > args.time_limit_s:
                print("Time limit reached, stopping (remaining futures will still finish in background threads).")
                break
            try:
                q = fut.result()
                done += 1
                if done % 10 == 0:
                    elapsed = time.perf_counter() - t0
                    print(f"done {done}/{len(passages)} ({elapsed:.0f}s, {done/elapsed:.2f}/s) -- last: {q!r}")
            except Exception as e:
                failed += 1
                if failed % 10 == 0:
                    print(f"failed so far: {failed} (last error: {e})")

    print(f"\nFinished. {done} succeeded, {failed} failed, {time.perf_counter()-t0:.0f}s elapsed.")


if __name__ == "__main__":
    main()
