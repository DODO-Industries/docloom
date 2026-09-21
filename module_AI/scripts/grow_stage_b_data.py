"""
Grows the real Loom/Qwen Stage B dataset by asking a batch of diverse
questions through the live /ai/ask pipeline. Extracted into a real script
(not an inline -c one-liner) after that one-liner failed with a SyntaxError —
`def` cannot follow other statements after a semicolon on the same line.

Usage:
    python -m module_AI.scripts.grow_stage_b_data
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

SERVER = "http://localhost:8001"

QUESTIONS = [
    "What is the tallest waterfall in the world?", "Who invented the printing press?",
    "What is the capital of Portugal?", "What is the boiling point of alcohol?",
    "Who wrote The Odyssey?", "What is the capital of Sweden?", "What is a black hole?",
    "What is the currency of India?", "Who painted Starry Night?", "What is the speed of light?",
    "What is the largest mammal?", "Who discovered penicillin?", "What is the capital of Norway?",
    "What is DNA made of?", "Who wrote The Great Gatsby?", "What is the freezing point of water?",
    "What is the capital of Greece?", "Who was Marie Curie?", "What is photosynthesis?",
    "What is the largest country by area?", "What is the capital of Japan?",
    "Who wrote Pride and Prejudice?", "What is the human heart?", "What is the capital of Egypt?",
    "Who was Albert Einstein?", "What is an ecosystem?", "What is the capital of Mexico?",
    "Who invented the light bulb?", "What is the largest planet?", "What is the capital of Turkey?",
]


def ask(q):
    r = requests.post(f"{SERVER}/ai/ask", json={"query": q, "remember": False}, timeout=90)
    r.raise_for_status()
    return q


def main():
    done, failed = 0, 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(ask, q): q for q in QUESTIONS}
        for fut in as_completed(futures):
            q = futures[fut]
            try:
                fut.result()
                done += 1
                print(f"done ({done}/{len(QUESTIONS)}): {q}")
            except Exception as e:
                failed += 1
                print(f"failed: {q} -> {e}")
    print(f"\nDone: {done} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()
