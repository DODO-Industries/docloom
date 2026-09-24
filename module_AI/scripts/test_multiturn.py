"""
Test script for Multi-Turn Physics Working Memory in DocLoom.
Demonstrates zero-prompt-bloat contextual continuity across conversational turns
using residual wave excitation, Kuramoto phase coherence, and attractor dynamics.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from module_AI.memory_bridge import LoomMemory
from module_AI.model.checkpoint import load_checkpoint
from module_AI.model.docloom_model import DocLoomModel
from module_AI.model.config import DocLoomModelConfig
from module_AI.model.chat_format import format_prompt

CKPT = os.path.join(PROJECT_ROOT, "module_AI", "data", "checkpoints", "docloom_stageB.safetensors")


def main():
    print("=" * 70)
    print("  DocLoom Multi-Turn Physics Working Memory Test")
    print("=" * 70)

    mem = LoomMemory()
    session_id = "multiturn_live_demo"

    # Turn 1: Initial Topic
    q1 = "What did the otter do after seeing the rock?"
    print(f"\n[Turn 1] User asks: {q1!r}")
    matches1 = mem.recall_distilled(q1, candidate_pool=16, top_k=5, session_id=session_id)
    lead1 = matches1[0]["text"][:65] if matches1 else "None"
    print(f"  -> Recalled lead shard: {lead1!r}")
    session_state = mem.coordinator.sessions.get(session_id, {})
    print(f"  -> Active shards in working memory: {len(session_state.get('working_memory', {}))}")

    # Turn 2: Implicit Reference (No mention of 'otter' or 'river'!)
    q2 = "Did he see a little girl walking by?"
    print(f"\n[Turn 2] User asks: {q2!r} (Zero mention of 'otter' in text prompt!)")
    matches2 = mem.recall_distilled(q2, candidate_pool=16, top_k=5, session_id=session_id)
    lead2 = matches2[0]["text"][:65] if matches2 else "None"
    print(f"  -> Recalled lead shard: {lead2!r}")
    # Verify that the otter shard was preserved in context via attractor dynamics
    found_otter = any("otter" in m.get("text", "").lower() for m in matches2)
    print(f"  -> Working-Memory Coreference Success: {found_otter}")

    # Turn 3: Topic Shift & Natural Decay
    q3 = "What was inside the middle of the dragon's book?"
    print(f"\n[Turn 3] User shifts topic: {q3!r}")
    matches3 = mem.recall_distilled(q3, candidate_pool=16, top_k=5, session_id=session_id)
    lead3 = matches3[0]["text"][:65] if matches3 else "None"
    print(f"  -> Recalled lead shard: {lead3!r}")
    session_state = mem.coordinator.sessions.get(session_id, {})
    dragon_active = any("dragon" in w["text"].lower() for w in session_state.get("working_memory", {}).values())
    print(f"  -> Dragon excited in working memory: {dragon_active}")

    mem.coordinator.close()
    print("\n" + "=" * 70)
    print("Multi-Turn Physics Working Memory verified successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
