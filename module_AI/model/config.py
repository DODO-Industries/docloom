from dataclasses import dataclass


@dataclass
class DocLoomModelConfig:
    """Roadmap Phase 3 prototype: ~15-30M params, GPT-2-style block, our own
    memory-token mechanism. See module_AI/ROADMAP.md, 'Model Architecture'."""

    # Stage 3 — backbone
    vocab_size: int = 50257       # tiktoken 'gpt2' encoding
    block_size: int = 256         # max total sequence length (memory + text tokens)
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1

    # Stage 1 — Loom memory projection
    n_memory_tokens: int = 4      # fixed number of excited-shard slots
    loom_content_dim: int = 128   # EMBEDDING_DIMENSION in module_loom's config
    loom_physics_dim: int = 7     # activation, energy, phase_angle, momentum, stability, resonance, attention
    memory_hidden_dim: int = 256

    # Stage 4 — adapters (LoRA)
    lora_rank: int = 8
    lora_alpha: int = 16

    @property
    def loom_shard_dim(self) -> int:
        return self.loom_content_dim + self.loom_physics_dim
