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
    n_memory_tokens: int = 8      # expanded memory slots (Upgrade B)
    loom_content_dim: int = 128   # EMBEDDING_DIMENSION in module_loom's config
    loom_physics_dim: int = 7     # activation, energy, phase_angle, momentum, stability, resonance, attention
    memory_hidden_dim: int = 256
    use_cross_attention: bool = True  # cross-attend to memory tokens at each layer

    # Stage 4 — adapters (LoRA)
    lora_rank: int = 32
    lora_alpha: int = 32


    @property
    def loom_shard_dim(self) -> int:
        return self.loom_content_dim + self.loom_physics_dim


def DocLoom20MConfig(**kwargs) -> DocLoomModelConfig:
    """Preserved 20M parameter architecture (6 layers, 256 embd, 256 block size)."""
    defaults = {
        "n_layer": 6,
        "n_head": 4,
        "n_embd": 256,
        "block_size": 256,
        "n_memory_tokens": 8,
        "memory_hidden_dim": 256,
        "lora_rank": 32,
        "lora_alpha": 32,
    }
    defaults.update(kwargs)
    return DocLoomModelConfig(**defaults)


def DocLoom50MConfig(**kwargs) -> DocLoomModelConfig:
    """Upgraded 60M parameter architecture with deeper hierarchical extraction
    (8 layers, 512 embd, 8 heads, 512 block size, 12 memory tokens)."""
    defaults = {
        "n_layer": 8,
        "n_head": 8,
        "n_embd": 512,
        "block_size": 512,
        "n_memory_tokens": 12,
        "memory_hidden_dim": 512,
        "lora_rank": 32,
        "lora_alpha": 64,
        "dropout": 0.1,
    }
    defaults.update(kwargs)
    return DocLoomModelConfig(**defaults)


def get_model_config(name: str = "20M", **kwargs) -> DocLoomModelConfig:
    """Returns model configuration by architecture name ('20M' or '50M'/'60M')."""
    name_clean = str(name).strip().upper()
    if "50" in name_clean or "60" in name_clean or "SCALED" in name_clean:
        return DocLoom50MConfig(**kwargs)
    return DocLoom20MConfig(**kwargs)
