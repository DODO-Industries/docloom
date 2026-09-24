"""
Roadmap Stage 1 — turns Loom's excited shards into "memory tokens" the
backbone reads. Input per shard slot is content vector + physics state
(activation, energy, phase, momentum, stability, resonance, attention) —
not just retrieved text, per ROADMAP.md's Stage 1 note on this being a real
edge over plain-text RAG.
"""
import torch
import torch.nn as nn

from module_AI.model.config import DocLoomModelConfig


class MemoryProjection(nn.Module):
    """
    Dual-Stream Physics-Gated Memory Projection (Upgrade A):
    Stream 1: Content Vector (128d) -> Latent Token Embedding (n_embd).
    Stream 2: Physics Dynamics (7d: activation, energy, phase, resonance, etc.) -> Gate in [0.5, 1.0].
    The physics dynamics physically modulate the memory token's attention salience.
    """
    def __init__(self, cfg: DocLoomModelConfig):
        super().__init__()
        self.cfg = cfg
        self.content_fc1 = nn.Linear(cfg.loom_content_dim, cfg.memory_hidden_dim)
        self.content_fc2 = nn.Linear(cfg.memory_hidden_dim, cfg.n_embd)
        
        self.physics_fc1 = nn.Linear(cfg.loom_physics_dim, 64)
        self.physics_fc2 = nn.Linear(64, cfg.n_embd)
        self.act = nn.GELU()

    def forward(self, shard_features: torch.Tensor) -> torch.Tensor:
        """shard_features: (batch, n_memory_tokens, loom_shard_dim) -> (batch, n_memory_tokens, n_embd)"""
        c_dim = self.cfg.loom_content_dim
        content = shard_features[..., :c_dim]
        physics = shard_features[..., c_dim:]
        
        content_embed = self.content_fc2(self.act(self.content_fc1(content)))
        physics_gate = torch.sigmoid(self.physics_fc2(self.act(self.physics_fc1(physics))))
        
        # Physics modulation: highly active/resonant shards achieve full salience (1.0x),
        # while low-energy background shards are softly attenuated (0.5x).
        return content_embed * (0.5 + 0.5 * physics_gate)


def pack_shard_features(excited_shards, cfg: DocLoomModelConfig) -> torch.Tensor:
    """
    Builds the (n_memory_tokens, loom_shard_dim) tensor for one interaction
    from LoomMemory.recall() + get_shard_physics() output — pads with zeros
    if fewer than n_memory_tokens shards were excited, truncates if more
    (already top-k ordered by recall's score).

    excited_shards: list of dicts, each with a 'vector' (content, loom_content_dim
        floats) and physics fields (activation/energy/phase_angle/momentum/
        stability/resonance/attention) — see module_AI/training_logger.py for
        the exact schema this matches.
    """
    import numpy as np

    rows = []
    physics_keys = ["activation", "energy", "phase_angle", "momentum", "stability", "resonance", "attention"]
    for shard in excited_shards[: cfg.n_memory_tokens]:
        content = np.asarray(shard.get("vector", np.zeros(cfg.loom_content_dim)), dtype=np.float32)
        physics = np.array([float(shard.get(k, 0.0)) for k in physics_keys], dtype=np.float32)
        rows.append(np.concatenate([content, physics]))
    while len(rows) < cfg.n_memory_tokens:
        rows.append(np.zeros(cfg.loom_shard_dim, dtype=np.float32))
    return torch.from_numpy(np.stack(rows))


def pack_shard_features_batch(batch_excited_shards, cfg: DocLoomModelConfig) -> torch.Tensor:
    """Batched version: one excited_shards list per batch item (each example
    has its own recalled memory — real training batches, unlike the old
    single-example path that wrongly shared one memory context across a
    whole batch)."""
    return torch.stack([pack_shard_features(shards, cfg) for shards in batch_excited_shards])
