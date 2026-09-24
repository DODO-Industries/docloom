"""
Roadmap Stage 3 (backbone) + Stage 5 (output head). Standard decoder-only
transformer blocks (GPT-2-style: LayerNorm -> causal self-attention -> residual
-> LayerNorm -> MLP -> residual) — borrowed, proven block design, per
ROADMAP.md's honest note that the innovation is Stage 1/2/4, not reinventing
attention math. LoRA injection points (Stage 4) sit inside each block's
attention and MLP Linear layers.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from module_AI.model.config import DocLoomModelConfig
from module_AI.model.lora import LoRALinear


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: DocLoomModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = LoRALinear(nn.Linear(cfg.n_embd, 3 * cfg.n_embd), cfg.lora_rank, cfg.lora_alpha)
        self.proj = LoRALinear(nn.Linear(cfg.n_embd, cfg.n_embd), cfg.lora_rank, cfg.lora_alpha)
        self.dropout = cfg.dropout
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """key_padding_mask: (batch, seq_len) bool, True = real token, False =
        padding — needed now that training batches multiple (differently
        long) examples together instead of one at a time."""
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float("-inf"))
        if key_padding_mask is not None:
            pad = ~key_padding_mask[:, None, None, :]  # (b, 1, 1, t) — mask padded KEY positions
            att = att.masked_fill(pad, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: DocLoomModelConfig):
        super().__init__()
        self.fc = LoRALinear(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), cfg.lora_rank, cfg.lora_alpha)
        self.act = nn.GELU()
        self.proj = LoRALinear(nn.Linear(4 * cfg.n_embd, cfg.n_embd), cfg.lora_rank, cfg.lora_alpha)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(self.act(self.fc(x))))


class MemoryCrossAttention(nn.Module):
    """
    Cross-attention to Loom's memory tokens (Upgrade B).
    Queries come from the current sequence tokens;
    Keys and Values come from the Loom Memory Tokens.
    """
    def __init__(self, cfg: DocLoomModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.q = LoRALinear(nn.Linear(cfg.n_embd, cfg.n_embd), cfg.lora_rank, cfg.lora_alpha)
        self.kv = LoRALinear(nn.Linear(cfg.n_embd, 2 * cfg.n_embd), cfg.lora_rank, cfg.lora_alpha)
        self.proj = LoRALinear(nn.Linear(cfg.n_embd, cfg.n_embd), cfg.lora_rank, cfg.lora_alpha)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, memory_tokens: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        _, m, _ = memory_tokens.shape
        q = self.q(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        kv = self.kv(memory_tokens)
        k, v = kv.split(c, dim=2)
        k = k.view(b, m, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, m, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg: DocLoomModelConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)
        self.use_cross_attention = getattr(cfg, "use_cross_attention", False)
        if self.use_cross_attention:
            self.ln_cross = nn.LayerNorm(cfg.n_embd)
            self.cross_attn = MemoryCrossAttention(cfg)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
                memory_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask)
        if self.use_cross_attention and memory_tokens is not None:
            x = x + self.cross_attn(self.ln_cross(x), memory_tokens)
        x = x + self.mlp(self.ln2(x))
        return x


class Backbone(nn.Module):
    """Stage 3 + Stage 5: token/positional embedding, N blocks, final norm,
    output head (weight-tied to the token embedding — standard GPT-2 trick,
    matters more at this small size than at 1B+)."""

    def __init__(self, cfg: DocLoomModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, inputs_embeds: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
                memory_tokens: Optional[torch.Tensor] = None, return_hidden: bool = False):
        """inputs_embeds: (batch, seq_len, n_embd) — already-embedded sequence.
        memory_tokens: (batch, n_memory_tokens, n_embd) — direct cross-attention target.
        return_hidden: if True, returns (logits, hidden_states)."""
        b, t, _ = inputs_embeds.shape
        pos = torch.arange(t, device=inputs_embeds.device).unsqueeze(0)
        x = self.drop(inputs_embeds + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, key_padding_mask, memory_tokens)
        x = self.ln_f(x)
        logits = self.head(x)
        if return_hidden:
            return logits, x
        return logits

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.tok_emb(token_ids)
