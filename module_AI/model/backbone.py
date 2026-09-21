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


class Block(nn.Module):
    def __init__(self, cfg: DocLoomModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask)
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

    def forward(self, inputs_embeds: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """inputs_embeds: (batch, seq_len, n_embd) — already-embedded sequence
        (memory tokens concatenated with token embeddings), see docloom_model.py.
        key_padding_mask: (batch, seq_len) bool, True = real token — required
        once a batch mixes examples of different lengths (padding needed)."""
        b, t, _ = inputs_embeds.shape
        pos = torch.arange(t, device=inputs_embeds.device).unsqueeze(0)
        x = self.drop(inputs_embeds + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, key_padding_mask)
        x = self.ln_f(x)
        return self.head(x)  # (batch, seq_len, vocab_size)

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.tok_emb(token_ids)
