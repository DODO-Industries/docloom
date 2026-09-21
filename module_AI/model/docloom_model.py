"""
The full assembled model: Stage 1 (memory projection) + Stage 3/5 (backbone +
head). Stage 0 (input encoders) lives outside this file — it's whatever
produces `excited_shards` (today: module_AI.memory_bridge.LoomMemory.recall +
get_shard_physics; future: photo/audio encoders producing the same shape).
Stage 2 (router) is also external — see module_AI.model.lora.set_active_adapter,
called by whoever picks the crystal/domain before forward().
"""
from typing import List, Optional

import torch
import torch.nn as nn

from module_AI.model.config import DocLoomModelConfig
from module_AI.model.memory_projection import MemoryProjection, pack_shard_features_batch
from module_AI.model.backbone import Backbone
from module_AI.model.lora import add_adapter_set, set_active_adapter, trainable_parameters, freeze_base_weights
from module_AI.model.tokenizer import encode, decode


class DocLoomModel(nn.Module):
    def __init__(self, cfg: Optional[DocLoomModelConfig] = None):
        super().__init__()
        self.cfg = cfg or DocLoomModelConfig()
        self.memory_proj = MemoryProjection(self.cfg)
        self.backbone = Backbone(self.cfg)

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)

    def forward(self, batch_excited_shards: List[List[dict]], token_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        batch_excited_shards: one excited-shards list PER BATCH ITEM — each
            example in a batch has its own recalled memory (real batching,
            not one shared context stretched across the batch).
        token_ids: (batch, text_len) already-tokenized, already-padded text.
        attention_mask: (batch, text_len) bool/0-1, True/1 = real token,
            False/0 = padding. None means no padding (all real, e.g. batch=1).
        Returns: (batch, n_memory_tokens + text_len, vocab_size) logits.
        """
        device = token_ids.device
        shard_features = pack_shard_features_batch(batch_excited_shards, self.cfg).to(device)
        memory_tokens = self.memory_proj(shard_features)  # (batch, n_memory_tokens, n_embd)

        text_embeds = self.backbone.embed_tokens(token_ids)  # (batch, text_len, n_embd)
        inputs_embeds = torch.cat([memory_tokens, text_embeds], dim=1)

        key_padding_mask = None
        if attention_mask is not None:
            mem_mask = torch.ones(token_ids.shape[0], self.cfg.n_memory_tokens, dtype=torch.bool, device=device)
            key_padding_mask = torch.cat([mem_mask, attention_mask.bool()], dim=1)

        max_len = self.cfg.block_size
        if inputs_embeds.shape[1] > max_len:
            inputs_embeds = inputs_embeds[:, -max_len:]
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask[:, -max_len:]

        return self.backbone(inputs_embeds, key_padding_mask)

    # --- Stage 4 convenience passthroughs ---
    def add_adapter(self, name: str) -> None:
        add_adapter_set(self, name)

    def use_adapter(self, name: Optional[str]) -> None:
        set_active_adapter(self, name)

    def trainable_params(self, adapter_name: Optional[str] = None):
        return trainable_parameters(self, adapter_name)

    def freeze_backbone(self) -> None:
        """Phase 3 -> Phase 4 transition — call once, after backbone
        pretraining, before training the first adapter set. Also freezes the
        token embedding (tied to the output head): leaving it trainable
        would drift the embedding space the frozen attention/MLP layers were
        pretrained against, decalibrating them even though their own weights
        never change — verified empirically, output got *more* fragmented
        when only attention/MLP were frozen and the tied embedding wasn't."""
        freeze_base_weights(self)
        self.backbone.tok_emb.weight.requires_grad_(False)

    @torch.no_grad()
    def generate(self, excited_shards: List[dict], prompt: str, max_new_tokens: int = 64,
                 temperature: float = 0.8) -> str:
        self.eval()
        device = next(self.parameters()).device
        ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            logits = self.forward([excited_shards], ids)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return decode(ids[0].tolist())
