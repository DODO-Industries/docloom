"""
The full assembled model: Stage 1 (memory projection) + Stage 3/5 (backbone +
head). Stage 0 (input encoders) lives outside this file — it's whatever
produces `excited_shards` (today: module_AI.memory_bridge.LoomMemory.recall +
get_shard_physics; future: photo/audio encoders producing the same shape).
Stage 2 (router) is also external — see module_AI.model.lora.set_active_adapter,
called by whoever picks the crystal/domain before forward().
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from module_AI.model.config import DocLoomModelConfig
from module_AI.model.memory_projection import MemoryProjection, pack_shard_features_batch
from module_AI.model.backbone import Backbone
from module_AI.model.lora import add_adapter_set, set_active_adapter, trainable_parameters, freeze_base_weights
from module_AI.model.tokenizer import encode, decode
from module_AI.model.multimodal_stem import VisualPatchStem, DualManifoldBridge
import torch.nn.functional as F


class DocLoomModel(nn.Module):
    def __init__(self, cfg: Optional[DocLoomModelConfig] = None):
        super().__init__()
        self.cfg = cfg or DocLoomModelConfig()
        self.memory_proj = MemoryProjection(self.cfg)
        self.backbone = Backbone(self.cfg)

        # Multi-Modal Thalamic Perception & Mirror-World Bridge
        self.visual_stem = VisualPatchStem(self.cfg)
        self.mirror_bridge = DualManifoldBridge(self.cfg, crystal_dim=self.cfg.loom_content_dim)

        # Native Self-Embedding Head (generates crystal coordinates directly from latent states)
        self.native_embed_head = nn.Sequential(
            nn.Linear(self.cfg.n_embd, self.cfg.n_embd),
            nn.GELU(),
            nn.Linear(self.cfg.n_embd, self.cfg.loom_content_dim)
        )

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)

    def embed_text_crystal(self, token_ids: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generates 128-d / 384-d normalized crystal coordinates directly from text tokens
        without requiring external SentenceTransformers or MiniLM.
        """
        text_embeds = self.backbone.embed_tokens(token_ids)
        _, hidden = self.backbone(text_embeds, return_hidden=True)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden.mean(dim=1)
        crystal_vector = self.native_embed_head(pooled)
        return F.normalize(crystal_vector, p=2, dim=-1)

    def embed_visual_patches(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes 2D visual patches with 2D-RoPE and projects into native crystal coordinates.
        Returns: (patch_tokens, normalized_crystal_vector)
        """
        patch_tokens, _ = self.visual_stem(pixel_values)
        crystal_vector = self.mirror_bridge.project_visual_crystal(patch_tokens)
        return patch_tokens, crystal_vector

    def forward(self, batch_excited_shards: List[List[dict]], token_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard causal forward pass on memory + text tokens."""
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

        return self.backbone(inputs_embeds, key_padding_mask, memory_tokens)

    def forward_multimodal(self, batch_excited_shards: List[List[dict]],
                           token_ids: Optional[torch.Tensor] = None,
                           pixel_values: Optional[torch.Tensor] = None,
                           attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Dual-manifold multimodal forward pass: binds visual patch tokens and text tokens
        into unified sequence with memory tokens.
        """
        device = next(self.parameters()).device
        shard_features = pack_shard_features_batch(batch_excited_shards, self.cfg).to(device)
        memory_tokens = self.memory_proj(shard_features)

        components = [memory_tokens]

        if pixel_values is not None:
            patch_tokens, _ = self.visual_stem(pixel_values.to(device))
            components.append(patch_tokens)

        if token_ids is not None:
            text_embeds = self.backbone.embed_tokens(token_ids.to(device))
            components.append(text_embeds)

        inputs_embeds = torch.cat(components, dim=1)
        max_len = self.cfg.block_size
        if inputs_embeds.shape[1] > max_len:
            inputs_embeds = inputs_embeds[:, -max_len:]

        return self.backbone(inputs_embeds, None, memory_tokens)

    # --- Stage 4 convenience passthroughs ---
    def add_adapter(self, name: str) -> None:
        add_adapter_set(self, name)

    def use_adapter(self, name: Optional[str]) -> None:
        set_active_adapter(self, name)

    def trainable_params(self, adapter_name: Optional[str] = None):
        return trainable_parameters(self, adapter_name)

    def freeze_backbone(self) -> None:
        """Phase 3 -> Phase 4 transition: freezes 100% of the backbone (weights,
        biases, LayerNorms, position embeddings, and token embeddings).
        Only the LoRA adapters, MemoryCrossAttention, and memory_proj train."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        freeze_base_weights(self)

    @torch.no_grad()
    def generate(self, excited_shards: List[dict], prompt: str, max_new_tokens: int = 100,
                 temperature: float = 0.7, top_k: int = 40, top_p: float = 0.9,
                 repetition_penalty: float = 1.15, stop_tokens: Optional[List[str]] = None,
                 return_full_text: bool = False) -> str:
        self.eval()
        device = next(self.parameters()).device
        prompt_ids = encode(prompt)
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        stops = stop_tokens if stop_tokens is not None else ["<|end|>", "|end|>", "<|end", "<|im_end|>", "<|endoftext|>", "<|user|>", "<|user"]
        generated_ids = []

        for _ in range(max_new_tokens):
            curr_ids = ids if ids.shape[1] <= self.cfg.block_size else ids[:, -self.cfg.block_size:]
            logits = self.forward([excited_shards], curr_ids)
            next_logits = logits[:, -1, :].clone()

            # Repetition penalty
            if repetition_penalty != 1.0 and ids.shape[1] > 0:
                for token_id in set(ids[0].tolist()):
                    if next_logits[0, token_id] > 0:
                        next_logits[0, token_id] /= repetition_penalty
                    else:
                        next_logits[0, token_id] *= repetition_penalty

            next_logits = next_logits / max(temperature, 1e-6)

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('Inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[0, indices_to_remove] = -float('Inf')

            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
            generated_ids.append(next_id.item())

            # Check stop sequences
            partial_text = decode(generated_ids)
            if any(stop in partial_text for stop in stops):
                break

        if return_full_text:
            out_text = decode(ids[0].tolist())
        else:
            out_text = decode(generated_ids)

        for stop in stops:
            if stop in out_text:
                out_text = out_text.split(stop)[0]
        return out_text.strip()

