"""
Roadmap Stage 4 — growable adapters (LoRA-style), one set per domain/crystal.
Injected INSIDE backbone Linear layers, not a separate sequential stage —
see ROADMAP.md, Stage 4.
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen base nn.Linear with a swappable set of low-rank adapters.
    Only one adapter set is active at a time (selected by the router, Stage 2);
    'base' (no adapter) is the default. New domains add a new named entry to
    `self.adapters` without touching the base weight or other adapters."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base  # trainable by default — Phase 3 pretrains these
        self.rank = rank
        self.scale = alpha / rank
        self.adapters = nn.ModuleDict()
        self.active: Optional[str] = None

    def freeze_base(self) -> None:
        """Call once Phase 3 pretraining is done, before Phase 4 adapter
        training — base weights become read-only, only adapters train."""
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def add_adapter(self, name: str) -> None:
        in_f, out_f = self.base.in_features, self.base.out_features
        device = self.base.weight.device
        dtype = self.base.weight.dtype
        A = nn.Linear(in_f, self.rank, bias=False, device=device, dtype=dtype)
        B = nn.Linear(self.rank, out_f, bias=False, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(A.weight, a=math.sqrt(5))
        nn.init.zeros_(B.weight)  # start as a no-op, standard LoRA init
        self.adapters[name] = nn.ModuleDict({"A": A, "B": B})

    def set_active(self, name: Optional[str]) -> None:
        """None or 'base' = no adapter (frozen base weights only)."""
        if name is not None and name not in self.adapters:
            raise KeyError(f"No adapter named {name!r}; call add_adapter first")
        self.active = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.active is not None:
            pair = self.adapters[self.active]
            out = out + self.scale * pair["B"](pair["A"](x))
        return out


def all_lora_layers(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def add_adapter_set(model: nn.Module, name: str) -> None:
    """Adds a same-named adapter to every LoRA injection point in the model —
    one 'domain/crystal' adapter set, per ROADMAP.md Stage 4."""
    for layer in all_lora_layers(model):
        layer.add_adapter(name)


def set_active_adapter(model: nn.Module, name: Optional[str]) -> None:
    for layer in all_lora_layers(model):
        layer.set_active(name)


def freeze_base_weights(model: nn.Module) -> None:
    """Phase 3 -> Phase 4 transition: call once backbone pretraining is done,
    before adding/training the first adapter set."""
    for layer in all_lora_layers(model):
        layer.freeze_base()


def trainable_parameters(model: nn.Module, adapter_name: Optional[str] = None):
    """Params to optimize for a given adapter's training pass — base weights
    stay frozen (requires_grad_(False) in LoRALinear.__init__), only that
    adapter's A/B matrices (plus any non-LoRA layers, e.g. Stage 1/5) train."""
    if adapter_name is None:
        return [p for p in model.parameters() if p.requires_grad]
    params = []
    for layer in all_lora_layers(model):
        if adapter_name in layer.adapters:
            params.extend(layer.adapters[adapter_name].parameters())
    return params
