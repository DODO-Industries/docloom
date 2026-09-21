"""
Safetensors checkpointing — replaces torch.save/.pt. torch.save uses Python's
pickle format internally, which can execute arbitrary code when loading an
untrusted file; safetensors stores only tensor data (no code execution
possible) and loads faster. Standard for real model releases today.
Config is saved alongside as plain JSON (safetensors only holds tensors).

Uses save_model/load_model (not save_file/load_file on a raw state_dict) —
DocLoomModel weight-ties backbone.head.weight to backbone.tok_emb.weight
(standard GPT-2 trick), and plain save_file refuses shared/aliased tensors
(would silently duplicate them on disk). save_model detects and handles this
correctly.
"""
import json
import re

from safetensors import safe_open
from safetensors.torch import save_model, load_model


def save_checkpoint(model, path: str) -> None:
    """path: e.g. '.../docloom_stageA.safetensors' — writes that file plus a
    sibling '<path>.config.json'."""
    save_model(model, path)
    with open(path + ".config.json", "w", encoding="utf-8") as f:
        json.dump(model.cfg.__dict__, f, indent=2)


def load_checkpoint(model_cls, config_cls, path: str, device: str = "cpu"):
    """Reconstructs a fresh model from its config + safetensors weights.
    If the checkpoint has LoRA adapters (Stage 4), re-adds them by name
    before loading — a freshly-built model has no adapter submodules, so
    load_model would otherwise fail on missing keys."""
    with open(path + ".config.json", "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = config_cls(**cfg_dict)
    model = model_cls(cfg)

    adapter_names = set()
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            m = re.search(r"\badapters\.([^.]+)\.", key)
            if m:
                adapter_names.add(m.group(1))
    for name in adapter_names:
        model.add_adapter(name)
        model.use_adapter(name)

    load_model(model, path, device=device)
    return model
