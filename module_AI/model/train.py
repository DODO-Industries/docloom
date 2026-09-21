"""
Roadmap Phase 3 — distillation training loop. Reads module_AI/data/training_log/
interactions.jsonl (logged for free by every /ai/ask call, see
module_AI/training_logger.py) and trains DocLoomModel to predict the answer
given the excited shards + question — the actual "beat Qwen3 at a fraction of
the size" test, once enough real examples have accumulated.

Real batching, LR warmup+decay, gradient clipping, held-out validation split,
safetensors checkpoints — same fixes as module_AI/model/pretrain.py, applied
here too (see module_AI/ROADMAP.md for why).

Usage:
    python -m module_AI.model.train --epochs 3 --lr 3e-4 --batch-size 8
"""
import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

torch.set_num_threads(os.cpu_count())  # default (16) left cores idle on a 22-core box

from module_AI.model.config import DocLoomModelConfig
from module_AI.model.docloom_model import DocLoomModel
from module_AI.model.tokenizer import encode
from module_AI.model.chat_format import format_prompt
from module_AI.model.batching import collate_batch, lr_lambda, IGNORE_INDEX, make_length_bucketed_batches
from module_AI.model.checkpoint import save_checkpoint
from module_AI.model.pretrain import evaluate  # same eval loop works for both stages
from module_AI.training_logger import LOG_PATH

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "checkpoints")


SQUAD_LOG_PATH = os.path.join(os.path.dirname(LOG_PATH), "squad_examples.jsonl")


class InteractionDataset(Dataset):
    """One example = (excited_shards, question+answer token ids, q_len marking
    where the answer starts — loss excludes the question portion). Reads
    real Qwen-distilled Loom interactions AND (if present) the SQuAD-derived
    set from module_AI/scripts/build_squad_dataset.py — same record schema,
    loaded together."""

    def __init__(self, log_paths, cfg: DocLoomModelConfig):
        if isinstance(log_paths, str):
            log_paths = [log_paths]
        self.examples = []
        for log_path in log_paths:
            if not os.path.exists(log_path):
                continue
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    q_ids = encode(format_prompt(rec["query"]))
                    a_ids = encode(rec["answer"])
                    ids = q_ids + a_ids
                    if len(ids) + cfg.n_memory_tokens > cfg.block_size:
                        ids = ids[-(cfg.block_size - cfg.n_memory_tokens):]
                        q_len = max(0, len(ids) - len(a_ids))
                    else:
                        q_len = len(q_ids)
                    self.examples.append({
                        "excited_shards": rec.get("excited_shards", []),
                        "ids": ids,
                        "q_len": q_len,
                    })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


STAGE_A_CKPT = os.path.join(CKPT_DIR, "docloom_pretrained_stageA.safetensors")


def _run_training_pass(model, cfg, device, examples, epochs, lr, batch_size,
                        val_fraction, log_every, eval_every, pass_name: str):
    """One training pass over `examples`, mutating `model` in place. Used
    twice by train(): once on SQuAD (general skill, large/cheap), once on
    real Loom/Qwen data (specialization, small/precious) — so the tiny real
    dataset isn't drowned out by SQuAD's much larger volume in one flat mix."""
    if len(examples) == 0:
        print(f"[{pass_name}] no examples, skipping")
        return
    random.Random(42).shuffle(examples)
    n_val = max(1, int(len(examples) * val_fraction)) if len(examples) >= 10 else 0
    val_examples, train_examples = examples[:n_val], examples[n_val:]
    print(f"[{pass_name}] train: {len(train_examples)}, held-out validation: {len(val_examples)}")

    effective_batch = min(batch_size, max(1, len(train_examples)))
    batches_per_epoch = max(1, len(train_examples) // effective_batch)
    optimizer = torch.optim.AdamW(model.trainable_params(), lr=lr)
    total_steps = max(1, epochs * batches_per_epoch)
    warmup_steps = max(5, int(0.05 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, warmup_steps, total_steps)
    )
    print(f"[{pass_name}] total steps: {total_steps}, warmup: {warmup_steps}")

    model.train()
    step = 0
    for epoch in range(epochs):
        batches = make_length_bucketed_batches(train_examples, effective_batch, seed=epoch)
        for raw_batch in batches:
            batch_excited_shards, input_ids, target_ids, attention_mask = collate_batch(raw_batch, device)

            logits = model(batch_excited_shards, input_ids, attention_mask)
            logits = logits[:, cfg.n_memory_tokens:, :]

            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), target_ids.reshape(-1), ignore_index=IGNORE_INDEX
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_params(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if step % log_every == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"[{pass_name}] epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} lr {cur_lr:.2e}")
            if val_examples and step % eval_every == 0 and step > 0:
                val_loss = evaluate(model, val_examples, cfg, device, batch_size=effective_batch)
                print(f"  -- [{pass_name}] validation loss: {val_loss:.4f} (held-out)")
            step += 1

    if val_examples:
        print(f"[{pass_name}] final validation loss: "
              f"{evaluate(model, val_examples, cfg, device, batch_size=effective_batch):.4f}")


def train(squad_epochs: int = 1, loom_epochs: int = 5, lr: float = 3e-4, batch_size: int = 16,
          val_fraction: float = 0.1, log_every: int = 25, eval_every: int = 500,
          init_from_stage_a: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if init_from_stage_a and os.path.exists(STAGE_A_CKPT):
        from module_AI.model.checkpoint import load_checkpoint
        model = load_checkpoint(DocLoomModel, DocLoomModelConfig, STAGE_A_CKPT, device=device).to(device)
        cfg = model.cfg
        print(f"Initialized from Stage A checkpoint: {STAGE_A_CKPT}")
        # Freeze the fluent backbone learned in Stage A — without this, full
        # fine-tuning on SQuAD/QA-format data catastrophically overwrites the
        # language fluency Stage A just learned (verified: raw Stage A samples
        # are coherent short stories; Stage B output was not). Only a small
        # LoRA adapter + the output head + memory_proj train from here.
        model.freeze_backbone()
        model.add_adapter("qa")
        model.use_adapter("qa")
        print(f"Backbone frozen, training QA adapter only: "
              f"{sum(p.numel() for p in model.trainable_params()):,} trainable params "
              f"(of {model.num_parameters():,} total)")
    else:
        cfg = DocLoomModelConfig()
        model = DocLoomModel(cfg).to(device)
        print("WARNING: no Stage A checkpoint found — training from random init")
    print(f"Model: {model.num_parameters():,} params, device={device}, batch_size={batch_size}")

    squad_examples = list(InteractionDataset(SQUAD_LOG_PATH, cfg))
    loom_examples = list(InteractionDataset(LOG_PATH, cfg))
    print(f"SQuAD examples: {len(squad_examples)}, real Loom/Qwen examples: {len(loom_examples)}")

    # Pass 1: general "read context, answer" skill — large, cheap, generic.
    _run_training_pass(model, cfg, device, squad_examples, squad_epochs, lr, batch_size,
                        val_fraction, log_every, eval_every, pass_name="squad")

    # Pass 2: specialize on real Loom data — small, precious, not drowned out
    # by pass 1's much larger volume since it's a separate pass, not a mix.
    _run_training_pass(model, cfg, device, loom_examples, loom_epochs, lr * 0.5, batch_size,
                        val_fraction, log_every, max(10, eval_every // 5), pass_name="loom")

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, "docloom_stageB.safetensors")
    save_checkpoint(model, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--squad-epochs", type=int, default=1)
    ap.add_argument("--loom-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--from-scratch", action="store_true",
                     help="ignore the Stage A checkpoint and train from random init")
    args = ap.parse_args()
    train(squad_epochs=args.squad_epochs, loom_epochs=args.loom_epochs, lr=args.lr,
          batch_size=args.batch_size, init_from_stage_a=not args.from_scratch)
