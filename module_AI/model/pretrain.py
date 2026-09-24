"""
Stage A — self-supervised pretraining on Loom's own stored text, zero LLM
calls. Run this BEFORE Stage B (module_AI/model/train.py's distillation
fine-tuning) — teaches basic language modeling + memory-token reading using
data we already have, at no per-example generation cost.

Real batching (not one example at a time), LR warmup+decay, gradient
clipping, and a held-out validation split — fixes applied after the first
prototype run showed a suspiciously fast/noisy loss plateau, traced to
batch_size=1 giving extremely noisy gradients (see module_AI/ROADMAP.md).

Usage:
    python -m module_AI.model.pretrain --brain-dir module_AI/data/stress_brain \
        --limit 20000 --epochs 2 --batch-size 16
"""
import argparse
import os
import random

import torch
import torch.nn.functional as F

torch.set_num_threads(os.cpu_count())  # default (16) left cores idle on a 22-core box

from module_AI.model.config import DocLoomModelConfig, get_model_config
from module_AI.model.docloom_model import DocLoomModel
from module_AI.model.pretrain_dataset import PretrainDataset, chat_format_examples
from module_AI.model.batching import collate_batch, lr_lambda, IGNORE_INDEX, make_length_bucketed_batches
from module_AI.model.checkpoint import save_checkpoint

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "checkpoints")


@torch.no_grad()
def evaluate(model, val_examples, cfg, device, max_batches=20, batch_size=16):
    if not val_examples:
        return None
    model.eval()
    total_loss, total_batches = 0.0, 0
    val_batches = make_length_bucketed_batches(val_examples, batch_size, seed=1)[:max_batches]
    for batch in val_batches:
        batch_excited_shards, input_ids, target_ids, attention_mask = collate_batch(batch, device)
        logits = model(batch_excited_shards, input_ids, attention_mask)
        logits = logits[:, cfg.n_memory_tokens:, :]
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), target_ids.reshape(-1), ignore_index=IGNORE_INDEX
        )
        total_loss += loss.item()
        total_batches += 1
    model.train()
    return total_loss / max(1, total_batches)


def pretrain(brain_dir: str, limit: int, epochs: int, lr: float, batch_size: int = 16,
             val_fraction: float = 0.1, log_every: int = 25, eval_every: int = 200,
             exclude_clusters: list = None, extra_chat_texts: list = None,
             model_size: str = "60M"):
    cfg = get_model_config(model_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DocLoomModel(cfg).to(device)
    print(f"Model ({model_size}): {model.num_parameters():,} params, device={device}, batch_size={batch_size}")

    dataset = PretrainDataset(brain_dir, cfg, limit=limit, exclude_clusters=exclude_clusters)
    examples = list(dataset.examples)
    if extra_chat_texts:
        extra = chat_format_examples(extra_chat_texts, cfg)
        print(f"Adding {len(extra)} chat-format examples (kept in memory, not written to Loom)")
        examples.extend(extra)
    if len(examples) == 0:
        print("No shards found — check --brain-dir points at a real, closed brain directory.")
        return

    random.Random(42).shuffle(examples)
    n_val = max(1, int(len(examples) * val_fraction))
    val_examples, train_examples = examples[:n_val], examples[n_val:]
    print(f"Train examples: {len(train_examples)}, held-out validation: {len(val_examples)}")

    batches_per_epoch = max(1, len(train_examples) // batch_size)
    optimizer = torch.optim.AdamW(model.trainable_params(), lr=lr)
    total_steps = epochs * batches_per_epoch
    warmup_steps = max(10, int(0.03 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, warmup_steps, total_steps)
    )
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")

    model.train()
    step = 0
    running_loss = 0.0
    for epoch in range(epochs):
        batches = make_length_bucketed_batches(train_examples, batch_size, seed=epoch)
        for raw_batch in batches:
            batch_excited_shards, input_ids, target_ids, attention_mask = collate_batch(raw_batch, device)

            logits = model(batch_excited_shards, input_ids, attention_mask)
            logits = logits[:, cfg.n_memory_tokens:, :]  # drop memory-token positions

            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), target_ids.reshape(-1), ignore_index=IGNORE_INDEX
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_params(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if step % log_every == 0:
                avg = running_loss / max(1, min(step + 1, log_every))
                cur_lr = scheduler.get_last_lr()[0]
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} (avg {avg:.4f}) lr {cur_lr:.2e}")
                running_loss = 0.0
            if step % eval_every == 0 and step > 0:
                val_loss = evaluate(model, val_examples, cfg, device, batch_size=batch_size)
                print(f"  -- validation loss: {val_loss:.4f} (held-out, never trained on)")
            step += 1

    final_val = evaluate(model, val_examples, cfg, device, batch_size=batch_size)
    print(f"Final validation loss: {final_val:.4f}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"docloom_{model_size.lower()}_stageA.safetensors")
    save_checkpoint(model, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")
    # Also save as default stage A checkpoint for pipeline compatibility
    alias_path = os.path.join(CKPT_DIR, "docloom_pretrained_stageA.safetensors")
    save_checkpoint(model, alias_path)
    print(f"Saved alias checkpoint: {alias_path}")
    return ckpt_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain-dir", type=str,
                     default=os.path.join(os.path.dirname(__file__), "..", "data", "stress_brain"))
    ap.add_argument("--limit", type=int, default=20000, help="max shards to load")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--model-size", type=str, default="60M", choices=["20M", "60M", "50M"],
                     help="Architecture size: 20M (6 layers, 256 embd) or 60M (8 layers, 512 embd)")
    ap.add_argument("--exclude-clusters", type=str, default="",
                     help="comma-separated crystal filenames to skip, e.g. crystal_1_E.loom")
    ap.add_argument("--extra-chat-samples", type=int, default=0,
                     help="mix N chat-formatted Q&A examples in from memory (not stored in Loom) "
                          "-- see module_AI/scripts/ingest_chat_examples.py for the source data")
    args = ap.parse_args()

    extra_chat_texts = None
    if args.extra_chat_samples > 0:
        from module_AI.scripts.ingest_chat_examples import _load_examples, LOG_PATH, SQUAD_LOG_PATH
        from module_AI.model.chat_format import format_turn
        import random as _random
        real = _load_examples(LOG_PATH)
        squad = _load_examples(SQUAD_LOG_PATH, sample_size=args.extra_chat_samples)
        pairs = real + squad
        if len(pairs) > args.extra_chat_samples:
            pairs = _random.Random(42).sample(pairs, args.extra_chat_samples)
        extra_chat_texts = [format_turn(q, a) for q, a in pairs]

    pretrain(args.brain_dir, args.limit, args.epochs, args.lr, args.batch_size,
             exclude_clusters=[c.strip() for c in args.exclude_clusters.split(",") if c.strip()],
             extra_chat_texts=extra_chat_texts, model_size=args.model_size)
