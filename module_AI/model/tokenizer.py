"""Reuse GPT-2's BPE tokenizer via tiktoken — no reason to build one, per
ROADMAP.md's Stage 3 note."""
import tiktoken

_enc = tiktoken.get_encoding("gpt2")


def encode(text: str) -> list:
    return _enc.encode(text)


def decode(token_ids) -> str:
    return _enc.decode(list(token_ids))


VOCAB_SIZE = _enc.n_vocab
