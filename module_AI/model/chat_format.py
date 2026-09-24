"""
Our own lightweight chat-template format (Loom's answer to ChatML) — plain
text tags, no new vocab/special tokens needed since the base GPT-2 BPE
tokenizer already splits "<|user|>" etc. into ordinary subword pieces it
knows. Root-cause fix for the Stage B collapse: the frozen Stage A backbone
never saw ANY question/answer structure during pretraining (only plain
story text), so a tiny Stage B adapter had to learn the structure from
scratch and failed. This format must appear in BOTH Stage A's corpus and
Stage B's QA data so the frozen base learns the structure natively.
"""

CONTEXT_TAG = "<|context|>"
USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
THINK_TAG = "<|think|>"
END_THINK_TAG = "</|think|>"
EVIDENCE_TAG = "<|evidence|>"
ANSWER_TAG = "<|answer|>"
END_TAG = "<|end|>"


def format_context(snippets: list) -> str:
    if not snippets:
        return ""
    return CONTEXT_TAG + "\n" + "\n".join(snippets) + "\n"


def format_prompt(question: str, context_snippets: list = ()) -> str:
    """Everything up to where the model should start generating."""
    return f"{format_context(list(context_snippets))}{USER_TAG}\n{question}\n{ASSISTANT_TAG}\n"


def format_turn(question: str, answer: str, context_snippets: list = (), evidence: str = "", thought: str = "") -> str:
    """A full structured training example: prompt + thinking/evidence + answer + end tag."""
    if thought:
        body = f"{THINK_TAG}\n{thought}\n{END_THINK_TAG}\n{ANSWER_TAG}\n{answer}"
    elif evidence:
        body = f"{EVIDENCE_TAG}\n{evidence}\n{ANSWER_TAG}\n{answer}"
    else:
        body = answer
    return f"{format_prompt(question, context_snippets)}{body}{END_TAG}"

