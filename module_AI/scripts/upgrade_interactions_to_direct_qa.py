"""
Upgrades all raw interactions in interactions.jsonl into clean, structured
direct QA examples with explicit <|think|> evidence citations and crisp answers.
"""
import json
import os
import re
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_LOG = os.path.join(PROJECT_ROOT, "module_AI", "data", "training_log", "interactions.jsonl")
DIRECT_QA_PATH = os.path.join(PROJECT_ROOT, "module_AI", "data", "training_log", "direct_grounded_qa.jsonl")
OUT_PATH = os.path.join(PROJECT_ROOT, "module_AI", "data", "training_log", "scaled_grounded_qa.jsonl")

def clean_answer(text: str) -> str:
    text = text.strip()
    for phrase in [
        "This is mentioned in the context",
        "This came from the memory context",
        "Answer came from memory",
        "Answer came from memory context",
        "Based on the provided context, ",
        "Based on the context, ",
        "The memory context mentions that ",
        "The context mentions that ",
        "According to the context, ",
        "According to the text, ",
        "From the story, ",
    ]:
        if phrase in text:
            idx = text.find(phrase)
            if idx > 0:
                text = text[:idx].strip()
            elif idx == 0:
                text = text[len(phrase):].strip()
                
    # Remove citation scores like (13.32) or (15.09)
    text = re.sub(r"\s*\(\d+\.\d+\)\.?$", "", text).strip()
    return text.strip()

def extract_evidence_sentence(shard_text: str, answer_text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', shard_text.strip())
    ans_words = set(re.findall(r'\w+', answer_text.lower()))
    
    best_sent = shard_text[:120]
    best_overlap = -1
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean.split()) < 4:
            continue
        s_words = set(re.findall(r'\w+', s_clean.lower()))
        overlap = len(ans_words.intersection(s_words))
        if overlap > best_overlap:
            best_overlap = overlap
            best_sent = s_clean
            
    return best_sent

def upgrade():
    total_raw = 0
    upgraded = []
    
    if os.path.exists(RAW_LOG):
        with open(RAW_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_raw += 1
                try:
                    rec = json.loads(line)
                    q = rec.get("query", "").strip()
                    ans = clean_answer(rec.get("answer", ""))
                    shards = rec.get("excited_shards", [])
                    
                    if not q or not ans or len(ans.split()) < 3 or not shards:
                        continue
                        
                    top_shard = shards[0].get("text", "")
                    evidence = extract_evidence_sentence(top_shard, ans)
                    thought = f"Evidence from retrieved memory: '{evidence}'"
                    
                    upgraded.append({
                        "ts": rec.get("ts"),
                        "query": q,
                        "thought": thought,
                        "answer": ans,
                        "excited_shards": shards,
                        "used_memory": True
                    })
                except Exception:
                    continue

    # Also keep all existing direct grounded QA records
    direct_count = 0
    if os.path.exists(DIRECT_QA_PATH):
        with open(DIRECT_QA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    upgraded.append(rec)
                    direct_count += 1
                except Exception:
                    continue
                    
    print(f"Read {total_raw} raw interactions.")
    print(f"Included {direct_count} existing direct QA records.")
    print(f"Total Scaled Grounded QA records: {len(upgraded)}")
    
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in upgraded:
            f.write(json.dumps(r) + "\n")
            
    # Also overwrite direct_grounded_qa.jsonl with this massive, pristine dataset
    with open(DIRECT_QA_PATH, "w", encoding="utf-8") as f:
        for r in upgraded:
            f.write(json.dumps(r) + "\n")
            
    print(f"Wrote {len(upgraded)} scaled records to {OUT_PATH} and {DIRECT_QA_PATH}!")

if __name__ == "__main__":
    upgrade()
