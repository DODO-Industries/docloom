# Transformer Docs (`transformer.py`)

The `LoomTransformer` is the underlying NLP engine for the entire Loom service. It wraps heavy machine learning models into simple, efficient utility methods.

## 1. Core Technologies
- **Sentence-Transformers:** `all-MiniLM-L6-v2` for generating 384-dimensional semantic embeddings.
- **SpaCy:** For high-accuracy sentence segmentation and Named Entity Recognition (NER).
- **RAKE-NLTK:** For rapid extraction of ranked keywords (concepts).

## 2. Core Capabilities

### A. Semantic Decomposition (`decompose`)
Instead of simple splitting, it uses SpaCy's sentence tokenizer to break paragraphs into "atomic units." It filters out fragments that are too short to hold meaning.

### B. Concept Extraction (`extract_concepts`)
Uses the RAKE algorithm to find the most important phrases in a block of text. These are used to build the `Concept Bridge` in the Weaver.

### C. Entity Recognition (`extract_entities`)
Extracts entities like `PERSON`, `ORG`, and `GPE`. These are used in the **Edge Scoring Function** to link nodes that discuss the same entities.

### D. Similarity Math
- `calculate_cosine`: Measures the angle between two embedding vectors.
- `calculate_jaccard`: Measures the overlap between two sets of concepts/entities.

## 3. Singleton Pattern
The `LoomTransformer` uses a singleton pattern (`__new__`) to ensure that heavy models (SentenceTransformer, SpaCy) are only loaded into memory **once**, even if multiple components initialize the class.

## 4. Interaction with other files
- Primary utility used by `LoomWeaver` during construction.
- Used by `LoomServerService` to embed user queries during retrieval.
