import spacy
import numpy as np
from rake_nltk import Rake
from sentence_transformers import SentenceTransformer
from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("LoomTransformer")

class LoomTransformer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LoomTransformer, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Ensure initialization only happens once
        if hasattr(self, 'model'): return
        
        log_service(logger, f"Initializing LoomTransformer Neural Engine ({model_name})...", "info")
        self.model = SentenceTransformer(model_name)
        self.rake = Rake()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            log_service(logger, "SpaCy model not found, falling back to basic NLP", "warning")
            self.nlp = None
        
    def decompose(self, text, is_heading=False):
        """
        Splits complex text into atomic semantic units (Looms).
        Uses SpaCy's sentence tokenizer for high accuracy.
        """
        min_len = 5 if is_heading else 10
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= min_len]
        # Fallback
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) >= min_len]
        return sentences

    def extract_concepts(self, text):
        """Extracts key concepts using RAKE with a basic fallback for safety."""
        try:
            self.rake.extract_keywords_from_text(text)
            return self.rake.get_ranked_phrases()[:5]
        except Exception:
            # Fallback: simple noun-phrase extraction or top words
            words = [w.strip() for w in text.split() if len(w) > 4]
            return list(set(words))[:5]

    def extract_entities(self, text):
        """Extracts entities like ORG, PERSON, GPE using SpaCy."""
        if not self.nlp: return []
        doc = self.nlp(text)
        entities = [f"{ent.text} ({ent.label_})" for ent in doc.ents]
        return list(set(entities))

    def get_embeddings(self, texts):
        """Batch generates embeddings."""
        if not texts:
            return []
        return self.model.encode(texts)

    def calculate_jaccard(self, list1, list2):
        """Calculates Jaccard similarity between two lists of concepts."""
        set1, set2 = set(list1), set(list2)
        if not set1 or not set2:
            return 0.0
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def calculate_cosine(self, emb1, emb2):
        """Calculates cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
