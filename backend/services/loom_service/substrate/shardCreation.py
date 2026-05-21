import sys
import os

def setup_python_path():
    """Dynamically finds the project root and adds it to sys.path."""
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr != os.path.dirname(curr):
        if os.path.isdir(os.path.join(curr, "backend")):
            if curr not in sys.path:
                sys.path.insert(0, curr)
            return
        curr = os.path.dirname(curr)

if __name__ == "__main__":
    setup_python_path()

import hashlib
import re
from typing import List, Dict, Any, Optional

import numpy as np

from backend.services.loom_service.substrate.transformer import LoomTransformer
from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("ShardCreator")


class ShardCreator:
    """
    ============================================================================
    DOCLOOM — TRUE ATOMIC SHARD INITIALIZATION ENGINE
    ============================================================================

    Converts raw language into a semantic neuron-state.

    Mathematical Model:
    ---------------------------------------------------------------------------
        Shard S_i = (D_i, V_i, X_i)

        D_i : Deterministic substrate (exact text)
        V_i : Semantic state vector
        X_i : Dynamic runtime-capable state

    Vector Construction:
    ---------------------------------------------------------------------------
        V_i = Σ (phi_k * e_k)

        phi_k : semantic contribution weight
        e_k   : semantic primitive embedding

    Architecture Goals:
    ---------------------------------------------------------------------------
    - Deterministic substrate preservation
    - Stable semantic geometry
    - Explainable semantic composition
    - Runtime-ready neuron state
    - Future compatibility with:
        * Graph reasoning
        * HDC/VSA
        * Activation propagation
        * Semantic field dynamics
    ============================================================================
    """

    # -------------------------------------------------------------------------
    # INIT
    # -------------------------------------------------------------------------

    def __init__(self):
        self.transformer = LoomTransformer()

        # Weak semantic-energy tokens
        self.glue_words = {
            "to", "us", "in", "the", "and", "a", "an",
            "of", "for", "with", "at", "by", "from",
            "on", "is", "are", "was", "were", "be",
            "been", "being", "that", "this", "it"
        }

        # Persistent semantic basis memory
        # IMPORTANT:
        # Stable concept vectors prevent semantic drift
        self.semantic_basis_cache: Dict[str, np.ndarray] = {}

        log_service(
            logger,
            "ShardCreator initialized with semantic neuron-state architecture.",
            "info"
        )

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def create_shard(self, text: str) -> Dict[str, Any]:
        """
        Main Atomic Shard Initialization Pipeline.
        """

        # ---------------------------------------------------------------------
        # STEP 1 — DETERMINISTIC SUBSTRATE
        # ---------------------------------------------------------------------

        substrate = self._normalize_text(text)

        if not substrate:
            raise ValueError("Cannot create shard from empty text.")

        # ---------------------------------------------------------------------
        # STEP 2 — CONCEPT EXTRACTION
        # ---------------------------------------------------------------------

        concepts = self._extract_semantic_concepts(substrate)

        if not concepts:
            raise ValueError("No semantic concepts extracted.")

        # ---------------------------------------------------------------------
        # STEP 3 — SEMANTIC BASIS VECTOR CREATION
        # ---------------------------------------------------------------------

        concept_embeddings = self._get_or_create_concept_embeddings(concepts)

        # ---------------------------------------------------------------------
        # STEP 4 — PHI WEIGHT CALCULATION
        # ---------------------------------------------------------------------

        phi_weights = self._calculate_phi_weights(
            concepts,
            concept_embeddings
        )

        # ---------------------------------------------------------------------
        # STEP 5 — VECTOR CONSTRUCTION
        # V_i = Σ (phi_k * e_k)
        # ---------------------------------------------------------------------

        final_vector = self._construct_semantic_vector(
            concept_embeddings,
            phi_weights
        )

        # ---------------------------------------------------------------------
        # STEP 6 — SHARD ID
        # ---------------------------------------------------------------------

        shard_id = self._generate_deterministic_id(substrate)

        # ---------------------------------------------------------------------
        # STEP 7 — PHI MAP
        # ---------------------------------------------------------------------

        phi_map = {
            concept: round(float(weight), 6)
            for concept, weight in zip(concepts, phi_weights)
        }

        # ---------------------------------------------------------------------
        # STEP 8 — SHARD CONSTRUCTION
        # ---------------------------------------------------------------------

        shard = {
            # -----------------------------------------------------------------
            # Core Identity
            # -----------------------------------------------------------------
            "id": shard_id,

            # -----------------------------------------------------------------
            # Deterministic Substrate
            # -----------------------------------------------------------------
            "content": substrate,

            # -----------------------------------------------------------------
            # Semantic State
            # -----------------------------------------------------------------
            "vector": final_vector.tolist(),

            # -----------------------------------------------------------------
            # Semantic Composition
            # -----------------------------------------------------------------
            "phi": phi_map,

            # -----------------------------------------------------------------
            # Runtime-Capable State
            # -----------------------------------------------------------------
            "state": {
                "activation": 0.1,
                "truth_score": 0.5,
                "energy": 0.0,
                "times_activated": 0
            },

            # -----------------------------------------------------------------
            # Metadata
            # -----------------------------------------------------------------
            "meta": {
                "dimension": int(len(final_vector)),
                "concept_count": len(concepts),
                "normalization": "l2",
                "vector_model": "semantic-composition-v1"
            }
        }

        log_service(
            logger,
            f"Shard created: {shard_id[:12]} | Concepts={len(concepts)}",
            "debug"
        )

        return shard

    # -------------------------------------------------------------------------
    # TEXT NORMALIZATION
    # -------------------------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        """
        Normalize substrate while preserving semantic meaning.
        """

        text = text.strip()

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)

        return text

    # -------------------------------------------------------------------------
    # CONCEPT EXTRACTION
    # -------------------------------------------------------------------------

    def _extract_semantic_concepts(self, text: str) -> List[str]:
        """
        Extract semantic primitives.

        Two-Level Concept Strategy:
        -----------------------------------------------------------------------
        1. Primitive concepts
        2. Compound semantic units
        """

        extracted = self.transformer.extract_concepts(text)

        cleaned = []

        for concept in extracted:

            concept = concept.strip().lower()

            if not concept:
                continue

            if concept in self.glue_words:
                continue

            if len(concept) < 2:
                continue

            cleaned.append(concept)

        # Remove duplicates while preserving order
        unique_concepts = list(dict.fromkeys(cleaned))

        # Fallback
        if not unique_concepts:

            words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

            unique_concepts = [
                w for w in words
                if w not in self.glue_words and len(w) > 2
            ]

        return unique_concepts

    # -------------------------------------------------------------------------
    # SEMANTIC BASIS MEMORY
    # -------------------------------------------------------------------------

    def _get_or_create_concept_embeddings(
        self,
        concepts: List[str]
    ) -> List[np.ndarray]:
        """
        Stable semantic basis layer.

        IMPORTANT:
        -----------------------------------------------------------------------
        Each concept must maintain stable semantic geometry across the system.
        """

        uncached = [
            c for c in concepts
            if c not in self.semantic_basis_cache
        ]

        if uncached:

            embeddings = self.transformer.get_embeddings(uncached)

            for concept, embedding in zip(uncached, embeddings):

                vector = np.array(embedding, dtype=np.float32)

                # Normalize basis vector
                norm = np.linalg.norm(vector)

                if norm > 0:
                    vector = vector / norm

                self.semantic_basis_cache[concept] = vector

        return [
            self.semantic_basis_cache[c]
            for c in concepts
        ]

    # -------------------------------------------------------------------------
    # PHI WEIGHTING
    # -------------------------------------------------------------------------

    def _calculate_phi_weights(
        self,
        concepts: List[str],
        embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """
        Semantic contribution weighting.

        IMPORTANT:
        -----------------------------------------------------------------------
        Uniform weighting destroys semantic hierarchy.

        This implementation approximates semantic importance using:
        - concept length
        - embedding energy
        """

        scores = []

        for concept, emb in zip(concepts, embeddings):

            # Embedding semantic energy
            emb_energy = np.linalg.norm(emb)

            # Longer concepts generally carry more semantic specificity
            lexical_weight = np.log(len(concept) + 1)

            score = emb_energy * lexical_weight

            scores.append(score)

        scores = np.array(scores, dtype=np.float32)

        total = np.sum(scores)

        if total <= 0:
            return np.ones(len(concepts)) / len(concepts)

        normalized = scores / total

        return normalized

    # -------------------------------------------------------------------------
    # VECTOR CONSTRUCTION
    # -------------------------------------------------------------------------

    def _construct_semantic_vector(
        self,
        embeddings: List[np.ndarray],
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Construct semantic neuron-state vector.

        V_i = Σ (phi_k * e_k)
        """

        dimension = len(embeddings[0])

        final_vector = np.zeros(dimension, dtype=np.float32)

        for weight, emb in zip(weights, embeddings):

            final_vector += weight * emb

        # ---------------------------------------------------------------------
        # NORMALIZATION
        # ---------------------------------------------------------------------

        norm = np.linalg.norm(final_vector)

        if norm > 0:
            final_vector = final_vector / norm

        return final_vector

    # -------------------------------------------------------------------------
    # DETERMINISTIC IDENTITY
    # -------------------------------------------------------------------------

    def _generate_deterministic_id(self, content: str) -> str:
        """
        Content-addressable deterministic identity.

        Blake2b chosen for:
        - speed
        - cryptographic stability
        - deterministic reproducibility
        """

        hasher = hashlib.blake2b(digest_size=16)

        hasher.update(content.encode("utf-8"))

        return f"shard_{hasher.hexdigest()}"

    # -------------------------------------------------------------------------
    # OPTIONAL — HDC BASIS VECTOR
    # -------------------------------------------------------------------------

    def create_hdc_vector(
        self,
        dimension: int = 10000,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Create Hyperdimensional Computing basis vector.

        e_k ∈ {-1,+1}^D

        These vectors are:
        - random
        - nearly orthogonal
        - extremely noise resistant
        """

        if seed is not None:
            np.random.seed(seed)

        return np.random.choice(
            [-1, 1],
            size=dimension
        ).astype(np.int8)


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":

    creator = ShardCreator()

    text = (
        "God help us to live in the earth "
        "and understand surrounding deeply"
    )

    shard = creator.create_shard(text)

    import json

    print(json.dumps(shard, indent=2))