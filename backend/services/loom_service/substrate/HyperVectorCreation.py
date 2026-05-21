import sys
import os
import hashlib
import numpy as np
import base64
from typing import List, Dict, Any, Optional, Tuple

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

from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("HyperVectorEngine")

class HyperVectorEngine:
    """
    ============================================================================
    DOCLOOM — ADVANCED HYPERDIMENSIONAL COGNITIVE ENGINE
    ============================================================================
    
    Version: 1.1 (Memory Optimized & Biologically Inspired)
    
    Changes:
    - Bit-packed storage (Base64) to prevent JSON memory explosion.
    - Split-layer representation: Binary Stable Vector + Analog Semantic Field.
    - Recipe-based reconstruction (Seed-only persistence).
    ============================================================================
    """

    def __init__(self, dimension: int = 10000):
        # Dimension must be a multiple of 8 for bit-packing
        self.dimension = (dimension // 8) * 8
        self.basis_cache: Dict[str, np.ndarray] = {}
        self.projection_matrix: Optional[np.ndarray] = None
        
        log_service(
            logger, 
            f"HyperVectorEngine v1.1 initialized. D={self.dimension}", 
            "info"
        )

    # -------------------------------------------------------------------------
    # MEMORY & COMPRESSION
    # -------------------------------------------------------------------------

    def pack_bipolar(self, vector: np.ndarray) -> str:
        """
        Compresses {-1, 1} vector into a bit-packed Base64 string.
        Reduces size from ~40KB (JSON) to ~1.25KB (Base64).
        """
        # Convert -1 to 0, 1 to 1
        binary = ((vector + 1) // 2).astype(np.uint8)
        # Pack bits into bytes
        packed = np.packbits(binary)
        return base64.b64encode(packed).decode('utf-8')

    def unpack_bipolar(self, packed_str: str) -> np.ndarray:
        """
        Decompresses Base64 string back into a {-1, 1} numpy array.
        """
        packed_bytes = base64.b64decode(packed_str)
        binary = np.unpackbits(np.frombuffer(packed_bytes, dtype=np.uint8))
        binary = binary[:self.dimension]
        # Convert 0 to -1, 1 to 1
        return (binary.astype(np.int8) * 2) - 1

    # -------------------------------------------------------------------------
    # CORE HDC OPERATIONS
    # -------------------------------------------------------------------------

    def permute(self, vector: np.ndarray, position: int = 1) -> np.ndarray:
        """
        Rotates the hypervector to encode sequence or structural hierarchy.
        Preserves temporal order, syntax, and structural cognition.
        Example: rho^1(e_god) + rho^2(e_help) + rho^3(e_earth)
        """
        return np.roll(vector, position)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Measures semantic resonance (Cosine Similarity in HDC space).
        For bipolar vectors, this is (a · b) / D.
        """
        return float(np.dot(a, b) / self.dimension)

    def get_basis_vector(self, concept: str) -> np.ndarray:
        """
        Generates a deterministic random hypervector for a concept.
        """
        concept = concept.lower().strip()
        if concept in self.basis_cache:
            return self.basis_cache[concept]

        seed_hash = hashlib.sha256(concept.encode()).digest()
        # Use 64-bit seed (8 bytes) to avoid collision risk at large scale
        seed = int.from_bytes(seed_hash[:8], "big")
        rng = np.random.default_rng(seed)
        
        vector = rng.choice([-1, 1], size=self.dimension).astype(np.int8)
        self.basis_cache[concept] = vector
        return vector

    def get_semantic_basis_vector(self, concept: str, transformer_vector: np.ndarray) -> np.ndarray:
        """
        Generates a semantic basis vector instead of a random one.
        Fixes the issue where random orthogonal vectors overlap incorrectly.
        By projecting a transformer embedding, related concepts (AI, ML) 
        will naturally maintain their correct semantic overlap in HDC space.
        """
        concept = concept.lower().strip()
        # Contextual semantic cache key
        embedding_hash = hashlib.md5(transformer_vector.tobytes()).hexdigest()[:8]
        cache_key = f"{concept}_{embedding_hash}"
        
        if cache_key in self.basis_cache:
            return self.basis_cache[cache_key]
            
        projected = self.project_to_hdc(transformer_vector)
        # Binarize to stable bipolar state
        vector = np.where(projected >= 0, 1, -1).astype(np.int8)
        self.basis_cache[cache_key] = vector
        return vector

    def bundle(self, vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a composite memory state.
        Returns: (Analog Field Vector, Binary Stable Vector)
        """
        if weights and len(weights) != len(vectors):
            raise ValueError("Weights must match vectors.")
        
        if not weights:
            weights = [1.0] * len(vectors)

        # Layer 2: Analog Semantic Field (Preserves energy and intensity)
        field_vector = np.zeros(self.dimension, dtype=np.float32)
        for v, w in zip(vectors, weights):
            field_vector += w * v
            
        # Normalize energy field to prevent activation inflation
        weight_norm = np.linalg.norm(weights)
        if weight_norm > 0:
            field_vector = field_vector / weight_norm

        # Layer 1: Binary Stable Vector (For retrieval and associative memory)
        binary_vector = np.where(field_vector >= 0, 1, -1).astype(np.int8)
        
        return field_vector, binary_vector

    # -------------------------------------------------------------------------
    # HYBRID COGNITION (SEMANTIC-SYMBOLIC FUSION)
    # -------------------------------------------------------------------------

    def project_to_hdc(self, vector: np.ndarray) -> np.ndarray:
        """
        Projects a continuous embedding (e.g., from a transformer) into the HDC space.
        Uses a persistent deterministic random projection matrix P(t_k).
        """
        input_dim = len(vector)
        if input_dim == self.dimension:
            return vector.astype(np.float32)
            
        # Create and persist the projection matrix once
        if self.projection_matrix is None or self.projection_matrix.shape[1] != input_dim:
            weights_dir = os.path.join(os.path.dirname(__file__), "weights")
            os.makedirs(weights_dir, exist_ok=True)
            matrix_path = os.path.join(weights_dir, f"projection_matrix_{input_dim}_{self.dimension}.npy")
            
            if os.path.exists(matrix_path):
                self.projection_matrix = np.load(matrix_path)
            else:
                rng = np.random.default_rng(42)
                self.projection_matrix = rng.normal(
                    0, 
                    1 / np.sqrt(input_dim), 
                    size=(self.dimension, input_dim)
                ).astype(np.float32)
                np.save(matrix_path, self.projection_matrix)
            
        return np.dot(self.projection_matrix, vector).astype(np.float32)

    def semantic_fusion(self, hdc_vector: np.ndarray, transformer_vector: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Fuses a symbolic HDC hypervector (h_k) with a continuous semantic Transformer vector (t_k).
        e_k = α * h_k + (1 - α) * P(t_k)
        Enables nuanced reasoning, analogy, and semantic clustering alongside perfect memory binding.
        """
        t_k_projected = self.project_to_hdc(transformer_vector)
        return (alpha * hdc_vector.astype(np.float32)) + ((1.0 - alpha) * t_k_projected)

    # -------------------------------------------------------------------------
    # COGNITIVE RECONSTRUCTION
    # -------------------------------------------------------------------------

    def reconstruct_from_recipe(self, phi_map: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dynamically regenerates the cognitive state from seeds and weights.
        This is the "Biologically Elegant" approach.
        """
        concepts = list(phi_map.keys())
        weights = list(phi_map.values())
        vectors = [
            self.permute(self.get_basis_vector(c), position=i + 1)
            for i, c in enumerate(concepts)
        ]
        return self.bundle(vectors, weights)

    # -------------------------------------------------------------------------
    # PUBLIC API: SHARD REPRESENTATION
    # -------------------------------------------------------------------------

    def create_hdc_representation(self, concepts: List[str], phis: Dict[str, float]) -> Dict[str, Any]:
        """
        Main pipeline to create the hdc_v1 representation.
        """
        field_vec, binary_vec = self.reconstruct_from_recipe(phis)

        # We store the binary vector (packed) for fast associative lookup,
        # and the PHI map as the "recipe" for regenerating the field vector.
        return {
            "version": "hdc_v1.1",
            "dimension": self.dimension,
            "vector_packed": self.pack_bipolar(binary_vec),
            "phi": phis, # The Recipe
            "meta": {
                "concept_count": len(concepts),
                "energy_norm": float(np.linalg.norm(field_vec)),
                "compression": "bit-packed-base64"
            }
        }

# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    
    engine = HyperVectorEngine()

    concepts = ["god", "earth", "understand"]
    phis = {"god": 0.3, "earth": 0.4, "understand": 0.3}

    # 1. Creation
    hdc_rep = engine.create_hdc_representation(concepts, phis)
    print(f"HDC v1.1 Created. Size: {len(hdc_rep['vector_packed'])} chars (was ~40,000)")

    # 2. Resonance Check
    # To check resonance correctly with sequence encoding, we must permute the concept to its sequence position
    earth_vec = engine.permute(engine.get_basis_vector("earth"), position=2)
    binary_vec = engine.unpack_bipolar(hdc_rep['vector_packed'])
    
    field_vec, stable = engine.reconstruct_from_recipe(hdc_rep['phi'])
    
    similarity = engine.similarity(field_vec, earth_vec)
    print(f"Resonance with 'earth': {similarity:.4f} (Expected > 0)")

    # 3. Reconstruction Check
    print(f"Reconstruction Stability: {np.array_equal(stable, binary_vec)}")


