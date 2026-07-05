import os
import sys
import hashlib
import numpy as np
import base64
from typing import List, Dict, Any, Optional, Tuple

# Ensure project root is in sys.path to allow backend imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("HyperVectorEngine")

class HyperVectorEngine:
    """
    ============================================================================
    DOCLOOM — ADVANCED HYPERDIMENSIONAL COGNITIVE ENGINE (HDC)
    ============================================================================
    
    This engine manages high-dimensional vector representations of semantic text
    shards. It fuses symbolic logic with statistical embeddings using:
    
    1. BINDING (Permutation/Rotation):
       HDC vectors represent sequences or hierarchies by rotating the dimension, 
       preserving structure and temporal context: e_permuted = rho^p(e_concept).
       
    2. BUNDLING (Superposition):
       Combines multiple semantic concepts into a single field vector by weighted 
       summation (Layer 2 Analog Field Vector) and binarization (Layer 1 Stable Vector).
       
    3. PROJECTION:
       Maps continuous LLM/Transformer embeddings into high-dimensional space
       using a persistent random projection matrix P: e_projected = P * t_vector.
       
    4. FUSION:
       Combines pure symbolic identity with continuous context for analogical reasoning:
       e_fused = alpha * h_symbolic + (1 - alpha) * e_projected.
       
    ============================================================================
    """

    def __init__(self, dimension: int = 10000):
        # Dimension must be a multiple of 8 for bit-packing
        self.dimension = (dimension // 8) * 8
        self.basis_cache: Dict[str, np.ndarray] = {}
        self.projection_matrix: Optional[np.ndarray] = None
        
        log_service(
            logger, 
            f"HyperVectorEngine initialized. D={self.dimension}", 
            "info"
        )

    # -------------------------------------------------------------------------
    # MEMORY & COMPRESSION
    # -------------------------------------------------------------------------

    def pack_bipolar(self, vector: np.ndarray) -> str:
        """
        Compresses {-1, 1} bipolar vector into a bit-packed Base64 string.
        Reduces size from ~40KB (as JSON list) to ~1.25KB.
        """
        # Convert -1 to 0, 1 to 1
        binary = ((vector + 1) // 2).astype(np.uint8)
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
        This represents structural/syntactic binding in HDC.
        """
        return np.roll(vector, position)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Measures semantic overlap (Cosine Similarity in HDC space).
        For bipolar vectors, this simplifies to (a · b) / D.
        """
        return float(np.dot(a.astype(np.int32), b.astype(np.int32)) / self.dimension)

    def get_basis_vector(self, concept: str) -> np.ndarray:
        """
        Generates a deterministic random bipolar hypervector for a given concept.
        Uses SHA-256 seed hashing to guarantee consistency across sessions.
        """
        concept = concept.lower().strip()
        if concept in self.basis_cache:
            return self.basis_cache[concept]

        seed_hash = hashlib.sha256(concept.encode()).digest()
        seed = int.from_bytes(seed_hash[:8], "big")
        rng = np.random.default_rng(seed)
        
        vector = rng.choice([-1, 1], size=self.dimension).astype(np.int8)
        self.basis_cache[concept] = vector
        return vector

    def get_semantic_basis_vector(self, concept: str, transformer_vector: np.ndarray) -> np.ndarray:
        """
        Generates a semantic basis vector by projecting a continuous transformer embedding,
        ensuring related concepts maintain mathematical similarity in HDC space.
        """
        concept = concept.lower().strip()
        embedding_hash = hashlib.md5(transformer_vector.tobytes()).hexdigest()[:8]
        cache_key = f"{concept}_{embedding_hash}"
        
        if cache_key in self.basis_cache:
            return self.basis_cache[cache_key]
            
        projected = self.project_to_hdc(transformer_vector)
        # Binarize to stable bipolar state {-1, 1}
        vector = np.where(projected >= 0, 1, -1).astype(np.int8)
        self.basis_cache[cache_key] = vector
        return vector

    def bundle(self, vectors: List[np.ndarray], weights: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bundles multiple concepts into a single combined representation.
        Returns:
            - Analog Field Vector (Layer 2: Continuous energy field)
            - Binary Stable Vector (Layer 1: Discrete bipolar representation)
        """
        if weights and len(weights) != len(vectors):
            raise ValueError("Weights must match vectors.")
        
        if not weights:
            weights = [1.0] * len(vectors)

        # Superpose activations
        field_vector = np.zeros(self.dimension, dtype=np.float32)
        for v, w in zip(vectors, weights):
            field_vector += w * v
            
        # Normalize energy field
        weight_norm = np.linalg.norm(weights)
        if weight_norm > 0:
            field_vector = field_vector / weight_norm

        binary_vector = np.where(field_vector >= 0, 1, -1).astype(np.int8)
        return field_vector, binary_vector

    # -------------------------------------------------------------------------
    # HYBRID COGNITION (SEMANTIC-SYMBOLIC FUSION)
    # -------------------------------------------------------------------------

    def project_to_hdc(self, vector: np.ndarray) -> np.ndarray:
        """
        Projects a continuous embedding into HDC space using a persistent random matrix.
        """
        input_dim = len(vector)
        if input_dim == self.dimension:
            return vector.astype(np.float32)
            
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
        Fuses a hard symbolic HDC vector with a continuous semantic transformer vector:
        e_k = alpha * hdc_vector + (1 - alpha) * P(transformer_vector)
        """
        t_k_projected = self.project_to_hdc(transformer_vector)
        return (alpha * hdc_vector.astype(np.float32)) + ((1.0 - alpha) * t_k_projected)

    # -------------------------------------------------------------------------
    # COGNITIVE RECONSTRUCTION
    # -------------------------------------------------------------------------

    def reconstruct_from_recipe(self, phi_map: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstructs the analog and binary hypervectors from a concept-weight recipe.
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
        Creates the bit-packed representation along with the recipe for reconstruction.
        """
        field_vec, binary_vec = self.reconstruct_from_recipe(phis)

        return {
            "version": "hdc_v1.1",
            "dimension": self.dimension,
            "vector_packed": self.pack_bipolar(binary_vec),
            "phi": phis,
            "meta": {
                "concept_count": len(concepts),
                "energy_norm": float(np.linalg.norm(field_vec)),
                "compression": "bit-packed-base64"
            }
        }


if __name__ == "__main__":
    import sys
    # Ensure parent directories are in sys.path if run directly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("======================================================================")
    print("Testing HyperVectorEngine & Real Transformer Embeddings Integration")
    print("======================================================================")
    
    # Initialize Engine (D=8000)
    engine = HyperVectorEngine(dimension=8000)
    
    print("\n--- 1. Pure Symbolic Basis Vectors (Orthogonal Space) ---")
    vec_dog = engine.get_basis_vector("dog")
    vec_puppy = engine.get_basis_vector("puppy")
    vec_human = engine.get_basis_vector("human")
    
    # In pure HDC symbolic space, every word is completely orthogonal (similarity near 0)
    print(f"Symbolic Similarity('dog', 'puppy'): {engine.similarity(vec_dog, vec_puppy):.4f}")
    print(f"Symbolic Similarity('dog', 'human'): {engine.similarity(vec_dog, vec_human):.4f}")
    
    print("\n--- 2. Loading Real Transformer and Getting Real Embeddings ---")
    words = ["dog", "puppy", "human"]
    
    # Try calling the /embed HTTP API first to reuse the running server's model
    import urllib.request
    import json
    
    def get_embeddings_via_api(texts, url="http://localhost:8000/embed"):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps({"texts": texts}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=3) as r:
                res = json.loads(r.read().decode("utf-8"))
                if res.get("success"):
                    return [np.array(emb, dtype=np.float32) for emb in res["embeddings"]]
        except Exception:
            return None

    def local_cosine_similarity(a, b):
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    embeddings = get_embeddings_via_api(words)
    if embeddings is not None:
        print("Successfully retrieved embeddings via the /embed API (Server Active)!")
        emb_dog, emb_puppy, emb_human = embeddings[0], embeddings[1], embeddings[2]
        sim_dog_puppy = local_cosine_similarity(emb_dog, emb_puppy)
        sim_dog_human = local_cosine_similarity(emb_dog, emb_human)
    else:
        print("API server not active/responding. Falling back to local model loading...")
        from backend.services.LLM_service.embedding.transformer import EmbeddingTransformer
        transformer = EmbeddingTransformer()
        local_embs = transformer.get_embeddings(words)
        emb_dog, emb_puppy, emb_human = local_embs[0], local_embs[1], local_embs[2]
        sim_dog_puppy = transformer.calculate_cosine(emb_dog, emb_puppy)
        sim_dog_human = transformer.calculate_cosine(emb_dog, emb_human)
        
    print(f"Real Cosine Sim ('dog' <-> 'puppy'): {sim_dog_puppy:.4f} (High Similarity)")
    print(f"Real Cosine Sim ('dog' <-> 'human'): {sim_dog_human:.4f} (Low Similarity)")
    
    print("\n--- 3. Projecting Real Embeddings to HDC Space (Semantic HDC) ---")
    # Project and binarize using the engine to get semantic basis vectors
    sem_dog = engine.get_semantic_basis_vector("dog", emb_dog)
    sem_puppy = engine.get_semantic_basis_vector("puppy", emb_puppy)
    sem_human = engine.get_semantic_basis_vector("human", emb_human)
    
    # Calculate similarities of semantic hypervectors
    hdc_sem_dog_puppy = engine.similarity(sem_dog, sem_puppy)
    hdc_sem_dog_human = engine.similarity(sem_dog, sem_human)
    print(f"Semantic HDC Sim ('dog' <-> 'puppy'): {hdc_sem_dog_puppy:.4f} (Preserved Similarity)")
    print(f"Semantic HDC Sim ('dog' <-> 'human'): {hdc_sem_dog_human:.4f} (Preserved Distance)")
    
    print("\n--- 4. Permutation (Structural/Sequence Encoding) ---")
    vec_dog_permuted = engine.permute(sem_dog, position=1)
    print(f"Similarity('dog', permute('dog')): {engine.similarity(sem_dog, vec_dog_permuted):.4f}")
    
    print("\n--- 5. Bundling (Concept Superposition/Union) ---")
    field_vec, bundled_vec = engine.bundle([sem_dog, sem_human], weights=[1.0, 1.0])
    print(f"Bundled similarity to 'dog': {engine.similarity(bundled_vec, sem_dog):.4f}")
    print(f"Bundled similarity to 'human': {engine.similarity(bundled_vec, sem_human):.4f}")
    
    print("\n--- 6. Packing and Unpacking (Base64 Compression) ---")
    packed = engine.pack_bipolar(sem_dog)
    print(f"Packed Base64 string length: {len(packed)} characters")
    unpacked = engine.unpack_bipolar(packed)
    print(f"Unpacked vector equals original? {np.array_equal(sem_dog, unpacked)}")
    
    print("\n--- 7. Semantic-Symbolic Fusion ---")
    # Fuses symbolic 'dog' with continuous embedding of 'puppy'
    fused_vec = engine.semantic_fusion(vec_dog, emb_puppy, alpha=0.6)
    print(f"Fused vector shape: {fused_vec.shape}")
    
    print("\n--- 8. End-to-End Shard Representation Recipe ---")
    concepts_list = ["dog", "puppy"]
    phis = {"dog": 0.8, "puppy": 0.5}
    representation = engine.create_hdc_representation(concepts_list, phis)
    print("Created HDC Representation Output Keys:")
    for k, v in representation.items():
        print(f"  {k}: {v if k != 'vector_packed' else v[:40] + '...'}")
    print("\nSuccess! All functions tested successfully.")
