import numpy as np
from typing import List, Set, Tuple, Optional
import hashlib

# A 256-element bit count lookup table for fast Hamming Distance calculations
BIT_COUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def count_bits(array: np.ndarray) -> np.ndarray:
    """
    Computes the number of set bits (1s) for each element in the input uint8 array.
    """
    return BIT_COUNT_LUT[array].sum(axis=-1)

# Root domain templates for Task 4 Label Probing
ROOT_DOMAINS = [
    "Quantum Physics",
    "Artificial Intelligence",
    "Organic Chemistry",
    "Finance & Economics",
    "Information Theory",
    "Computer Science",
    "Thermodynamics",
    "Mathematics",
    "Philosophy & Episteme",
    "Biology",
    "Cognitive Science",
    "Linguistics"
]

def get_packed_hash_signature(name: str) -> np.ndarray:
    """
    Generates a deterministic 128-bit packed bipolar uint8 signature by hashing the name.
    """
    h = hashlib.md5(name.encode("utf-8")).digest()
    return np.frombuffer(h, dtype=np.uint8)

# Precompute packed uint8 vectors for the root domains
CONCEPT_DICT_PACKED = np.array([get_packed_hash_signature(name) for name in ROOT_DOMAINS], dtype=np.uint8)

class DynamicFieldSubstrateEngine:
    """
    ============================================================================
    DOCLOOM — DYNAMIC FIELD SUBSTRATE & RELATIONAL ENTANGLEMENT
    ============================================================================
    Vectorized high-performance engine implementing coordinates, physics tensors,
    HDC packed bit signatures, and causal links in pure memory (RAM).
    ============================================================================
    """
    def __init__(
        self,
        max_nodes: int = 10000,
        max_links: int = 10,
        sigma: float = 0.5,
        tau_collapse: float = 10.0,
        dt: float = 0.1
    ):
        self.max_nodes = max_nodes
        self.max_links = max_links
        self.sigma = sigma
        self.tau_collapse = tau_collapse
        self.dt = dt

        # Node index concept registration
        self.concept_ids: List[str] = []
        self.collapsed_indices: Set[int] = set()
        self.current_node_count: int = 0

        # Memory mapping initialization (now pure RAM)
        self.coords_map = np.zeros((self.max_nodes, 3), dtype=np.float32)
        self.physics_map = np.zeros((self.max_nodes, 3), dtype=np.float32)
        self.hdc_map = np.zeros((self.max_nodes, 16), dtype=np.uint8)
        self.causal_map = np.full((self.max_nodes, self.max_links, 3), -1.0, dtype=np.float32)

    def ingest(self, concept_id: str, embedding: np.ndarray) -> int:
        """
        TASK 1: Real-Time Ingestion Space Matrix Routing
        Converts raw continuous embeddings to packed bipolar uint8 signatures,
        inserts them into the flat array maps, and positions them near nearest semantic neighbors.
        """
        if concept_id in self.concept_ids:
            idx = self.concept_ids.index(concept_id)
            self.physics_map[idx, 2] = 1.0  # Reinforce activation energy
            return idx

        if self.current_node_count >= self.max_nodes:
            raise ValueError("Max node limits exceeded in substrate.")

        idx = self.current_node_count

        # Slices first 128 parameters & extracts bipolar [-1, 1] sign signature
        emb_128 = embedding[:128]
        if len(emb_128) < 128:
            emb_128 = np.pad(emb_128, (0, 128 - len(emb_128)), constant_values=1.0)
        bipolar = np.sign(emb_128)
        bipolar[bipolar == 0] = 1.0

        # Binarize and pack to 16 bytes of uint8
        binary = ((bipolar + 1) // 2).astype(np.uint8)
        packed = np.packbits(binary)
        self.hdc_map[idx] = packed

        # Spatial routing coordinates calculation
        if idx == 0:
            self.coords_map[idx] = np.zeros(3, dtype=np.float32)
        else:
            # Vectorized Hamming distance lookup to place near closest neighbors
            xors = np.bitwise_xor(self.hdc_map[:idx], packed)
            hamming_dists = count_bits(xors)
            nearest_idx = np.argmin(hamming_dists)

            # Generate coordinate slightly perturbed from nearest neighbor
            nearest_pos = self.coords_map[nearest_idx]
            r = np.random.uniform(0.05, 0.2)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            dx = r * np.sin(phi) * np.cos(theta)
            dy = r * np.sin(phi) * np.sin(theta)
            dz = r * np.cos(phi)
            self.coords_map[idx] = nearest_pos + [dx, dy, dz]

        # Physics setup: [Mass = 1.0, Phase = random, Activation = 1.0]
        self.physics_map[idx] = [1.0, np.random.uniform(0.0, 2 * np.pi), 1.0]

        self.concept_ids.append(concept_id)
        self.current_node_count += 1

        return idx

    def run_physics_step(self) -> None:
        """
        TASK 2: Vectorized Physics Simulation & Spawning (Single Tick)
        Computes localized densities, manages Kuramoto phase synchronization, and triggers collapses.
        """
        n = self.current_node_count
        if n <= 1:
            return

        coords = self.coords_map[:n].copy()
        tensors = self.physics_map[:n].copy()
        links = self.causal_map[:n].copy()

        # Pairwise distances: vectorized distance matrix
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)

        # Mass accumulation index calculations: \rho_i
        masses = tensors[:, 0]
        exp_dists = np.exp(-(dists ** 2) / (2 * (self.sigma ** 2)))
        rho = np.sum(masses * exp_dists, axis=1)

        # Check density limits for collapse spawning
        collapse_triggered = False
        for i in range(n):
            # Only collapse nodes that are active, not collapsed, and are baseline nodes (mass < 10)
            if rho[i] > self.tau_collapse and masses[i] < 10.0 and i not in self.collapsed_indices:
                neighborhood = [
                    j for j in range(n) 
                    if dists[i, j] < 3 * self.sigma and j not in self.collapsed_indices and masses[j] < 10.0
                ]
                if len(neighborhood) >= 3:
                    self._trigger_collapse(neighborhood)
                    collapse_triggered = True
                    break

        if collapse_triggered:
            return

        # Task 3: Physics Torque and Attraction Update
        K = 0.5  # Kuramoto phase sync coupling constant
        phases = tensors[:, 1]
        new_phases = phases.copy()
        entangle_forces = np.zeros_like(coords)

        for i in range(n):
            d_theta = 0.0
            node_links = links[i]
            for link in node_links:
                target_idx = int(link[0])
                if target_idx != -1 and target_idx < n:
                    strength = link[1]
                    phase_diff = phases[target_idx] - phases[i]
                    d_theta += strength * np.sin(phase_diff)
                    
                    # Harmonized spatial attraction force
                    harmony = np.cos(phase_diff)
                    entangle_forces[i] += strength * (coords[target_idx] - coords[i]) * harmony
            
            new_phases[i] = (phases[i] + self.dt * K * d_theta) % (2 * np.pi)

        # Subtle spatial repulsion force to prevent clumping of unrelated nodes
        repulsion_forces = np.zeros_like(coords)
        repulsion_radius = 0.3
        repulsion_constant = 0.02
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = dists[i, j]
                if d < repulsion_radius and d > 1e-5:
                    repulsion_forces[i] += (diff[i, j] / d) * repulsion_constant

        # Compute acceleration and step coordinates (drifts slower if node has more mass)
        total_forces = entangle_forces + repulsion_forces
        acceleration = total_forces / masses[:, np.newaxis]
        new_coords = coords + self.dt * acceleration
        
        # Boundary mapping constraint
        new_coords = np.clip(new_coords, -10.0, 10.0)

        # Decay activation energy slowly over ticks
        new_activations = tensors[:, 2] - self.dt * 0.02
        new_activations = np.clip(new_activations, 0.0, 1.0)

        # Commit updates back
        self.coords_map[:n] = new_coords
        self.physics_map[:n, 1] = new_phases
        self.physics_map[:n, 2] = new_activations

    def _trigger_collapse(self, neighborhood: List[int]) -> None:
        """Collapses neighborhood nodes to create a new high-mass macro field anchor."""
        n = self.current_node_count
        if n >= self.max_nodes:
            return

        valid_indices = [idx for idx in neighborhood if idx < n]
        if not valid_indices:
            return

        coords = self.coords_map[valid_indices]
        masses = self.physics_map[valid_indices, 0]

        # 1. Calculate Center of Mass coordinates
        total_mass = np.sum(masses)
        com = np.sum(coords * masses[:, np.newaxis], axis=0) / (total_mass if total_mass > 0 else 1.0)

        # 2. HDC Vector synthesis (Majority voting)
        packed_sigs = self.hdc_map[valid_indices]
        unpacked = np.unpackbits(packed_sigs, axis=1)[:, :128]
        bipolar = (unpacked.astype(np.int8) * 2) - 1
        summed = np.sum(bipolar, axis=0)
        majority = np.sign(summed).astype(np.int8)
        majority[majority == 0] = 1

        # Pack back into 16-byte uint8 row
        binary = ((majority + 1) // 2).astype(np.uint8)
        packed_new = np.packbits(binary)

        # 3. Create Consolidation Anchor
        new_idx = self.current_node_count
        self.coords_map[new_idx] = com
        # Consolidation Anchor has Mass = 100.0
        self.physics_map[new_idx] = [100.0, np.random.uniform(0.0, 2 * np.pi), 1.0]
        self.hdc_map[new_idx] = packed_new

        # Mark components as collapsed
        for idx in valid_indices:
            self.collapsed_indices.add(idx)
        
        macro_id = f"macro_{len(self.collapsed_indices)}"
        self.concept_ids.append(macro_id)
        self.current_node_count += 1

        # TASK 3: Wire up links
        for member_idx in valid_indices:
            self._add_causal_link(member_idx, new_idx, strength=1.0)
            self._add_causal_link(new_idx, member_idx, strength=1.0)

    def _add_causal_link(self, src_idx: int, dst_idx: int, strength: float = 1.0) -> None:
        """Adds a link or updates strength."""
        links = self.causal_map[src_idx]
        found_slot = -1
        empty_slot = -1
        for i in range(self.max_links):
            target = int(links[i, 0])
            if target == dst_idx:
                found_slot = i
                break
            if target == -1 and empty_slot == -1:
                empty_slot = i

        if found_slot != -1:
            self.causal_map[src_idx, found_slot, 1] += strength
        elif empty_slot != -1:
            self.causal_map[src_idx, empty_slot] = [dst_idx, strength, 0.0]

    def record_transition(self, concept_id_a: str, concept_id_b: str) -> None:
        """
        TASK 3: Chronological sequence transition registration.
        """
        if concept_id_a in self.concept_ids and concept_id_b in self.concept_ids:
            idx_a = self.concept_ids.index(concept_id_a)
            idx_b = self.concept_ids.index(concept_id_b)
            self._add_causal_link(idx_a, idx_b, strength=1.0)

    def probe_node_label(self, node_idx: int) -> str:
        """
        TASK 4: Autonomous Associative Probing
        Performs parallel bitwise Hamming distance matching against domain dictionary references.
        """
        if node_idx >= self.current_node_count:
            return "Unknown Node"
        packed_sig = self.hdc_map[node_idx].copy()

        # Bitwise XOR evaluation against CONCEPT_DICT
        xors = np.bitwise_xor(CONCEPT_DICT_PACKED, packed_sig)
        hamming_dists = count_bits(xors)
        similarities = (128 - hamming_dists) / 128.0

        # Sort and return top 3 matching domains
        top_indices = np.argsort(similarities)[::-1][:3]
        parts = []
        for i in top_indices:
            pct = int(similarities[i] * 100)
            parts.append(f"{pct}% {ROOT_DOMAINS[i]}")

        return " / ".join(parts)
