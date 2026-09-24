"""
DocLoom Universal Knowledge & Perception Ingestor
Ingests structured text, multi-page books, 2-column documents, diagrams, and photos
into the Dual-Manifold 'Mirror World' Crystalline Memory without data duplication.
Strictly compliant with Rule 1 (<= 700 lines).
"""
import os
import re
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import torch

from module_AI.model.tokenizer import encode
from module_AI.model.docloom_model import DocLoomModel


@dataclass
class MirrorWorldRecord:
    """Represents a unified Dual-Manifold 'Mirror World' binding."""
    shard_id: str
    symbolic_text: str
    visual_ref: Optional[str] = None
    layout_bbox: Optional[List[float]] = None
    figure_ref: Optional[str] = None
    source_uri: str = "memory"
    timestamp: float = field(default_factory=time.time)


@dataclass
class IngestedDocumentShard:
    """A prepared crystalline memory shard ready for physical ingestion."""
    shard_id: str
    vector: np.ndarray
    text: str
    mass: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeIngestor:
    """
    Universal Cognitive Perception Gateway.
    Transduces multi-modal documents, images, and text into zero-duplication
    crystalline shards and commits them directly to WeaveBrainCoordinator.
    """

    def __init__(self, coordinator, cortex_model: Optional[DocLoomModel] = None, dimension: int = 128):
        self.coordinator = coordinator
        self.cortex_model = cortex_model
        self.dimension = dimension
        self.seen_hashes: set = set()

    def _hash_text(self, text: str) -> str:
        """Deterministic sha256 fingerprint for deduplication."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]

    def _generate_vector(self, text: str) -> np.ndarray:
        """
        Generates native 128-d crystal vector.
        Prefers Cortex's native embedding head; falls back to deterministic projection.
        """
        if self.cortex_model is not None:
            try:
                self.cortex_model.eval()
                with torch.no_grad():
                    device = next(self.cortex_model.parameters()).device
                    token_ids = torch.tensor([encode(text)[:self.cortex_model.cfg.block_size]], device=device)
                    emb = self.cortex_model.embed_text_crystal(token_ids)
                    vec = emb[0].cpu().numpy().astype(np.float32)
                    if len(vec) == self.dimension:
                        return vec
            except Exception:
                pass

        # Deterministic semantic hash projection fallback (128-d unit sphere)
        raw_hash = hashlib.sha512(text.encode("utf-8")).digest()
        seed = int.from_bytes(raw_hash[:8], byteorder="big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dimension).astype(np.float32)
        norm = np.linalg.norm(v)
        return v / (norm if norm > 1e-9 else 1.0)

    def _generate_visual_vector(self, pixel_tensor: torch.Tensor) -> Tuple[np.ndarray, str]:
        """
        Encodes visual pixel patches through the native thalamic stem.
        Returns: (128-d crystal vector, visual fingerprint hash).
        """
        if self.cortex_model is not None:
            try:
                self.cortex_model.eval()
                with torch.no_grad():
                    device = next(self.cortex_model.parameters()).device
                    _, emb = self.cortex_model.embed_visual_patches(pixel_tensor.to(device))
                    vec = emb[0].cpu().numpy().astype(np.float32)
                    fp = hashlib.sha256(vec.tobytes()).hexdigest()[:16]
                    return vec, fp
            except Exception:
                pass

        # Deterministic fallback for image tensor
        flat = pixel_tensor.flatten()[:512].cpu().numpy()
        seed = int(np.abs(flat).sum() * 1000) % (2**31)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dimension).astype(np.float32)
        norm = np.linalg.norm(v)
        vec = v / (norm if norm > 1e-9 else 1.0)
        fp = hashlib.sha256(vec.tobytes()).hexdigest()[:16]
        return vec, fp

    def chunk_document_text(self, text: str, max_words: int = 120, overlap_words: int = 20) -> List[Dict[str, Any]]:
        """
        Sentence-bounded sliding-window chunker with section-path preservation.
        Prevents breaking thoughts mid-sentence.
        """
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_section = "General"

        for p in paragraphs:
            p_strip = p.strip()
            if not p_strip:
                continue

            # Heading detection (# Header or Section Title)
            if p_strip.startswith('#') or (len(p_strip) < 60 and not p_strip.endswith('.')):
                current_section = p_strip.lstrip('#').strip()
                continue

            words = p_strip.split()
            if len(words) <= max_words:
                chunks.append({
                    "text": p_strip,
                    "section": current_section,
                    "word_count": len(words)
                })
            else:
                # Sliding window chunking
                step = max(max_words - overlap_words, 10)
                for i in range(0, len(words), step):
                    window = words[i:i + max_words]
                    chunks.append({
                        "text": " ".join(window),
                        "section": current_section,
                        "word_count": len(window)
                    })

        return chunks

    def ingest_text_document(
        self,
        text: str,
        source_uri: str = "doc://raw_text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Ingests a text or Markdown document into .loom crystalline memory.
        Guarantees zero duplicate shards and establishes causal transitions.
        """
        chunks = self.chunk_document_text(text)
        ingested_ids = []
        base_meta = dict(metadata or {})

        prev_shard_id = None
        for idx, chunk in enumerate(chunks):
            content = chunk["text"]
            content_hash = self._hash_text(content)

            # Deduplication Check
            if content_hash in self.seen_hashes:
                continue
            self.seen_hashes.add(content_hash)

            shard_id = f"doc_{content_hash}_{idx:04d}"
            vec = self._generate_vector(content)

            shard_meta = {
                **base_meta,
                "source_uri": source_uri,
                "section": chunk["section"],
                "content_hash": content_hash,
                "manifold": "symbolic_text",
                "chunk_idx": idx,
                "total_chunks": len(chunks)
            }

            self.coordinator.ingest_shard(
                shard_id=shard_id,
                true_vector=vec,
                text=content,
                mass=1.0,
                metadata=shard_meta
            )
            ingested_ids.append(shard_id)

            # Establish causal transition across sequential paragraphs
            if prev_shard_id is not None and hasattr(self.coordinator, "causal"):
                self.coordinator.causal.record_transition(prev_shard_id, shard_id, weight=1.0, prediction_error=0.3)
            prev_shard_id = shard_id

        return ingested_ids

    def ingest_image_with_mirror_binding(
        self,
        image_pixels: torch.Tensor,
        descriptive_text: str,
        figure_ref: Optional[str] = None,
        layout_bbox: Optional[List[float]] = None,
        source_uri: str = "img://photo",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        The 'Mirror World' Ingestion Gate:
        Binds a visual image/diagram (Side A) with its explanatory text/caption (Side B)
        into a synchronized cross-modal crystal shard without duplicating content.
        """
        v_vec, v_fingerprint = self._generate_visual_vector(image_pixels)
        t_vec = self._generate_vector(descriptive_text)

        # Coordinate Invariant: Unified fused centroid
        fused_vec = (v_vec + t_vec) / 2.0
        norm = np.linalg.norm(fused_vec)
        fused_vec = fused_vec / (norm if norm > 1e-9 else 1.0)

        record_hash = self._hash_text(descriptive_text + v_fingerprint)
        shard_id = f"mirror_{record_hash}"

        meta = {
            **(metadata or {}),
            "source_uri": source_uri,
            "figure_ref": figure_ref or "Figure",
            "layout_bbox": layout_bbox or [0.0, 0.0, 1.0, 1.0],
            "visual_fingerprint": v_fingerprint,
            "manifold": "mirror_world_bipartite",
            "has_visual_grounding": True,
            "text": descriptive_text
        }

        # Commit to living memory crystal
        self.coordinator.ingest_shard(
            shard_id=shard_id,
            true_vector=fused_vec,
            text=descriptive_text,
            mass=1.5,  # Higher cognitive gravitational mass for multi-modal facts
            metadata=meta
        )

        return shard_id, v_fingerprint

    def ingest_multi_column_page(
        self,
        left_column_text: str,
        right_column_text: str,
        page_num: int,
        source_uri: str,
        diagram_box: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Solves multi-column PDF reading order without OCR confusion.
        Ingests left column first, then right column, binding to page diagrams.
        """
        ingested = []
        # Column 1
        c1_ids = self.ingest_text_document(
            left_column_text,
            source_uri=f"{source_uri}#page_{page_num}_col_left",
            metadata={"page": page_num, "column": "left"}
        )
        ingested.extend(c1_ids)

        # Column 2
        c2_ids = self.ingest_text_document(
            right_column_text,
            source_uri=f"{source_uri}#page_{page_num}_col_right",
            metadata={"page": page_num, "column": "right"}
        )
        ingested.extend(c2_ids)

        # Optional Diagram attached to this page
        if diagram_box and "pixels" in diagram_box:
            diag_id, _ = self.ingest_image_with_mirror_binding(
                image_pixels=diagram_box["pixels"],
                descriptive_text=diagram_box.get("caption", f"Diagram on page {page_num}"),
                figure_ref=diagram_box.get("figure_ref", f"Fig_P{page_num}"),
                layout_bbox=diagram_box.get("bbox", [0.1, 0.1, 0.9, 0.5]),
                source_uri=f"{source_uri}#page_{page_num}_diagram"
            )
            ingested.append(diag_id)

            # Connect diagram to both columns
            if c1_ids and hasattr(self.coordinator, "causal"):
                self.coordinator.causal.record_transition(c1_ids[-1], diag_id, weight=1.2, prediction_error=0.1)
            if c2_ids and hasattr(self.coordinator, "causal"):
                self.coordinator.causal.record_transition(diag_id, c2_ids[0], weight=1.2, prediction_error=0.1)

        return ingested
