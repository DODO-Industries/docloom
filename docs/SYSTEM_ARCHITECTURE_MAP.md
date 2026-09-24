# DocLoom Complete System Architecture Map & Directory Guide

This document is the **single source of truth** for understanding the `docLoom` codebase, the data flow, folder responsibilities, and how all components connect.

---

## 1. Master System Topology Chart

```
                                  [USER INTERACTION]
                   ┌──────────────────────┴──────────────────────┐
                   │                                             │
      [Document Upload / CLI]                          [Chat / Query API]
                   │                                             │
                   ▼                                             ▼
  ┌─────────────────────────────────┐           ┌─────────────────────────────────┐
  │   backend/services/document/    │           │      backend/app.py             │
  │   • pdf_parser.py               │           │      Unified FastAPI Gateway    │
  │   • docx_parser.py              │           │      (Port 8000)                │
  │   • table_parsing.py            │           └───────────────┬─────────────────┘
  │   • OCR / Image parsers         │                           │
  └────────────────┬────────────────┘                           │
                   │ (Extracted Text Chunks)                    │
                   ▼                                            ▼
  ┌───────────────────────────────────────────────────────────────────────────────┐
  │                       module_loom/services/embedding/                         │
  │                       • transformer.py (MiniLM-L6-v2 384d)                    │
  └───────────────────────────────────────┬───────────────────────────────────────┘
                                          │ 384d Semantic Vector + Text
                                          ▼
  ┌───────────────────────────────────────────────────────────────────────────────┐
  │                        module_loom/services/weaver/                           │
  │                       [The Living Memory Substrate]                           │
  │                                                                               │
  │   weaver_coordinator.py (Brain Boss)                                          │
  │   ├── Ingestion Engine (Shards, Coherent Edges, Micro-Buckets)                │
  │   ├── Spatiotemporal Physics (Kuramoto Oscillators, Exp Decay e^-0.25)        │
  │   ├── Multi-Cluster Embassy Router (atlas_router.py)                          │
  │   │   ├── Primary Crystal Retrieval                                           │
  │   │   ├── Secondary Border Embassy Sweep                                      │
  │   │   └── Deep-Memory Fallback Sweep (Resonance < 3.5)                        │
  │   └── Substrate Layout (substrate_layout.py -> universe.loom & crystal_*.loom)│
  └───────────────────┬───────────────────────────────────────┬───────────────────┘
                      │                                       │
                      │ (Live Node State WebSocket)           │ (Top 8 Distilled Shards)
                      ▼                                       ▼
  ┌───────────────────────────────────────┐   ┌───────────────────────────────────┐
  │       Vizualization/index.html        │   │    module_AI/memory_bridge.py     │
  │   3D Interactive NeuroVisualizer      │   │    • Extracts physics features    │
  │   (WebGL / Three.js glowing clusters) │   │    • Formats candidate memories   │
  └───────────────────────────────────────┘   └───────────────┬───────────────────┘
                                                              │
                                                              ▼
  ┌───────────────────────────────────────────────────────────────────────────────┐
  │                           module_AI/model/                                    │
  │                     [DocLoom 60M Neural Reasoning Cortex]                     │
  │                                                                               │
  │   1. memory_projection.py: 8 Shards ──> 12 Latent Continuous Memory Tokens    │
  │   2. backbone.py: 8-Layer Decoder with MemoryCrossAttention Blocks            │
  │   3. lora.py: Modular LoRA Skill Adapters (QA, Decision, Code-Edit)           │
  │   4. docloom_model.py: Integrated Generation Engine                           │
  └───────────────────────────────────────┬───────────────────────────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │ <|think|> Evidence Citation </|think|>│
                      │ <|answer|> Direct Factual Answer<|end|>│
                      └───────────────────────────────────────┘
```

---

## 2. Directory Matrix: Active vs. Deprecated / Archive

| Directory Path | Role | Status | Description |
| :--- | :--- | :---: | :--- |
| `backend/` | Gateway Server & Parsers | **ACTIVE** | Main FastAPI app, PDF/DOCX/OCR file parsers. |
| `backend/app.py` | Unified Server Entry Point | **ACTIVE** | Mounts all routers on `http://localhost:8000`. |
| `backend/config/` | Server Configuration | **ACTIVE** | Unified environment and logging config. |
| `backend/services/` | File Extraction Services | **ACTIVE** | PDF, Word, Table, and OCR text extractors. |
| `module_loom/` | Memory Substrate Engine | **ACTIVE** | Core Loom physics, crystals, routing, and recall. |
| `module_loom/app.py` | Proxy Entry Point | **CONSOLIDATED** | Proxies directly to `backend.app:app`. |
| `module_loom/services/weaver/` | Core Memory Coordinator | **CRITICAL** | Substrate storage, Kuramoto physics, multi-cluster recall. |
| `module_loom/services/embedding/`| Neural Text Embedder | **ACTIVE** | Converts text to 384-dimensional MiniLM vectors. |
| `module_loom/services/decoder/`| Binary Crystal Inspector | **UTILITY** | Decodes binary `.loom` files to JSON for inspection. |
| `module_loom/services/cortex/` | Early Latent Field Research| **HISTORICAL** | Predictive processing & HDC prototypes (referenced by Weaver). |
| `module_loom/legacy_archive/` | Pre-Weaver Code | **ARCHIVE** | Old prototypes (`TemporalCognitiveStream`, etc.) - **Do Not Edit**. |
| `module_AI/` | Neural Cortex & Models | **ACTIVE** | 60M Transformer decoder, LoRA adapters, training loops. |
| `module_AI/model/` | Model Architecture | **CRITICAL** | Backbone, MemoryProjection, CrossAttention, Checkpoints. |
| `module_AI/data/checkpoints/` | Safetensors Weights | **CRITICAL** | Preserved 20M & 60M Stage A and Stage B model checkpoints. |
| `module_AI/scripts/` | Testing & Data Scaling | **ACTIVE** | `test_multicluster_fusion.py`, `eval_60m_vs_20m.py`. |
| `assets/.brain_data/` | Database Storage | **DATA** | Binary `.loom` files (`crystal_1.loom`, `universe.loom`). |
| `Vizualization/` | 3D NeuroVisualizer GUI | **FRONTEND** | Three.js WebGL interface for visual brain exploration. |

---

## 3. The 3 Core Pillars & How They Connect

### Pillar 1: The Input & Parsing Layer (`backend/services/`)
- **What it does**: Ingests raw user files (books, manuals, chat exports, PDFs) and breaks them into clean text passages.
- **Where it connects**: Calls `EmbeddingTransformer` in `module_loom` to produce 384d vectors and passes them to `WeaveBrainCoordinator.ingest_shard()`.

### Pillar 2: The Living Memory Substrate (`module_loom/services/weaver/`)
- **What it does**: Holds millions of memory shards across binary crystal files (`crystal_*.loom`). Calculates live Kuramoto oscillator wave synchronization, decay, and cross-cluster embassies.
- **Where it connects**:
  1. Broadcasts node positions to `Vizualization/index.html` via WebSockets.
  2. Surfaces the top-ranked shards through `recall_multi_pass()` to `module_AI/memory_bridge.py`.

### Pillar 3: The Reasoning Cortex (`module_AI/`)
- **What it does**: A fast, 60M parameter neural decoder (8 layers, 512 embedding dim).
- **Where it connects**: Takes the 8 top shards from Pillar 2, projects them into **12 latent memory tokens**, injects them directly via cross-attention, and produces structured `<|think|>` evidence citations and direct answers.

---

## 4. API Endpoints Directory (Running on Port 8000)

| Endpoint | Method | Source File | Description |
| :--- | :---: | :--- | :--- |
| `/ai/ask` | `POST` | `module_AI/routes.py` | Query the complete brain: runs Loom recall + 60M generation |
| `/ai/remember` | `POST` | `module_AI/routes.py` | Ingest new text into memory crystals in real time |
| `/ai/stats` | `GET` | `module_AI/routes.py` | Model status, active checkpoint, and memory statistics |
| `/embed` | `POST` | `module_loom/routes/embedding_routes.py` | Generates 384d semantic vectors |
| `/loom/neuro` | `GET` | `module_loom/routes/visualizer_routes.py` | Serves the 3D NeuroVisualizer interface |
| `/loom/testing` | `GET` | `module_loom/routes/testing_routes.py` | Interactive developer testing sandbox |
