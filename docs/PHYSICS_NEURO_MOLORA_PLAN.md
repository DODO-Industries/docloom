\\\\\# Physics & Neuroscience-Guided Mixture of LoRAs (MoPA)
## Dynamic Layer-by-Layer Skill Routing via Dynamical Phase Coherence & Energy Minimization

---

## 1. Executive Summary & Foundational Axiom

Traditional Mixture of Experts (MoE) uses a learned linear softmax router:
$$g = \text{Softmax}(W_g \cdot x)$$
This naive routing often suffers from **expert collapse**, lacks temporal context, and cannot sense whether the model is grounded in retrieved memory or hallucinating.

**MoPA (Mixture of Physics-guided Adapters)** replaces heuristic gating with **biophysical and statistical mechanics observables** directly derived from DocLoom's living memory field:
1. **Kuramoto Phase Coherence ($r \in [0, 1]$)**: Measures associative agreement among retrieved memory crystals.
2. **Hamiltonian Energy Gap ($\Delta H$)**: Measures cognitive uncertainty and entropy of the working set.
3. **ACT-R Spatiotemporal Activation ($A_i$)**: Measures recency and biological retention of active facts.

---

## 2. Fundamental Distinction: Sensory Modalities vs Cognitive Skills

To prevent architectural confusion as DocLoom evolves, we define a strict neurological boundary:

```
                      EXTERNAL REALITY (PHYSICAL WORLD)
     [ Photons (Vision) ]       [ Sound Waves (Audio) ]       [ Text / Code / JSON ]
              │                           │                              │
              ▼                           ▼                              ▼
  ┌───────────────────────┐   ┌───────────────────────┐      ┌───────────────────────┐
  │  Visual Patch Stem    │   │  Audio Spectral Stem  │      │  Text Embedding Stem  │
  │  Conv2d(3, 576, 16)   │   │  Conv1d(1, 576, 32)   │      │  Embedding(Vocab, 576)│
  └───────────┬───────────┘   └───────────┬───────────┘      └───────────┬───────────┘
              │                           │                              │
              └───────────────────────────┼──────────────────────────────┘
                                          │
                        UNIVERSAL CORTICAL SPIKES: $\mathbf{z} \in \mathbb{R}^{576}$
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    DocLoom 60M Cognitive Cortex       │
                      │  (Unified Latent Physics Substrate)   │
                      └───────────────────┬───────────────────┘
                                          │
                     Dynamic Physics Routing ($r, \Delta H$)
                                          │
                      ┌───────────────────┴───────────────────┐
                      ▼                                       ▼
       [ Skill: Symbolic Math ]               [ Skill: Document Fact QA ]
```

### The Rule:
- **Sensory Perception is a LAYER (The Thalamic Ingestion Gateway)**: Vision, audio, and sensors are **not** skills. Their sole job is converting physical signals (pixels, frequencies) into the cortex's native $576$-dimensional latent language. The 60M cortex doesn't care whether a token originated from a pixel or a word; it processes all tokens with the exact same attention laws.
- **Skills are REASONING ACTIONS (MoPA Adapters)**: Skills (Math, Code, Synthesis, Fact Extraction) operate **inside** the cortex on the unified $576$-d representations.

---

## 3. Mathematical Capacity Limits & Infinite Expansion

### The Linear Subspace Bound (Why Blind Merging Fails)
In a transformer layer with hidden dimension $d = 576$, each weight matrix has rank $576$. A LoRA adapter of rank $r = 16$ occupies a 16-dimensional subspace:
$$\text{Max Purely Orthogonal Skills} = \frac{d}{r} = \frac{576}{16} = 36 \text{ Skills}$$

- Merging $1\text{--}10$ skills directly into base weights ($W_0 + \sum \Delta W_i$) works cleanly.
- Merging $50\text{--}100$ skills into the same base weights causes **Catastrophic Weight Collision**: newer skills overwrite and degrade older skills.

### The MoPA Solution: Isolated Micro-Adapters + Instant Swapping
Instead of destructively baking 100 skills into the base weights:
1. **Base Cortex Frozen**: The 60M parameter brain ($W_0$) stays permanently frozen at $120\text{ MB}$.
2. **Modular Micro-Files**: Each skill is stored as an independent, lightweight file (`math.safetensors`, `code.safetensors`, $3\text{ MB}$ each) in a `skills/` folder.
3. **Dynamic Hot-Swapping**: The Physics Router monitors cognitive state:
   - When calculating algebraic proofs, the Math adapter is active in VRAM.
   - When extracting medical citations, it swaps to the Fact QA adapter.
   - **VRAM Footprint**: Strictly $< 25\text{ MB}$ extra VRAM, allowing **unlimited (1000+) skills with 0% catastrophic forgetting!**

---

## 4. Mathematical Formulation of the Physics Router

For transformer layer $l$, the routing weight for expert $k$ is given by:

$$g_k^{(l)} = \text{Softmax}\left( \mathbf{W}_g^{(l)} \mathbf{h}^{(l)} + \mathbf{\Phi}_k(r, \Delta H, S) \right)$$

Where the **Physics Bias Potential $\mathbf{\Phi}_k$** is analytically defined:

### 1. Grounded Fact Extraction Expert ($k = \text{fact}$)
$$\Phi_{\text{fact}} = \beta_{\text{fact}} \cdot r \cdot \left(1 - \frac{\Delta H}{H_{\max}}\right)$$
*When Kuramoto resonance is high ($r \to 1$) and energy is near the ground state ($\Delta H \to 0$), memory crystals are highly coherent and verified. The model routes heavily to the Fact Grounding adapter.*

### 2. Symbolic & Mathematical Logic Expert ($k = \text{math}$)
$$\Phi_{\text{math}} = \beta_{\text{math}} \cdot (1 - r) \cdot \Delta H$$
*When coherence is disrupted (low $r$) and cognitive energy surges ($\Delta H \gg 0$), the context requires multi-step deduction rather than direct recall. The model routes to the Symbolic Math adapter.*

### 3. Code & Structured Data Expert ($k = \text{code}$)
$$\Phi_{\text{code}} = \beta_{\text{code}} \cdot \mathbb{I}(\text{SyntaxEntropy} > \theta) + \gamma_{\text{code}} \cdot \Delta H$$
*Activated when the token stream exhibits high indentation entropy, programming delimiters (`{`, `}`, `def`, `SELECT`), or schema signatures.*

---

## 5. Hardware & VRAM Feasibility (RTX 3050 4GB)

- **Base DocLoom 60M (FP16)**: $120\text{ MB}$
- **Active LoRA Expert (Single active skill)**: $\approx 8\text{ MB}$
- **Native Embedding Head**: $\approx 2\text{ MB}$
- **Native 2D Patch Stem**: $< 1\text{ MB}$
- **Inference KV Cache (Context 1024)**: $240\text{ MB}$
- **Total Operational VRAM**: **$\approx 371\text{ MB}$** (Well below our $1.5\text{ GB}$ hardware budget!)
