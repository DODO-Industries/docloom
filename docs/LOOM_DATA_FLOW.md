# DocLoom — The Loom Structure & Data Flow (with the full maths)

> **Who this is for:** you (Atri). You built this by *combining* several ideas — vector
> embeddings, Hyperdimensional Computing (HDC), Locality-Sensitive Hashing (LSH),
> Newtonian "gravity", Kuramoto oscillators, a mass-spring-damper field, active
> inference, and a sleep/consolidation cycle. This document walks a single piece of
> text **all the way through the machine** and writes down the exact formula used at
> every step, so you can see *why* each part is there and *what it computes*.
>
> **How to read it:** each stage has three parts — **(a) What / Why** in plain words,
> **(b) The maths** (the actual equation from the code), and **(c) Where** (file it
> lives in). Leave your comments inline next to any stage; we'll refine from there.
>
> **Two important truths up front:**
> 1. There are **two vector worlds** in DocLoom. Keep them separate in your head:
>    - **The continuous embedding world** (float vectors, e.g. 384-dim from MiniLM).
>      This is what actually flows through routing, gravity, cosine, and recall. This
>      is the *operational* substrate.
>    - **The HDC world** (bipolar `{-1,+1}` vectors, 8000-dim). This is a *symbolic
>      signature / compression* layer built in the document-weave path. It is where
>      "HDC vector creation" happens. It is **not** on the hot recall path today — it
>      is stored as a compact signature and used for symbolic algebra (bind/bundle).
> 2. There are **two write paths**:
>    - **Live path** — `ingest_shard` / `ingest_batch` / `recall` in
>      `weaver_coordinator.py`. Operates on continuous embeddings. This is the main flow.
>    - **Bulk document-weave path** — `SubstrateWeaver.weave()` + `orchestrate_weave()`.
>      Builds an in-memory graph, computes 8000-dim HDC signatures + concepts + runs a
>      small physics sim, then flushes to storage through `ingest_batch`.

---

## 0. The 10,000-foot mental model

Think of the brain as a **procedurally-generated universe of "crystals"**:

```
                    ┌─────────────────────────────────────────────┐
   text ──► embed ──►│  a shard = one memory (a point in space)     │
                    │  a crystal = one .loom file (a galaxy)       │
                    │  the atlas = the star-map of all galaxies    │
                    │  the seed = the laws of physics (a 64-bit #) │
                    └─────────────────────────────────────────────┘

  WRITE (ingest):  text → embedding → mass/energy/phase → seed scaffold
                   → momentum vector → route to a crystal → LSH bucket
                   → append to disk → update ledger + causal graph + run one "thought"

  READ (recall):   query → embedding → route to crystal → jump to LSH bucket
                   → hop along causal graph → cosine → gravity → excitation
                   → rank → re-awaken memories → lock phases (Kuramoto)

  SLEEP (offline): recurring concepts → reinforce → synthesise a summary crystal
                   → mark the originals "superseded" (never deleted)
```

Nothing is ever rewritten in place. The `.loom` file is an **append-only log** (like a
brain that only ever adds neurons); everything else (activation, phase, causal edges,
supersession) is *living state* layered on top.

---

## 1. Symbols & constants (the vocabulary)

| Symbol | Meaning | In code |
|---|---|---|
| $d$ | embedding dimension (e.g. 384 for MiniLM) | `self.dimension` |
| $D$ | HDC dimension (8000) | `HyperVectorEngine(dimension=8000)` |
| $\mathbf{t}$ | the raw transformer embedding of a text | `true_vector` / `emb_vec` |
| $\mathbf{s}$ | the seed **scaffold** (a memory's "resting socket") | `seed_scaffold` |
| $\mathbf{p}$ | the **momentum vector** (where the memory actually lands) | `momentum` |
| $m$ | **mass** of a shard (semantic "heaviness") | `mass` |
| $\theta$ | **phase** angle of a shard (Kuramoto oscillator) | `phase_angle` |
| $A$ | **activation** (how "awake" a memory is, decays over time) | `activation` |
| $\lambda$ | base memory decay rate | `MEMORY_DECAY_RATE = 0.05` |

Key tunable constants (all live in [`backend/.modelTunning`](../backend/.modelTunning), hot-reloadable):

```
SCAFFOLD_WARP          = 1.0     # how hard mass pulls a shard off its seed socket
GRAVITY_DENOM_OFFSET   = 1e-4    # epsilon so gravity never divides by zero
WAVE_EXCITATION_GAIN   = 0.4     # η — strength of the phase-resonance kick
KURAMOTO_COUPLING      = 0.1     # K — how strongly oscillators pull each other into sync
MEMORY_DECAY_RATE      = 0.05    # λ — base forgetting speed
RANK_ACTIVATION_WEIGHT = 0.25    # how much "recently used" nudges ranking
CAUSAL_SPREAD_WEIGHT   = 0.0     # multi-hop spreading (off by default)
RECALL_FULL_SCAN_BELOW = 20000   # below this many shards, don't bother with LSH narrowing
RECALL_SEED_LEADERS    = 24      # how many LSH neighbourhoods to open on recall
```

---

## 2. Stage 0 — Text → Embedding (continuous meaning)

**(a) What / Why.** Everything starts as text. We need a *number vector* whose geometry
encodes meaning: similar sentences → nearby vectors. We use a sentence-transformer
(`all-MiniLM-L6-v2`, 384-dim). This is the one "borrowed, solved" piece — we don't
reinvent semantic embedding, we build a brain *around* it.

**(b) The maths.** The model is a frozen neural net $f_\phi$; we just call it:

$$\mathbf{t} = f_\phi(\text{text}) \in \mathbb{R}^{384}$$

Similarity between two texts is cosine similarity:

$$\text{sim}(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}$$

**(c) Where.** [`embedding_service.py`](../backend/services/LLM_service/embedding/embedding_service.py) —
a 4-tier lookup: (1) reuse the in-process model, (2) HTTP `/embed`, (3) local
SentenceTransformer, (4) a deterministic dummy. The tiers exist so the brain never
*blocks* on a missing embedding server (there's a circuit breaker: after 2 failures it
stops trying the dead endpoint for 120s — otherwise bulk ingest of thousands of shards
would each wait for a socket timeout).

---

## 3. Stage 1 — HDC vector creation (the symbolic world) 🔬

> This is the part you specifically asked about. HDC = **Hyperdimensional Computing**.
> The idea: represent concepts as very long random vectors, where you can *do algebra*
> on meaning (combine, sequence, compare) using cheap operations, and where random
> vectors are almost always near-orthogonal (so different concepts don't collide).
> Everything here lives in
> [`HyperVectorCreation.py`](../backend/services/loom_service/cortex/HyperVectorCreation.py).

We use $D = 8000$ **bipolar** vectors: every element is $-1$ or $+1$.

### 3.1 Basis vector (a concept's random "fingerprint")

**Why.** Every distinct word/concept needs a stable, reproducible vector — the same
across sessions and machines. We derive it deterministically from the concept's text
via a hash, so no storage is needed.

**Maths.** Hash the concept, use it to seed a random generator, draw a bipolar vector:

$$\text{seed} = \text{SHA256}(\text{concept})_{[:8\text{ bytes}]}, \qquad
\mathbf{h}_c = \text{RNG}(\text{seed}).\text{choice}(\{-1,+1\})^{D}$$

**Property (near-orthogonality).** For two *independent* random bipolar vectors, each
coordinate product $h_{c,i}\,h_{c',i}$ is $\pm 1$ with equal probability, so

$$\mathbb{E}\!\left[\frac{\mathbf{h}_c\cdot\mathbf{h}_{c'}}{D}\right] = 0,
\qquad \text{std} \approx \frac{1}{\sqrt{D}} = \frac{1}{\sqrt{8000}} \approx 0.011$$

That's why `similarity('dog','human') ≈ 0` in pure symbolic space — random concepts are
essentially orthogonal. (See test output in the file's `__main__`.)

### 3.2 Similarity (HDC cosine)

For bipolar vectors, $\lVert\mathbf{a}\rVert = \sqrt{D}$ always, so cosine collapses to a
normalised dot product:

$$\text{sim}(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{D}$$

### 3.3 Projection (continuous embedding → HDC) — **the bridge between the two worlds**

**Why.** Pure random basis vectors throw away the *meaning* the transformer learned
(`dog` and `puppy` would be orthogonal). We want HDC vectors that **preserve** semantic
similarity. So we project the 384-dim embedding into 8000-dim with a fixed random matrix
(a **Johnson–Lindenstrauss** random projection: random linear maps approximately
preserve distances/angles).

**Maths.** A persistent projection matrix $P \in \mathbb{R}^{D \times d}$ with entries

$$P_{ij} \sim \mathcal{N}\!\left(0,\ \tfrac{1}{d}\right)$$

(the $1/\sqrt{d}$ scale keeps output magnitudes stable). Then

$$\mathbf{e}_{\text{proj}} = P\,\mathbf{t} \in \mathbb{R}^{D}$$

The matrix is generated once (seed 42) and saved to disk, so the mapping is identical
forever. **Semantic basis vector** = binarise the projection to bipolar:

$$\mathbf{h}^{\text{sem}}_c = \text{sign}(P\,\mathbf{t}) \in \{-1,+1\}^{D}$$

Because JL projection preserves angles, $\text{sim}(\mathbf{h}^{\text{sem}}_{dog},
\mathbf{h}^{\text{sem}}_{puppy})$ stays **high** — semantics survive the jump to HDC.

### 3.4 Binding — `permute` (encode order / structure)

**Why.** To represent *sequences* or *roles* ("A then B" ≠ "B then A") you need an
operation that scrambles a vector reversibly and makes the result dissimilar to the
original. HDC uses a **permutation** (here, a cyclic shift $\rho$).

**Maths.** Shift by $p$ positions:

$$\rho^{p}(\mathbf{v}) = \text{roll}(\mathbf{v}, p)$$

Since a shift moves every $\pm 1$ to a new random-ish slot, $\text{sim}(\mathbf{v},
\rho^{p}(\mathbf{v})) \approx 0$ — so "dog in position 1" and "dog in position 2" are
distinguishable, yet the shift is invertible.

### 3.5 Bundling — `bundle` (superpose several concepts into one)

**Why.** To make a *set/summary* vector that is similar to **all** its members at once
("this shard is about dog AND human"), HDC **adds** the vectors (superposition) and
re-binarises.

**Maths.** Given vectors $\mathbf{v}_1..\mathbf{v}_n$ with weights $w_1..w_n$:

$$\underbrace{\mathbf{F} = \frac{\sum_i w_i \mathbf{v}_i}{\lVert \mathbf{w}\rVert}}_{\text{Layer 2: analog "field" vector}},
\qquad
\underbrace{\mathbf{B} = \text{sign}(\mathbf{F})}_{\text{Layer 1: stable bipolar vector}}$$

The dot with any member stays positive, so $\text{sim}(\mathbf{B}, \mathbf{v}_i) > 0$ for
each member — the bundle "remembers" all of them (until you overload it; a bundle of too
many vectors becomes noise, which is HDC's natural capacity limit). Note the **two
layers**: a continuous field $\mathbf{F}$ (energy) and a discrete signature $\mathbf{B}$.

### 3.6 Fusion — `semantic_fusion` (mix symbolic identity + continuous context)

**Why.** Sometimes you want a vector that is *part* hard symbol, *part* soft meaning —
tunable by $\alpha$.

**Maths.**

$$\mathbf{e}_k = \alpha\,\mathbf{h}_{\text{symbolic}} + (1-\alpha)\,P\,\mathbf{t}$$

$\alpha=1$ → pure symbol; $\alpha=0$ → pure projected meaning.

### 3.7 Compression — `pack_bipolar` (store 1 bit per dimension)

**Why.** A bipolar vector is 1 bit of real information per element. Storing it as JSON
floats wastes ~40 KB; bit-packing → ~1.25 KB.

**Maths.** Map $\{-1,+1\}\to\{0,1\}$, pack 8 to a byte, base64-encode:

$$b_i = \tfrac{v_i + 1}{2}, \qquad \text{bytes} = \text{packbits}(\mathbf{b}), \qquad \text{str} = \text{base64}(\text{bytes})$$

> **Comment hook 💬:** Today the HDC signature is computed and stored, but recall ranks on
> the *continuous* embedding, not the HDC vector. A real design decision for us: do we
> want HDC bind/bundle to become load-bearing (e.g. structural/relational queries), or
> keep it as a compact signature? Note here what you want.

---

## 4. Stage 2 — Physics scalars: mass, energy, phase

Before a shard is placed, we compute three scalars from its embedding. These give each
memory its "physical" personality. (Computed in the weave path,
[`SubstrateWeaver.weave()`](../backend/services/loom_service/cortex/SubstrateWeaver.py#L84-L98);
on the live path `mass` may just be passed in, default $1.0$.)

**Mass** $m$ — semantic heaviness. Longer, denser text → heavier → harder to move and
stronger gravity:

$$m = \log(\text{word\_count}+1)\cdot\sum_i |t_i|, \qquad m \leftarrow \max(0.1,\ m)$$

**Energy** — average activation magnitude:

$$E = \frac{1}{d}\sum_i |t_i|$$

**Phase** $\theta$ — a deterministic angle from the vector itself (splits the vector in
half and takes the angle between the two halves' sums). This seeds each shard's Kuramoto
oscillator with **real variation** — critical, because if every shard started at
$\theta=0$, the Kuramoto coupling would be a permanent no-op ($\sin 0 = 0$):

$$\theta = \left(\operatorname{atan2}\!\Big(\textstyle\sum_{i<d/2} t_i,\ \sum_{i\ge d/2} t_i\Big) + 2\pi\right) \bmod 2\pi$$

**Where.** [`SubstrateWeaver.py:88-93`](../backend/services/loom_service/cortex/SubstrateWeaver.py#L88-L93)
and `_seed_phase()` in
[`weaver_coordinator.py:67-84`](../backend/services/loom_service/weaver/weaver_coordinator.py#L67-L84).

---

## 5. Stage 3 — Seed scaffold & the momentum vector (write geometry) 🌌

**(a) What / Why.** This is one of the most original parts. Instead of storing a shard
exactly where its embedding says, we give every shard a **resting socket** derived purely
from the universe seed and the shard's id — a deterministic random position that needs
**zero disk reads** to reproduce. Then **mass warps the shard away from its socket toward
its true semantic position**. Heavy (important) memories snap to their real meaning; light
ones stay near their procedurally-generated socket. This is "procedural generation of
space" (like a game world generated from a seed).

**(b) The maths.**

*Scaffold (micro socket)* — seed-derived small random vector:

$$\mathbf{s} = \text{RNG}\big(\text{SHA256}(\text{seed}\,\Vert\,\text{"micro\_"}+\text{shard\_id})\big).\mathcal{N}(0,\,0.1)^{d}$$

*Momentum* — interpolate from socket toward the true vector, by a **warp factor** driven
by mass:

$$w = \frac{m}{\text{SCAFFOLD\_WARP} + \max(0,m)}, \qquad
\mathbf{p} = \mathbf{s} + w\,(\mathbf{t} - \mathbf{s}), \qquad
\mathbf{p} \leftarrow \frac{\mathbf{p}}{\lVert\mathbf{p}\rVert}$$

**Read the behaviour of $w$** (with `SCAFFOLD_WARP=1`):
- $m \to 0$  ⟹ $w \to 0$ ⟹ $\mathbf{p} \approx \mathbf{s}$ (light memory stays at its seed socket)
- $m = 1$   ⟹ $w = 0.5$ ⟹ $\mathbf{p}$ is halfway between socket and meaning
- $m \to \infty$ ⟹ $w \to 1$ ⟹ $\mathbf{p} \approx \mathbf{t}$ (heavy memory sits at its true meaning)

**(c) Where.** Scaffold: `UniverseSeedCore.get_micro_socket`
([`seed_core.py:47`](../backend/services/loom_service/weaver/seed_core.py#L47)).
Warp: `LatentFieldPhysicsEngine.calculate_momentum_vector`
([`storage_physics.py:34`](../backend/services/loom_service/weaver/storage_physics.py#L34)).

> The seed core also has **macro** (per-crystal) and **meso** (per-bucket) offsets — the
> same trick at three scales, so the whole coordinate universe is reproducible from one
> 64-bit number.

---

## 6. Stage 4 — Routing: which crystal does this shard join? (the Atlas)

**(a) What / Why.** Crystals (`.loom` files) are galaxies of related memories. The
**atlas** keeps only a lightweight centroid per crystal. A new shard joins the crystal
whose centroid is most similar to its momentum vector — "centroid gravity tracking."

**(b) The maths.** Route to the crystal maximising cosine of momentum vs centroid:

$$\text{target} = \arg\max_{c}\ \frac{\mathbf{p}\cdot\mathbf{c}_{\text{centroid}}}{\lVert\mathbf{p}\rVert\,\lVert\mathbf{c}_{\text{centroid}}\rVert}$$

After placing, the centroid updates as a running mean (centre of mass):

$$\mathbf{c}_{\text{centroid}} \leftarrow \frac{\mathbf{c}_{\text{centroid}}\cdot n + \mathbf{p}}{n+1}, \qquad n \leftarrow n+1$$

**Cellular division.** When every crystal is "full" ($\ge$ `MAX_CRYSTAL_SIZE = 500000`),
the nearest one freezes (`hot → warm`) and **splits**: a new crystal is spawned with a
*perturbed* centroid (kept near the parent so related memories stay close), rescaled to
the parent's magnitude:

$$\mathbf{c}_{\text{new}} = \big(\mathbf{c}_{\text{old}} + 0.05\,\mathbf{o}\big)\cdot\frac{\lVert\mathbf{c}_{\text{old}}\rVert}{\lVert\mathbf{c}_{\text{old}} + 0.05\,\mathbf{o}\rVert}, \quad \mathbf{o} = \text{macro offset}$$

> **Hard-won bug note (in the code):** ingest routes **only** to crystals with headroom.
> Routing to a *full* crystal used to trigger a new split on every hit — a "division
> cascade" that made 1,798 files at 1M shards. Now full crystals still serve recall but
> never receive new shards.

**(c) Where.** [`atlas_router.py`](../backend/services/loom_service/weaver/atlas_router.py) +
`_route_for_ingest` in
[`weaver_coordinator.py:670`](../backend/services/loom_service/weaver/weaver_coordinator.py#L670).

---

## 7. Stage 5 — LSH bucketing: the "resonance-jump" partition ⚡

**(a) What / Why.** Even within one crystal there can be hundreds of thousands of shards.
Scoring all of them per query is O(n). So each crystal **self-organises into
neighbourhoods** using **random-hyperplane LSH**: vectors on the same side of a set of
hyperplanes get the same *signature*, and near vectors get *near* signatures (small
Hamming distance). Recall then "jumps" straight to the right neighbourhood.

**(b) The maths.**

*Signature* — the sign pattern of the first 128 dimensions, packed into 128 bits
(16 bytes):

$$\text{sig}(\mathbf{v}) = \text{packbits}\big(\tfrac{\text{sign}(\mathbf{v}_{[:128]})+1}{2}\big)$$

*Hamming distance* between two signatures = number of differing bits (computed with a
byte-wise popcount lookup table for speed):

$$H(\text{sig}_a,\text{sig}_b) = \sum \text{popcount}(\text{sig}_a \oplus \text{sig}_b)$$

*Assimilate vs accommodate* — find the nearest existing "leader" signature:
- if $H_{\min} \le 32$ (out of 128) → **assimilate** into that neighbourhood (reuse its bucket id)
- else if room remains (`LOOM_MAX_LEADERS = 512`) → **accommodate**: create a new leader
- else → fall back to the nearest leader (never leave a shard unbucketed)

> **Critical correctness note (from the code, 2026-07-13):** the LSH signature must be
> computed on the **semantic vector** $\mathbf{t}$, *not* the scaffold-blended momentum
> $\mathbf{p}$. Reason: recall signatures the *raw query* (which has no per-shard
> scaffold). If ingest bucketed by momentum, semantically identical content would land in
> different buckets (each shard's scaffold differs by id), and a shard could fail to
> retrieve *even its own vector*. Fixing this took self-recall from 1/7 → 7/7.

**(c) Where.** `_momentum_signature`, `_resolve_bucket`
([`weaver_coordinator.py:728-772`](../backend/services/loom_service/weaver/weaver_coordinator.py#L728-L772)).

---

## 8. Stage 6 — Storage append (the `.loom` file)

**(a) What / Why.** The shard's vector + mass + metadata (including its text and bucket
id) are **appended** to the crystal's binary log. Never rewritten. A separate, disposable
`.idx` snapshot (mmap) holds the fast-access columns.

**(b) The format.**

```
universe.loom container ("LUNI")
 ├── 1 KB header (seed, dimension)
 ├── named segments: atlas, snapshot, cortex-state bins
 └── append-only journal tail (TXN records + crystal registrations)

each crystal .loom ("LOM2")            each .idx snapshot ("LIX2", mmap)
 ├── 64-byte header                     ├── offsets[]   (where each record is)
 └── append-only shard/leader records   ├── mass[]      (per-shard mass)
     (id, float32 vector, mass, meta)   ├── norms[]     (‖vector‖, precomputed)
                                        ├── contiguous float32 matrix (all vectors)
                                        ├── leaders[]   (LSH signatures)
                                        ├── ids[]       (shard ids)
                                        └── bucket_ids[](uint16 per shard)
```

The `.idx` is *derived* data — it can be rebuilt from the log at any time (crash-safe:
torn records are rejected, complete ones adopted). `norms[]` is precomputed so recall's
cosine is one matrix-multiply, not a per-row norm.

**(c) Where.** [`substrate_layout.py`](../backend/services/loom_service/weaver/substrate_layout.py)
(`LoomStore.append_batch`, `checkpoint`) and
[`universe_container.py`](../backend/services/loom_service/weaver/universe_container.py).

---

## 9. Stage 7 — After the write: ledger, causal edge, one "thought"

Placing a shard also updates **living state**:

1. **RAM ledger** — the "awake" set. A dict of shards with `activation`, `phase_angle`,
   `latent_position`, `velocity`, `hits`, plus 7 cognitive fields
   (energy/entropy/stability/resonance/momentum/decay/attention). Capped at
   `RAM_LEDGER_MAX = 50000`; when full, a stronger newcomer evicts the weakest resident
   (the loser stays safely asleep on disk).
2. **Journal** — a 32-byte transaction record for crash recovery.
3. **Sequential causal edge** — "what was ingested right before this" becomes a directed
   edge $\text{prev}\to\text{this}$ in the causal graph (see §11).
4. **One cognitive tick** — the new shard is a *percept*; the shared latent field "feels"
   it (see §10).

**Where.** `ingest_shard` / `ingest_batch`
([`weaver_coordinator.py:774-960`](../backend/services/loom_service/weaver/weaver_coordinator.py#L774-L960)).

---

## 10. The Cognitive Tick — a "thought" as physics 🧠

**(a) What / Why.** This is the brain's continuous inner life. Whenever memories are
touched (ingest or recall), a shared **latent field** evolves like a physical system, and
each touched shard is a particle pulled around by forces. Out of this fall two field
metrics — **coherence** and **entropy** — and an **energy budget** that gates sleep. Runs
on a bounded hot set (`COGNITION_WORKING_SET_MAX = 128`) so it's O(128²), never O(ledger).

**(b) The maths.**

**Step 1 — Input tensor** (superpose whatever was just touched, weighted by intensity):

$$\mathbf{u} = \frac{\sum_{\text{sid}} \text{intensity}_{\text{sid}}\cdot \mathbf{x}_{\text{sid}}}{\big\lVert \sum \dots \big\rVert}$$

**Step 2 — Shared latent field: a second-order mass-spring-damper.** This is a damped
oscillator being *driven* by the input and *pulled* toward working memory, with
competitive inhibition:

$$\mathbf{v}_{\text{field}} \leftarrow \underbrace{\beta\,\mathbf{v}_{\text{field}}}_{\text{drag }0.8}
+ \underbrace{g_{\text{in}}\,\mathbf{u}}_{\text{input }0.4}
+ \underbrace{g_{\text{sync}}(\mathbf{w}-\mathbf{L}_{\text{prev}})}_{\text{sync }0.2}
- \underbrace{\gamma\,\mathbf{w}}_{\text{inhibition }0.15}$$

$$\mathbf{L} \leftarrow \frac{\mathbf{L}_{\text{prev}} + \mathbf{v}_{\text{field}}}{\lVert\cdot\rVert}
\qquad\text{(new latent field)}$$

**Working memory** follows the field via an exponential moving average (a slow, stable
attractor) with $\alpha = 0.1$:

$$\mathbf{w} \leftarrow \frac{(1-\alpha)\,\mathbf{w} + \alpha\,\mathbf{L}}{\lVert\cdot\rVert}$$

**Step 3 — Per-shard forces** (for each hot shard, $dt=0.1$):

$$\mathbf{f}_{\text{latent}} = (\mathbf{L}-\mathbf{x})\cdot 0.25\,A \quad\text{(pull toward the field)}$$
$$\mathbf{f}_{\text{causal}} = \sum_{\text{nbr}} (\mathbf{x}_{\text{nbr}}-\mathbf{x})\cdot 0.15\,w_{\text{edge}} \quad\text{(spring to causal neighbours)}$$
$$\mathbf{f}_{\text{repel}} = \sum_{\lVert\mathbf{x}-\mathbf{x}_o\rVert < r} \frac{\mathbf{x}-\mathbf{x}_o}{\lVert\cdot\rVert}\cdot k \quad\text{(push apart if too close)}$$
$$\mathbf{v} \leftarrow 0.7\,\mathbf{v} + (\mathbf{f}_{\text{latent}}+\mathbf{f}_{\text{causal}}+\mathbf{f}_{\text{repel}})\,dt, \qquad \mathbf{x} \leftarrow \frac{\mathbf{x}+\mathbf{v}\,dt}{\lVert\cdot\rVert}$$

Each shard then updates its own scalars: `resonance` = $\mathbf{x}\cdot\mathbf{L}$ clipped
to $[0,1]$, `momentum` = $\lVert\mathbf{v}\rVert$, `energy` rises (+0.2) if touched else
decays (−0.02), `stability` and `attention` similarly.

**Step 4 — Field metrics.**

*Coherence* = the **Kuramoto order parameter** analogue (how aligned are the active
vectors?). With centroid $\bar{\mathbf{x}}$ (normalised):

$$r = \frac{1}{N}\sum_i \frac{\mathbf{x}_i}{\lVert\mathbf{x}_i\rVert}\cdot\bar{\mathbf{x}} \in [0,1]$$

$r\to1$ = one coherent thought; $r\to0$ = scattered/incoherent.

*Entropy* = field turbulence (how much the field moved this tick):

$$S = \text{clip}\big(\lVert\mathbf{L}-\mathbf{L}_{\text{prev}}\rVert,\ 0, 1\big)$$

*Energy budget* — a thermodynamics: coherence gains recharge it, entropy (churn) drains
it. When it runs low → sleep pressure:

$$\text{budget} \leftarrow \text{clip}\big(\text{budget} + \underbrace{(0.15\,\Delta r + 0.01\,r)}_{\text{recovery}} - \underbrace{0.02\,S}_{\text{load}},\ 0, 1\big)$$

**Step 5 — Assembly crystallisation.** Shards resonating with the field above 0.4, then
filtered by a **coherence gate** (they must also cohere with the group's dominant member,
`ASSEMBLY_COHERENCE_MIN = 0.5`), form a **cognitive assembly** — an emergent "this cluster
of real shards is firing together right now." Successive distinct assemblies leave a
causal edge (§11).

**(c) Where.** `process_cognitive_tick`
([`weaver_coordinator.py:991-1191`](../backend/services/loom_service/weaver/weaver_coordinator.py#L991))
and `CognitiveMetrics`
([`cognitive_field_substrate.py:59-94`](../backend/services/loom_service/cortex/latent_field_cognition/cognitive_field_substrate.py#L59)).

> **Why a coherence gate exists (bug #1):** the shared field is a *blend* of everything
> touched. When two unrelated topics are active together, both resonate with the blend and
> get bundled into one assembly → one topic-mixed semantic crystal (90–92% were mixed).
> The gate forces an assembly to be *one* concept.

---

## 11. The Causal Graph — "A causes B" (active inference)

**(a) What / Why.** A directed graph over real shard ids. Edges come from three sources:
(1) sequential adjacency at ingest, (2) transitions between successive assemblies, (3)
Hebbian co-occurrence at recall ("fire together → wire together"). This graph is what lets
recall do **multi-hop / associative** retrieval that plain cosine cannot.

**(b) The maths.** Temporal adjacency ≠ causation. An edge's strength is discounted by
*prediction error* (pure surprise = temporal noise, not cause):

$$\text{strength} = w\,(1 - \text{prediction\_error})$$

Reject if below `CAUSAL_THRESHOLD = 0.1`. Repeated edges **average in** and count
frequency:

$$w_{ab} \leftarrow \frac{w_{ab} + \text{strength}}{2}, \qquad \text{freq} \mathrel{+}= 1$$

Stale edges (unseen > 60 s) decay by `STALE_DECAY_RATE = 0.1` and are pruned below
`STALE_PRUNE_THRESHOLD = 0.05`; orphaned nodes are removed. So the graph **forgets**
associations that stop recurring.

**(c) Where.** `CausalGraphs`
([`predictive_processing.py:94-153`](../backend/services/loom_service/cortex/latent_field_cognition/predictive_processing.py#L94)).

---

## 12. Stage 8 — RECALL: querying the brain 🔎

A query text becomes an embedding $\mathbf{q}$, then flows through the read pipeline.
There are **two modes**:

- **`learn=True` (learning mode, default):** recall also *remembers* the interaction —
  decay, reinforcement, re-awakening, Kuramoto, a cognitive tick, journaling. This is what
  makes it a *memory* not an *index*, but costs a full write-and-think per query
  (~200 ms measured).
- **`learn=False` (serving mode):** pure read-only ranking (route → score → top-k),
  ~1.4 ms, matching FAISS quality. It still records a cheap **trace** and consolidates
  later in batches, so the brain *still* learns from reads — just deferred (the real
  perception/consolidation split).

### 12.1 Decay first (learning mode) — the forgetting curve

Every awake shard's activation decays exponentially since it was last recalled, but a
memory recalled *often* decays **slower** (log-buffered friction — an ACT-R-like effect):

$$\lambda_{\text{eff}} = \frac{\lambda}{1 + \ln(1 + \text{hits})}, \qquad A(t) = A_0\,e^{-\lambda_{\text{eff}}\,\Delta t}$$

Shards whose activation falls below `SLEEP_EVICTION_LIMIT = 0.001` are **evicted** from RAM
(they sleep on disk, still fully durable). **Where:**
[`memory_fluidity.py`](../backend/services/loom_service/weaver/memory_fluidity.py).

### 12.2 Route + resonance-jump + causal hop (narrow the candidates)

1. **Route** $\mathbf{q}$ to the nearest crystal (same cosine-to-centroid as §6).
2. If the crystal is large ($n >$ `RECALL_FULL_SCAN_BELOW = 20000`): compute the query
   signature, Hamming it against the leaders, open the `RECALL_SEED_LEADERS = 24` nearest
   neighbourhoods. Candidates = shards in those buckets (∪ unbucketed safety net).
3. **Causal hop** (`_causal_hop_indices`): expand candidates by walking the causal graph
   up to `CAUSAL_HOP_DEPTH = 2` from the seed hits — so an answer *causally linked* to a
   match is reachable even if it lives in a different bucket. This is the multi-hop path.
4. Below the threshold, or for legacy crystals → **full scan** (identical results, just
   slower). Superseded shards (from sleep) are masked out here.

### 12.3 Score the candidates (the physics)

Let $\cos_i$ = cosine of candidate $i$ to the query (one matrix-multiply using the
precomputed norms). Then:

**Gravitational attraction** — Newton's law with cosine *distance* $(1-\cos)$ as the
"distance", masses as the bodies:

$$G_i = \frac{m_a\,m_i}{\max\big((1-\cos_i)^2,\ \varepsilon\big)}, \qquad \varepsilon = 10^{-4}$$

As $\cos_i \to 1$ (query ≈ shard), distance → 0, gravity → huge (clamped by $\varepsilon$).
This is monotonic in similarity, so it reproduces cosine ranking on plain retrieval, but
**mass** lets an important heavy memory outrank a slightly-closer trivial one.

**Wave excitation gain** — a phase-resonance kick (memory *dynamics*, drives waking, **not**
ranking):

$$\text{gain}_i = \eta\,\cos_i\,\cos(\theta_q - \theta_i), \qquad \eta = 0.4$$
$$A^{\text{new}}_i = \text{clip}(A_i + \text{gain}_i,\ 0, 1)$$

**Final ranking score** — relevance-first (gravity), with a *monotonic* activation nudge.
**Phase is deliberately excluded from ranking** (it's uncorrelated with relevance;
including it dragged R@1 from 0.50 → 0.43):

$$\boxed{\ \text{score}_i = G_i\,\big(1 + \text{RANK\_ACTIVATION\_WEIGHT}\cdot A_i\big)\ }, \quad \text{RANK\_ACTIVATION\_WEIGHT}=0.25$$

**Optional causal spread** (`CAUSAL_SPREAD_WEIGHT`, off by default) — HippoRAG-style
spreading activation. Each relevant candidate lets its causal neighbours *inherit* a
fraction of its score, **additively** (so a cosine-distant multi-hop answer can rise into
top-k even though its own base score is ~0):

$$\text{score}_j \leftarrow \max\Big(\text{score}_j,\ \text{spread\_w}\cdot \max_{i\to j}\big[\text{score}_i\cdot\min(w_{ij},1)\big]\Big)$$

Then take **top-k** by score.

### 12.4 Plasticity: reinforce, re-awaken, phase-lock (learning mode)

- **Reinforce** the winners: `hits += 1`, activation += `RECALL_BOOST_VALUE = 0.15`.
- **Re-awaken** sleeping shards excited by this query — but only the `RECALL_MAX_WAKE = 256`
  strongest (a strong query could otherwise wake half a crystal).
- **Kuramoto phase-locking** — the oscillator sync, in **closed form** (O(L) not O(L²)).
  The pairwise update $\sum_j a_j \sin(\theta_j-\theta_i)$ expands via the sine
  angle-subtraction identity into two global sums:

$$S_1 = \sum_j a_j\sin\theta_j, \quad S_2 = \sum_j a_j\cos\theta_j$$
$$\Delta\theta_i = \frac{K}{N}\big(\cos\theta_i\,S_1 - \sin\theta_i\,S_2\big), \qquad \theta_i \leftarrow \big((\theta_i + \Delta\theta_i + \pi)\bmod 2\pi\big) - \pi$$

This is mathematically identical to computing every pairwise interaction, but linear in
the number of oscillators. $K =$ `KURAMOTO_COUPLING = 0.1`. Result: co-active memories
gradually **synchronise their phases** — a soft binding signal.

Finally, metadata (text etc.) is decoded **only for the top-k winners**, and the recall
becomes a percept for the next cognitive tick.

**(c) Where.** `recall`
([`weaver_coordinator.py:1382-1745`](../backend/services/loom_service/weaver/weaver_coordinator.py#L1382)).

---

## 13. Stage 9 — SLEEP: consolidation (NREM + REM) 😴

**(a) What / Why.** When the brain has been busy then goes idle, it consolidates — the
one experiment the research says nobody has done at this scale with continuous dynamics.
Concepts that keep **recurring** (assembly frequency ≥ `SLEEP_CONSOLIDATION_FREQUENCY_THRESHOLD = 3`)
are compressed into a single new "semantic crystal."

**(b) The process.**
- Trigger: idle ≥ 120 s **and** ≥ 50 shards since last sleep (or forced).
- Gather the recurring concept's member shards; apply the **coherence gate** again so the
  summary is faithful to *one* concept.
- **NREM-equivalent:** reinforce causal edges + activation among members.
- **REM-equivalent:** synthesise a summary via the LLM (faithfulness-constrained prompt,
  falls back to a deterministic template if the LLM is down), embed it, and **ingest it as
  a new shard** at the members' mean position:

$$\mathbf{v}_{\text{new}} = \frac{\frac{1}{n}\sum_i \mathbf{x}_i}{\lVert\cdot\rVert}, \qquad m_{\text{new}} = \min(5.0,\ 1.5 + 0.1\,n)$$

- **Mark sources superseded** — deprioritised in recall's fast path, but **never
  deleted**; `decode_cluster()` can always see them (provenance preserved). A
  `synthesis_mode` flag ("llm"/"template") is stamped so a faithfulness audit can tell
  them apart.

**(c) Where.** `maybe_sleep_cycle`, `_synthesize_semantic_text`, `_mark_superseded`
([`weaver_coordinator.py:1798-1961`](../backend/services/loom_service/weaver/weaver_coordinator.py#L1798)).

---

## 14. Persistence — one file, crash-safe

All state folds into a single **`universe.loom`** container: a header (seed/dim), named
segments (atlas, snapshot, cognitive-state bins), and a typed append-only journal tail.
Crystal registrations and ledger transactions are journaled immediately (survive a crash);
the heavy ledger snapshot is folded in at checkpoint cadence. Boot replays the journal +
snapshot to rebuild the RAM ledger in ~1 s. Legacy loose-file brains auto-consolidate on
first boot (originals moved to `legacy_backup/`, nothing deleted).

**Where.** `checkpoint`, `_load_persistence_layers`, `_consolidate_universe`
([`weaver_coordinator.py`](../backend/services/loom_service/weaver/weaver_coordinator.py)) +
[`universe_container.py`](../backend/services/loom_service/weaver/universe_container.py).

---

## 15. End-to-end worked example (numbers)

Ingest the sentence **"the dog ran home"** (assume $d=384$, and after embedding we get
some $\mathbf{t}$ with $\sum|t_i| = 30$, and 4 words):

1. **Embed:** $\mathbf{t} = f_\phi(\text{"the dog ran home"}) \in \mathbb{R}^{384}$.
2. **Mass:** $m = \ln(4+1)\cdot 30 = 1.609\times 30 \approx 48.3$. (Heavy → will snap to
   its true meaning.)
3. **Phase:** $\theta = \operatorname{atan2}(\sum \text{first }192,\ \sum \text{last }192)$
   → some angle in $[0,2\pi)$, say $2.1$ rad.
4. **Scaffold:** $\mathbf{s}$ = seed-derived $\mathcal N(0,0.1)^{384}$ (tiny, near origin).
5. **Warp:** $w = 48.3/(1 + 48.3) = 0.98$. So $\mathbf{p} = \mathbf{s} + 0.98(\mathbf{t}-\mathbf{s}) \approx \mathbf{t}$ → normalised. (Heavy memory ≈ its meaning.)
6. **Route:** cosine $\mathbf{p}$ vs each crystal centroid → say `crystal_1.loom` wins.
7. **Bucket:** $\text{sign}(\mathbf{t}_{[:128]})$ → 128-bit sig; nearest leader Hamming = 20
   ≤ 32 → assimilate into that neighbourhood.
8. **Append** to `crystal_1.loom`; ledger gets `{activation:1.0, phase:2.1, hits:0, ...}`;
   causal edge `prev_shard → this`; one cognitive tick.

Now query **"where did the animal go?"** → $\mathbf{q}$:

1. Route → `crystal_1.loom`. Small crystal → full scan.
2. Suppose $\cos(\mathbf{q},\mathbf{t}) = 0.55$. Gravity: $G = (1\cdot 48.3)/\max((1-0.55)^2, 10^{-4}) = 48.3/0.2025 \approx 238.5$.
3. Score $= 238.5\cdot(1 + 0.25\cdot 1.0) = 298.1$. High → lands in top-k.
4. If "the dog ran home" and "the dog ate food" were ingested adjacently, the causal hop
   can also pull in "ate food" even though it's less cosine-similar — that's the
   associative win.

---

## 16. The knobs (what to turn to change behaviour)

| Knob | Effect if you raise it |
|---|---|
| `SCAFFOLD_WARP` | shards stay **closer to their random socket** (less meaning-driven placement) |
| `MEMORY_DECAY_RATE` | brain **forgets faster** |
| `WAVE_EXCITATION_GAIN` | queries **wake** more/stronger memories |
| `KURAMOTO_COUPLING` | oscillators **sync harder** (more phase binding) |
| `RANK_ACTIVATION_WEIGHT` | recently-used memories rank **higher** |
| `CAUSAL_SPREAD_WEIGHT` | more **multi-hop/associative** pull (raise for narrative corpora) |
| `RECALL_SEED_LEADERS` | recall opens **more neighbourhoods** → better recall, slower |
| `RECALL_FULL_SCAN_BELOW` | crystals stay on **exact full scan** longer |
| `ASSEMBLY_COHERENCE_MIN` | assemblies must be **purer** single concepts |

All in [`backend/.modelTunning`](../backend/.modelTunning), hot-reloaded via `tuning_manager`.

---

## 17. File map (where each idea lives)

| Idea | File |
|---|---|
| Text → embedding (4-tier, circuit-broken) | `LLM_service/embedding/embedding_service.py` |
| HDC: basis, project, bind, bundle, fuse, pack | `cortex/HyperVectorCreation.py` |
| Seed scaffold (procedural coordinates) | `weaver/seed_core.py` |
| Momentum, gravity, excitation, Kuramoto (math kernels) | `weaver/storage_physics.py` |
| Decay / forgetting curve | `weaver/memory_fluidity.py` |
| Routing + centroids + division | `weaver/atlas_router.py` |
| `.loom` / `.idx` / `universe.loom` storage | `weaver/substrate_layout.py`, `weaver/universe_container.py` |
| **The orchestrator**: ingest, recall, sleep, cognition, persistence | `weaver/weaver_coordinator.py` |
| Causal graph (active inference) | `cortex/latent_field_cognition/predictive_processing.py` |
| Cognitive field metrics + reference tick | `cortex/latent_field_cognition/cognitive_field_substrate.py` |
| Assemblies / semantic field | `cortex/latent_field_cognition/attractor_basin_compilation.py` |
| Document-weave path (builds HDC graph) | `cortex/SubstrateWeaver.py` |
| Read/inspect the brain | `decoder/`, `weaver/decode/` |

---

## 18. Open questions for us (leave comments here 💬)

1. **HDC's role.** Should bind/bundle become load-bearing (structural queries), or stay a
   compact stored signature? (§3.7)
2. **Two dimensions.** The live path uses continuous $d$-dim; HDC is 8000-dim. Do we ever
   want recall to score on HDC similarity too, or keep it purely continuous?
3. **Phase / Kuramoto.** It's excluded from ranking (correctly), and the benchmark says
   oscillator terms don't pay rent on plain retrieval. Keep it only as a binding/dynamics
   signal, or design an experiment where it's causally load-bearing? (see revolution notes)
4. **Causal spread default.** Off (0.0) because sequential-adjacency edges are noise on
   shuffled loads. Do we auto-detect "narrative order" corpora and turn it on?
5. **Sleep faithfulness.** The `synthesis_mode` flag is in place — do we want the
   entailment-based faithfulness audit wired into the sleep loop itself?
```
