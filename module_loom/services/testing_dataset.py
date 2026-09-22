"""
DocLoom 5-Topic Synthetic Dataset Generator.
Generates 500 distinct, semantically rich sentences across 5 diverse domains:
1. Quantum Physics & Cosmology
2. Genetics & Molecular Biology
3. Computer Architecture & AI
4. World History & Archaeology
5. Global Economics & Finance
"""

TOPICS = {
    "physics": {
        "label": "Quantum & Astrophysics",
        "color": "#38bdf8",  # Cyan
        "templates": [
            "Quantum {concept} demonstrates {phenomenon} in {system}.",
            "The {phenomenon} observed near {cosmic} reveals {concept} principles.",
            "In theoretical astrophysics, {cosmic} exerts {property} governed by {concept}.",
            "{concept} dictates that {system} undergoes {phenomenon} under extreme conditions.",
            "Measurement of {property} in {system} confirms {concept} predictions."
        ],
        "concepts": [
            "superposition", "entanglement", "wave-particle duality", "uncertainty principle",
            "spacetime curvature", "Hawking radiation", "Higgs mechanism", "quantum tunneling",
            "general relativity", "dark energy expansion", "string theory vibration",
            "cosmic microwave background", "gravitational wave distortion", "Planck-scale foam",
            "electroweak symmetry breaking", "black hole thermodynamics", "neutrino oscillation",
            "Casimir effect", "quantum decoherence", "Bose-Einstein condensation"
        ],
        "phenomena": [
            "spontaneous state collapse", "non-local correlation", "asymptotic freedom",
            "gravitational lensing", "vacuum polarization", "anomalous magnetic moment",
            "time dilation", "frame dragging", "spectral redshift", "pair production"
        ],
        "systems": [
            "isolated trapped ions", "superconducting qubits", "degenerate neutron stars",
            "particle accelerator collisions", "interstellar plasma clouds", "event horizon boundaries",
            "optical lattice cavities", "cryogenic diluters", "cosmological horizons"
        ],
        "cosmics": [
            "supermassive black holes", "collapsing magnetars", "quasar accretion disks",
            "binary pulsar systems", "early inflationary universe", "galactic dark matter halos"
        ],
        "properties": [
            "intense tidal forces", "sub-atomic mass resonance", "zero-point energy fluctuations",
            "non-abelian gauge invariance", "negative curvature topology"
        ]
    },
    "biology": {
        "label": "Genetics & Biology",
        "color": "#10b981",  # Emerald
        "templates": [
            "{process} inside {organelle} regulates {pathway} during cellular activity.",
            "The {macromolecule} facilitates {process} to maintain {outcome}.",
            "In molecular genetics, {mutation} alters {macromolecule} expression leading to {outcome}.",
            "{organelle} utilizes {macromolecule} for {process} across membranes.",
            "Epigenetic regulation of {pathway} controls {outcome} via {process}."
        ],
        "processes": [
            "CRISPR-Cas9 gene editing", "DNA transcription elongation", "ribosomal translation",
            "mitochondrial ATP phosphorylation", "homologous recombination repair",
            "telomere cap replication", "post-translational glycosylation", "ubiquitin-mediated proteolysis",
            "synaptic vesicle endocytosis", "monoclonal antibody affinity maturation",
            "RNA interference silencing", "stem cell lineage differentiation", "chemotactic signal transduction",
            "histone acetylation remodeling", "membrane depolarization transport"
        ],
        "organelles": [
            "mitochondrial cristae", "endoplasmic reticulum lumen", "eukaryotic cell nuclei",
            "synaptic neural clefts", "Golgi apparatus cisternae", "lysosomal hydrolytic chambers",
            "peroxisomal membranes", "chloroplast thylakoid stacks"
        ],
        "pathways": [
            "mTOR metabolic signaling", "JAK-STAT cytokine pathway", "p53 tumor suppression",
            "Wnt developmental cascades", "notch intercellular communication", "MAPK cell proliferation"
        ],
        "macromolecules": [
            "messenger RNA polymerase", "DNA helicase enzymes", "myosin molecular motors",
            "transmembrane G-protein receptors", "immunoglobulin gamma antibodies", "chaperone heat-shock proteins"
        ],
        "mutations": [
            "single nucleotide polymorphism", "frameshift insertion", "chromosomal translocation",
            "promoter CpG hypermethylation", "alternative splicing deviation"
        ],
        "outcomes": [
            "homeostatic cellular balance", "selective tissue immunity", "accelerated DNA repair",
            "programmed apoptotic clearance", "synaptic plasticity strengthening", "metabolic energy efficiency"
        ]
    },
    "ai_systems": {
        "label": "Computer Architecture & AI",
        "color": "#818cf8",  # Indigo/Violet
        "templates": [
            "{ai_arch} optimizes {task} by exploiting {technique}.",
            "In modern hardware architectures, {hardware} enhances {task} using {technique}.",
            "Distributed {system} coordinates {technique} to guarantee {guarantee}.",
            "{technique} in {ai_arch} reduces {bottleneck} while accelerating {task}.",
            "Compiler {technique} generates optimized machine instructions for {hardware} executing {task}."
        ],
        "ai_arch": [
            "Transformer self-attention models", "Deep convolutional networks", "Recurrent neural state models",
            "Mixture of Experts routing", "Variational autoencoders", "Graph neural propagation networks",
            "Diffusion generative models", "Retrieval-augmented generation pipelines",
            "State space selective networks", "Binarized low-precision neural models"
        ],
        "tasks": [
            "high-dimensional vector similarity recall", "dense token generation inference",
            "gradient backpropagation convergence", "context KV cache compression",
            "sparse matrix multiplication", "real-time object segmentation", "cross-attention scoring"
        ],
        "techniques": [
            "locality-sensitive hashing", "quantized weight factorization", "speculative decoding",
            "out-of-order pipelined execution", "asynchronous non-blocking I/O", "ring-allreduce distributed synchronization",
            "zero-copy memory mapping", "kernel fusion optimizations", "SIMD vectorization", "SIMD branch prediction"
        ],
        "hardware": [
            "high-bandwidth memory (HBM3) stacks", "tensor processing units", "vector systolic arrays",
            "neuromorphic memristor crossbars", "RISC-V multi-core dies", "optically connected switch fabrics"
        ],
        "systems": [
            "Raft consensus clusters", "decentralized p2p ledgers", "mmap-backed binary substrates",
            "distributed key-value caches", "fault-tolerant persistent journals"
        ],
        "guarantees": [
            "linearizable ACID consistency", "deterministic microsecond latency", "high-availability Byzantine resilience"
        ],
        "bottlenecks": [
            "memory bandwidth saturation", "PCIe bus transfer latency", "instruction cache cache misses"
        ]
    },
    "history": {
        "label": "World History & Archaeology",
        "color": "#f59e0b",  # Amber
        "templates": [
            "The {event} during the {era} fundamentally altered {institution}.",
            "Archaeological discoveries at {site} reveal ancient {artifact} utilized for {purpose}.",
            "In {civilization}, the rise of {institution} was catalyzed by {event}.",
            "The trade of {commodity} across {route} connected {civilization} with distant cultures.",
            "Deciphering {artifact} uncovered how {civilization} governed {institution} during {era}."
        ],
        "events": [
            "Late Bronze Age collapse", "Fall of the Western Roman Empire", "signing of the Magna Carta",
            "invention of movable type printing", "construction of the Great Pyramid of Giza",
            "Peloponnesian war between Sparta and Athens", "Mongol expansion across Eurasia",
            "storming of the Bastille fortress", "signing of the Peace of Westphalia",
            "coronation of Charlemagne", "voyage of the Silk Road caravans", "Alexandrian conquest of Persia"
        ],
        "eras": [
            "classical Hellenistic period", "early Renaissance era", "Mesopotamian Uruk epoch",
            "feudal medieval century", "Age of Enlightenment", "Song Dynasty agricultural boom"
        ],
        "sites": [
            "ancient Pompeii ruins", "the Indus Valley Mohenjo-daro", "the Knossos Minoan palace",
            "the underground Terracotta Army tombs", "the Rosetta archaeological dig", "Göbekli Tepe stone circles"
        ],
        "artifacts": [
            "cuneiform clay legal tablets", "bronze ritual ceremonial vessels", "papyrus administrative scrolls",
            "Byzantine gold solidus coinage", "ancient astrolabe navigation instruments"
        ],
        "civilizations": [
            "ancient Egyptian dynasties", "the Han Empire", "the Roman Republic", "the Achaemenid Empire",
            "the Mayan city-states", "the Abbasid Caliphate", "the Venetian maritime republic"
        ],
        "institutions": [
            "centralized imperial bureaucracy", "maritime commercial law", "codified civil judiciary",
            "agrarian land taxation systems", "feudal knight chivalry charters"
        ],
        "commodities": [
            "raw Baltic amber", "fine mulberry silk", "valuable Indian spices", "Nubian gold bullion"
        ],
        "routes": [
            "the overland Trans-Saharan trails", "the maritime Spice Route", "the Silk Road network", "the Mediterranean sea lanes"
        ],
        "purposes": [
            "maritime celestial navigation", "agricultural harvest accounting", "astronomical equinox observation"
        ]
    },
    "economics": {
        "label": "Economics & Financial Markets",
        "color": "#f43f5e",  # Rose
        "templates": [
            "{instrument} pricing models incorporate {factor} to manage {risk}.",
            "Central bank adjustments of {policy} influence {market} through {mechanism}.",
            "In macroeconomics, persistent {phenomenon} disrupts {market} despite {policy}.",
            "Market makers adjust {mechanism} on {market} when {factor} increases volatility.",
            "The interaction between {policy} and {factor} drives investment in {instrument}."
        ],
        "instruments": [
            "Sovereign treasury bonds", "over-the-counter interest rate swaps", "credit default swap derivatives",
            "fractional reserve bank deposits", "venture capital preferred equity", "collateralized debt obligations",
            "foreign exchange currency futures", "inflation-indexed securities"
        ],
        "factors": [
            "purchasing power parity divergence", "unanticipated inflation shocks", "yield curve inversions",
            "liquidity coverage ratio shifts", "counterparty credit exposure", "geopolitical supply disruption",
            "aggregate demand elasticity", "marginal propensity to consume"
        ],
        "risks": [
            "systemic bank solvency default", "foreign exchange sovereign risk", "tail-risk market downturns",
            "liquidity dry-up events", "interest rate duration mismatch"
        ],
        "policies": [
            "overnight benchmark interest rates", "quantitative tightening asset sales", "statutory reserve requirements",
            "counter-cyclical capital buffers", "forward guidance communication"
        ],
        "markets": [
            "interbank repo lending markets", "secondary municipal bond exchanges", "global foreign exchange pairs",
            "real estate mortgage securitization", "emerging market debt markets"
        ],
        "mechanisms": [
            "bid-ask spread widening", "order book depth dynamics", "automated algorithmic clearing",
            "collateral margin haircut revaluation", "transmission through bank credit lending"
        ],
        "phenomena": [
            "stagflationary wage-price spirals", "liquidity trap equilibrium", "asset bubble speculation",
            "bank run contagion cascades"
        ]
    }
}


def _singular(key: str) -> str:
    if key == "phenomena":
        return "phenomenon"
    if key == "properties":
        return "property"
    if key == "policies":
        return "policy"
    if key == "commodities":
        return "commodity"
    if key == "processes":
        return "process"
    if key.endswith("s"):
        return key[:-1]
    return key


def generate_5_topic_dataset(sentences_per_topic: int = 100):
    """
    Generates deterministic sentences across 5 distinct domains.
    Returns list of dicts with text, topic, and color.
    """
    results = []
    seen = set()
    
    for topic_key, topic_data in TOPICS.items():
        templates = topic_data["templates"]
        color = topic_data["color"]
        label = topic_data["label"]
        
        count = 0
        idx = 0
        max_attempts = sentences_per_topic * 15
        attempts = 0
        while count < sentences_per_topic and attempts < max_attempts:
            attempts += 1
            template = templates[idx % len(templates)]
            idx += 1
            
            # Deterministic fill using modulo offsets
            replacements = {}
            for key, val_list in topic_data.items():
                if isinstance(val_list, list) and key != "templates":
                    stride = 7 if key == "concepts" else 13
                    chosen_val = val_list[(count * stride + idx * 3) % len(val_list)]
                    replacements[key] = chosen_val
                    replacements[_singular(key)] = chosen_val
            
            try:
                sentence = template.format(**replacements)
                if sentence in seen:
                    v = 1
                    candidate = f"{sentence} [ref-{v}]"
                    while candidate in seen:
                        v += 1
                        candidate = f"{sentence} [ref-{v}]"
                    sentence = candidate
                seen.add(sentence)
                results.append({
                    "text": sentence,
                    "topic": topic_key,
                    "topic_label": label,
                    "color": color
                })
                count += 1
            except (KeyError, IndexError, ValueError):
                continue
            
    return results
