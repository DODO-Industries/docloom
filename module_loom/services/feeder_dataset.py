"""
DocLoom Feeder Dataset Topics — Sample text corpus for bulk seeding across distinct semantic domains.
"""
from typing import Dict, List, Tuple

_FEEDER_TOPICS: Dict[str, List[str]] = {
    "marine_biology": [
        "Coral reefs are built over centuries by colonies of tiny calcium-carbonate-secreting polyps.",
        "Bioluminescent plankton produce flashes of light through a luciferin-luciferase chemical reaction.",
        "The deep ocean's midnight zone receives no sunlight and hosts pressure-adapted, often blind species.",
        "Whale falls create temporary deep-sea ecosystems that can sustain scavenger life for decades.",
        "Octopuses have three hearts and blue, copper-based blood instead of iron-based hemoglobin.",
        "Kelp forests along cold coastlines provide shelter and food for sea otters, fish, and urchins.",
        "Hydrothermal vent communities derive energy from chemosynthesis rather than sunlight.",
        "Sea turtles navigate thousands of miles using Earth's magnetic field as an internal compass.",
        "Mangrove root systems stabilize coastlines and serve as nurseries for juvenile fish.",
        "The ocean's twilight zone hosts the largest daily animal migration on the planet.",
        "Coral bleaching occurs when rising water temperatures cause polyps to expel their symbiotic algae.",
        "Sharks have electroreceptor organs called ampullae of Lorenzini that detect faint electric fields.",
        "Tidal pools create harsh, fluctuating micro-environments that shape highly adaptable species.",
        "Plankton blooms can be tracked from space by the color they impart to surface waters.",
        "Deep-sea anglerfish use a bioluminescent lure to attract prey in total darkness.",
        "Ocean currents distribute heat globally and strongly influence regional climate patterns.",
        "Sea sponges lack true tissues or organs yet filter enormous volumes of water for nutrients.",
        "Barnacles permanently cement themselves to a surface after a brief free-swimming larval stage.",
        "Squid can change skin color and texture almost instantly using chromatophores and papillae.",
        "Coastal upwelling brings nutrient-rich deep water to the surface, fueling productive fisheries.",
    ],
    "renaissance_art": [
        "Linear perspective, formalized by Brunelleschi, gave Renaissance painting a convincing sense of depth.",
        "Leonardo da Vinci's sfumato technique blends tones so gradually that outlines seem to dissolve.",
        "Michelangelo carved the Pietà from a single block of Carrara marble in his early twenties.",
        "Fresco painting required artists to work quickly onto wet plaster before it dried.",
        "Botticelli's Primavera weaves classical mythology into an allegory of springtime abundance.",
        "The Medici family's patronage financed much of Florence's early Renaissance artistic output.",
        "Raphael's School of Athens places ancient philosophers within a grand classical architecture.",
        "Oil paint, refined in the Renaissance, allowed slower work and richer, layered color.",
        "Donatello's bronze David was the first freestanding nude sculpture since antiquity.",
        "Renaissance workshops trained apprentices through years of grinding pigments before ever painting.",
        "The Sistine Chapel ceiling took Michelangelo roughly four years to complete while lying on scaffolding.",
        "Vasari's Lives of the Artists is one of the earliest works of art history and biography.",
        "Venetian painters like Titian became known for vivid color achieved through oil glazing.",
        "Humanist scholarship encouraged Renaissance artists to study anatomy and classical proportion.",
        "Chiaroscuro uses strong contrasts between light and dark to model three-dimensional form.",
        "Patrons commissioned portraits partly to display wealth, status, and dynastic continuity.",
        "Brunelleschi's dome for Florence Cathedral solved an engineering problem thought unsolvable.",
        "Engraving and woodcut prints spread Renaissance ideas across Europe far faster than paintings.",
        "Northern Renaissance painters like Jan van Eyck brought astonishing optical realism to oil.",
        "The term Renaissance, meaning rebirth, reflected the rediscovery of classical Greek and Roman texts.",
    ],
    "astrophysics": [
        "A star's mass at birth almost entirely dictates how it will evolve and how it will die.",
        "Gravitational waves are ripples in spacetime caused by accelerating massive objects like merging black holes.",
        "Neutron stars pack roughly the mass of our sun into a sphere only about twenty kilometers across.",
        "The cosmic microwave background radiation is the oldest observable electromagnetic light in the universe.",
        "Dark matter interacts through gravity but does not emit, absorb, or reflect any electromagnetic radiation.",
        "Dark energy is the hypothetical form of energy proposed to explain the universe's accelerating expansion.",
        "Supernovae synthesize elements heavier than iron and disperse them across the interstellar medium.",
        "The event horizon of a black hole is the boundary beyond which nothing, not even light, can escape.",
        "Spectroscopy allows astronomers to identify a distant star's chemical composition and temperature.",
        "Pulsars are rapidly rotating magnetized neutron stars that emit beams of electromagnetic radiation.",
        "The interstellar medium is mostly hydrogen and helium gas mixed with sparse microscopic dust grains.",
        "A galaxy's rotation curve gave some of the earliest strong evidence for the existence of dark matter.",
        "Stellar nucleosynthesis inside burning stars creates the carbon, oxygen, and nitrogen essential for life.",
        "Hubble's law states that galaxies appear to recede from us at speeds proportional to their distance.",
        "Tidal forces around a supermassive black hole can tear an approaching star apart in a disruption event.",
        "A planetary nebula is the glowing shell of ionized gas expelled by a dying low-mass star.",
        "Gamma-ray bursts are the most energetic electromagnetic events known to occur in the universe.",
        "Cosmic rays are high-energy protons and atomic nuclei moving through space at nearly the speed of light.",
        "The solar wind is a continuous stream of charged particles released from the sun's upper atmosphere.",
        "Redshift occurs when light emitted from an object moving away from an observer stretches to longer wavelengths.",
    ],
    "culinary_spices": [
        "Saffron is harvested by hand from the delicate stigmas of the Crocus sativus flower.",
        "Black pepper's heat comes from the chemical compound piperine, distinct from chili's capsaicin.",
        "True cinnamon, derived from the inner bark of Cinnamomum verum, has a softer, sweeter flavor than cassia.",
        "Cardamom pods belong to the ginger family and are native to the tropical forests of southern India.",
        "Star anise contains anethole, the same aromatic compound that gives fennel and licorice their distinctive flavor.",
        "Nutmeg and mace come from the same tree: nutmeg is the seed, while mace is the lacy red covering around it.",
        "Turmeric's bright golden-yellow color and earthy flavor come from the polyphenol curcumin.",
        "Clove buds are harvested while still immature and dried until they resemble small brown nails.",
        "Cumin seeds are an essential base note in cuisines stretching from Mexico to the Indian subcontinent.",
        "Vanilla beans are the cured seed pods of a tropical climbing orchid native to Mesoamerica.",
        "Paprika ranges from sweet to fiery depending on the specific Capsicum varieties dried and ground.",
        "Coriander seeds have a citrusy, floral flavor very different from the fresh leaves of the same plant.",
        "Chili heat is measured in Scoville units, based on the concentration of capsaicin present.",
        "Fenugreek seeds taste bitter raw but mellow considerably once toasted or slow-cooked.",
        "Za'atar is a Middle Eastern spice blend typically combining thyme, sumac, and toasted sesame.",
        "Sumac's tart, lemony flavor comes from drying and grinding the berries of the sumac shrub.",
        "Five-spice powder balances sweet, sour, bitter, salty, and umami in one blend.",
        "Whole spices generally stay potent far longer than pre-ground versions of the same spice.",
        "Curry leaves, distinct from curry powder, are used fresh or fried to release their aroma.",
        "Asafoetida is used in tiny amounts, mellowing into a savory, onion-like flavor once cooked.",
    ],
    "mountain_geology": [
        "The Himalayas continue to rise as the Indian and Eurasian tectonic plates keep colliding.",
        "Fold mountains form when compressive tectonic forces buckle layers of rock upward.",
        "Glacial erosion carves distinctive U-shaped valleys, unlike the V-shaped valleys rivers cut.",
        "Volcanic mountains like Mount Fuji build up over time from repeated layers of lava and ash.",
        "Cirques are bowl-shaped hollows carved into mountainsides by the head of a glacier.",
        "The tree line on a mountain marks the elevation above which temperatures are too low for tree growth.",
        "Isostatic rebound causes land to slowly rise after the weight of ancient ice sheets melts away.",
        "Scree slopes form from freeze-thaw weathering that fractures exposed rock over time.",
        "Fault-block mountains, like the Sierra Nevada, form when large crustal blocks tilt along faults.",
        "Alpine environments experience dramatic temperature swings between day and night at high altitude.",
        "Moraines are ridges of rock and sediment deposited at the edges of a moving glacier.",
        "Orogeny is the geological term for the process of mountain formation through crustal deformation.",
        "Permafrost beneath some high-altitude soils remains frozen year-round despite surface thaw in summer.",
        "Hanging valleys form where a smaller glacier's valley floor sits higher than the main valley it joins.",
        "Mountain rain shadows create sharply drier climates on the leeward side of a range.",
        "Metamorphic rock like schist and gneiss often forms deep within mountain-building collision zones.",
        "Avalanches are most likely when a weak snow layer lies buried beneath heavier recent snowfall.",
        "Karst topography can form in mountainous limestone regions, riddled with caves and sinkholes.",
        "Seismic activity along mountain fault lines is a direct signature of ongoing tectonic uplift.",
        "Alpine glaciers are retreating at accelerating rates as global average temperatures rise.",
    ],
}


def feeder_texts(total: int) -> List[Tuple[str, str]]:
    """Cycles through all 5 topic domains, padding each domain's real base
    sentences with light numbered variants once exhausted, to reach `total`
    (topic, text) pairs."""
    topics = list(_FEEDER_TOPICS.keys())
    per_topic = max(1, total // len(topics))
    out: List[Tuple[str, str]] = []
    for topic in topics:
        base = _FEEDER_TOPICS[topic]
        for i in range(per_topic):
            sentence = base[i % len(base)]
            if i >= len(base):
                sentence = f"{sentence} (further notes, batch {i // len(base)})"
            out.append((topic, sentence))
    return out[:total]
