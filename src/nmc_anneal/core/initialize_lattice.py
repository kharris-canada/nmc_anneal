import numpy as np
from nmc_anneal import SimulationConfig

# Species labels
LI = "Li"
MN = "Mn"
NI2 = "Ni2+"
NI3 = "Ni3+"
CO = "Co3+"
VACANCY = "Vac"

# Species charges
SPECIES_CHARGE = {
    LI: 1,
    NI2: 2,
    NI3: 3,
    CO: 3,
    MN: 4,
    VACANCY: 0,
}


def initialize_lattice(
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate initial species and charge lattices using random positioning for stoichometry in the config data class

    Args:
        config (SimulationConfig): Dict style data class containing most simulation parameters. See parser.py

    Returns
    -------
    lattice_species : np.ndarray
        Array of species labels with shape (2*n_layers, width, width)
    lattice_charges : np.ndarray
        Array of integer charges with same shape

    Convention is that even layers (0,2,4, etc.) are Li
                  while odd layers are TM layers
                  oxygen layers not explicity stored as unchanging
    """

    if config.random_seed is not None:
        rng = np.random.default_rng(config.random_seed)
    else:
        rng = np.random.default_rng()

    n_total_layers = 2 * config.n_layers
    shape = (n_total_layers, config.width, config.width)

    # U4 stores 4-entry-string for atom name + charge
    lattice_species = np.full(shape, VACANCY, dtype="<U4")
    lattice_charges = np.zeros(shape, dtype=np.int8)

    # layers 0, 2, etc. are lithium
    # layerss 1,3, etc. are transition metals
    for layer_idx in range(n_total_layers):
        if layer_idx % 2 == 0:
            species_layer = _populate_li_layer(config, rng)
        else:
            species_layer = _populate_tm_layer(config, rng)

        lattice_species[layer_idx] = species_layer
        lattice_charges[layer_idx] = _charges_from_species(species_layer)

    # generate an easy to read empirical formula and store in config data class
    config.stoich_string = _stoich_to_namestring(config)

    return lattice_species, lattice_charges


def _populate_li_layer(
    config: SimulationConfig, rng: np.random.Generator
) -> np.ndarray:
    """

    Args:
        config (SimulationConfig): Dict style data class containing most simulation parameters. See parser.py
        rng (np.random.Generator): random number generator that gets passed on to next layer of initialization process

    Returns:
        np.ndarray: Array with one lithium layer at the specified stoichiometry and random atomic positions
    """
    width = config.width
    n_sites = width * width

    # NOTE!!! _counts_from_fractions assumes li is 1st entry in fractions and species
    fractions = [
        config.li_fraction_li_layer,
        config.mn_fraction_li_layer,
        config.ni2_fraction_li_layer,
        config.ni3_fraction_li_layer,
        config.vac_fraction_li_layer,
    ]

    # NOTE!!! order of species must match order of fractions
    species = [LI, MN, NI2, NI3, VACANCY]

    layer = "li"
    counts = _counts_from_fractions(n_sites, fractions, layer)
    flat = _build_layer_array(n_sites, species, counts, rng)

    return flat.reshape((width, width))


def _populate_tm_layer(
    config: SimulationConfig, rng: np.random.Generator
) -> np.ndarray:
    """_summary_

    Args:
        config (SimulationConfig): Dict style data class containing most simulation parameters. See parser.py
        rng (np.random.Generator): random number generator to use when picking positions

    Returns:
        np.ndarray:Array with one transition metal layer at the specified stoichiometry and random atomic positions
    """
    width = config.width
    n_sites = width * width

    # NOTE!!! _counts_from_fractions assumes Co is 5th entry in fractions and species
    fractions = [
        config.li_fraction_tm_layer,
        config.mn_fraction_tm_layer,
        config.ni2_fraction_tm_layer,
        config.ni3_fraction_tm_layer,
        config.co_fraction_tm_layer,
        config.vac_fraction_tm_layer,
    ]

    # NOTE!!! order of species must match order of fractions
    species = [LI, MN, NI2, NI3, CO, VACANCY]

    layer = "TM"
    counts = _counts_from_fractions(n_sites, fractions, layer)
    flat = _build_layer_array(n_sites, species, counts, rng)

    return flat.reshape((width, width))


def _counts_from_fractions(
    n_sites: int,
    fractions: list[float],
    layer: str,
) -> list[int]:
    """
    Convert fractions into integer site counts.

    Args:
        n_sites (int): Total atomic sites required
        fractions (list[float]): list of fraction of total atomic number of each species
        layer (str): transition metal or lithium layer

    Returns:
        list[int]: integer list of number of each atom type (uses simple list order rather than labels to keep track of species)
    """
    counts = [int(round(f * n_sites)) for f in fractions]

    # Adjust rounding error by adding a small number of Li or Co atoms
    # just to fill last few holes
    total_assigned = sum(counts)
    if total_assigned < n_sites:
        if layer == "li":
            counts[0] += (
                n_sites - total_assigned
            )  # correct rounding errors by adding li
        if layer == "TM":
            counts[4] += (
                n_sites - total_assigned
            )  # correct rounding errors by adding Co3+

    # quick double check that all sites assigned:
    total_assigned = sum(counts)
    if total_assigned < n_sites:
        raise ValueError(
            "Initialize_lattice.py tried to correct rounding errors with Li/Co but failed"
        )

    return counts


def _build_layer_array(
    n_sites: int,
    species: list[str],
    counts: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate random positions for all the species in they array called layer.
    NOTE: doesn't have the correct 2D shape, it gets reformatted to a square later

    Args:
        n_sites (int): Total atomic sites required
        species (list[str]): Ordered list of species atomic *names* (not charges)
        counts (list[int]): Ordered list of number of atoms of each species in the species list (!order matters!)
        rng (np.random.Generator): random number generator to use when picking positions

    Returns:
        np.ndarray: _description_
    """
    layer = np.full(n_sites, VACANCY, dtype="<U4")

    start = 0
    for specie, count in zip(species, counts):
        layer[start : start + count] = specie
        start += count

    rng.shuffle(layer)
    return layer


def _charges_from_species(species_layer: np.ndarray) -> np.ndarray:
    """
    Map species labels to default charges.

    Args:
        species_layer (np.ndarray): The already randomly initialized array containing one layer with the atomic species names

    Returns:
        np.ndarray: 2D layer with atomic charges matching the input species names
    """
    charges = np.zeros(species_layer.shape, dtype=np.int8)

    for specie, charge in SPECIES_CHARGE.items():
        if specie == VACANCY:
            continue
        charges[species_layer == specie] = charge

    return charges


def _stoich_to_namestring(config: SimulationConfig) -> str:
    """
    Inspects the stoichiometry described in the config data class object and converts it to an easily readable empirical formula

    Args:
        config (SimulationConfig): Dict style data class containing most simulation parameters. See parser.py

    Returns:
        str: empirical formula in standard format
    """
    parts = []

    def fmt(x):
        return f"{x:.2g}"

    if config.li_fraction_li_layer != 0:
        parts.append(f"$Li_{{{fmt(config.li_fraction_li_layer)}}}$")

    if config.mn_fraction_li_layer != 0:
        parts.append(f"$Mn_{{{fmt(config.mn_fraction_li_layer)}}}$")

    if config.ni2_fraction_li_layer != 0:
        parts.append(f"$Ni^{{2+}}_{{{fmt(config.ni2_fraction_li_layer)}}}$")

    if config.ni3_fraction_li_layer != 0:
        parts.append(f"$Ni^{{3+}}_{{{fmt(config.ni3_fraction_li_layer)}}}$")

    if config.vac_fraction_li_layer != 0:
        parts.append(f"$Vac_{{{fmt(config.vac_fraction_li_layer)}}}$")

    # ---- TM layer (square brackets) ----
    tm_parts = []

    if config.li_fraction_tm_layer != 0:
        tm_parts.append(f"$Li_{{{fmt(config.li_fraction_tm_layer)}}}$")

    if config.mn_fraction_tm_layer != 0:
        tm_parts.append(f"$Mn_{{{fmt(config.mn_fraction_tm_layer)}}}$")

    if config.ni2_fraction_tm_layer != 0:
        tm_parts.append(f"$Ni^{{2+}}_{{{fmt(config.ni2_fraction_tm_layer)}}}$")

    if config.ni3_fraction_tm_layer != 0:
        tm_parts.append(f"$Ni^{{3+}}_{{{fmt(config.ni3_fraction_tm_layer)}}}$")

    if config.co_fraction_tm_layer != 0:
        tm_parts.append(f"$Co_{{{fmt(config.co_fraction_tm_layer)}}}$")

    if config.vac_fraction_tm_layer != 0:
        tm_parts.append(f"$Vac_{{{fmt(config.vac_fraction_tm_layer)}}}$")

    tm_parts.append(r"$O_{2}$")

    if tm_parts:
        parts.append("[" + "".join(tm_parts) + "]")

    return "".join(parts)
