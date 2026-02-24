"""Convert crystal structures to NMR spectroscopy observables.

Calculates lithium NMR chemical shifts based on local charge environments
around each Li atom, grouping and counting shift degeneracies to generate
spectroscopic signatures comparable to experimental NMR data.
"""

import numpy as np
from numpy.typing import NDArray

# Type alias for species lattices (dtype="<U4")
SpeciesLattice = NDArray[np.str_]
# Type alias for charges lattices (dtype=np.int8)
ChargesLattice = NDArray[np.int8]
# Therefore must be careful with the word shift that is used in two ways


def get_all_nmr_shifts(
    whole_lattice_charges: ChargesLattice,
    whole_lattice_species: SpeciesLattice,
    nmr_shifts_dict_90s: dict,
    nmr_shifts_dict_180s: dict,
    nmr_shifts_dict_inlayer: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads in a structure and calculates the chemical shift of every Li from the table of values. Many atoms will have the same nmr shift, so these are
    grouped and counted in the output

    May have to play around with the exact chemical shifts, because these are temperature dependent.

    Args:
        whole_lattice_charges (ChargesLattice): 3D array of all charges with proper geometry. Not strictly needed for current algorithm, but may be useful in extensions.
        whole_lattice_species (SpeciesLattice): 3D array of all ion names with proper geometry.
        nmr_shifts_dict_90s (dict): Chemical shifts (ppm) for 90° bonds; keys are TM names (e.g., "Ni2+", "Mn").
        nmr_shifts_dict_180s (dict): Chemical shifts (ppm) for 180° bonds; keys are TM names.
        nmr_shifts_dict_inlayer (dict): Chemical shifts (ppm) for in-layer bonds; keys are TM names. By geometry, these are connected by TWO 90 degree pathways.

    Returns:
        tuple[np.ndarray, np.ndarray]: (unique_shifts, counts) where unique_shifts are the distinct shifts in ppm and counts are their frequencies.
    """

    # The dict objects should contain the 7Li nmr chemical shifts in ppm caused by the nearest-neighbor
    # TM atoms bonded through an oxygen atom along a straight (180 deg) or bent (90 deg) bond if up a layer or down a layer
    # Within a layer (if any paramagnetic ions are present), neighbors als connected by two 90 degree pathways (via oxgyen atoms above AND below the plane)
    # These shifts are temperature dependent (perhaps stoich-induced structurally too), so may be changed somewhat
    # Can add any atom name from initialize_lattice.py
    # NOTE: The geometry of these two bond angles in the geometry convention used here is described with index shifts below in this function

    # example shift dictionaries:
    # nmr_shifts_dict_90s = {
    #     "Mn": 255,
    #     "Ni2+": -25,
    # }

    # nmr_shifts_dict_180s = {
    #     "Mn": -52,
    #     "Ni2+": 120,
    # }

    num_li_plus_tm_layers = whole_lattice_species.shape[0]
    lattice_width = whole_lattice_species.shape[1]

    # address shifts for metals above/below a given site and related by a 90 deg bonding pathway
    idx_shifts_TM90 = [
        [1, -1, -1],
        [1, -1, 0],
        [1, 0, -1],
        [-1, 0, 1],
        [-1, 1, 0],
        [-1, 1, 1],
    ]

    # address shifts for metals above/below a given site and related by a 180 deg bonding pathway
    idx_shifts_TM180 = [
        [1, 0, 0],
        [1, 0, -2],
        [1, -2, 0],
        [-1, 0, 0],
        [-1, 2, 0],
        [-1, 0, 2],
    ]

    # address shifts for metals in the same layer (note that these are all 90 degree shifts by nature of the geometry)
    # NOTE: every site here is connected by TWO independent pathways (via the oxygen atoms above and below the metal plane)
    idx_shifts_inlayer = [
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [0, 1, -1],
        [0, -1, 0],
        [0, -1, 1],
    ]

    # Get indices of Li locations from the species list (should be the only +1, but being careful to use only atom names in NMR code)
    # NOTE: this includes both nominal Li layer and nominal TM layer
    indices = np.where(whole_lattice_species == "Li")

    all_nmr_shifts = []
    for i, j, k in zip(*indices, strict=True):
        tot_nmr_shift_this_Li = 0

        # Sum neighbors up or down one layer at 90 degrees
        for idx_shift in idx_shifts_TM90:
            # get whole index from index shift:
            neighbor_index = (
                (i + idx_shift[0]) % num_li_plus_tm_layers,
                (j + idx_shift[1]) % lattice_width,
                (k + idx_shift[2]) % lattice_width,
            )
            # get name of neighbor atom:
            neighbor_name = whole_lattice_species[neighbor_index]
            one_TM_neighbor_shift = nmr_shifts_dict_90s.get(neighbor_name, 0)
            tot_nmr_shift_this_Li += one_TM_neighbor_shift

            # Sum neighbors up or down one layer at 180 degrees
        for idx_shift in idx_shifts_TM180:
            # get whole index from index shift:
            neighbor_index = (
                (i + idx_shift[0]) % num_li_plus_tm_layers,
                (j + idx_shift[1]) % lattice_width,
                (k + idx_shift[2]) % lattice_width,
            )
            # get name of neighbor atom:
            neighbor_name = whole_lattice_species[neighbor_index]
            one_TM_neighbor_shift = nmr_shifts_dict_180s.get(neighbor_name, 0)
            tot_nmr_shift_this_Li += one_TM_neighbor_shift

        for idx_shift in idx_shifts_inlayer:
            # get whole index from index shift:
            neighbor_index = (
                (i + idx_shift[0]) % num_li_plus_tm_layers,
                (j + idx_shift[1]) % lattice_width,
                (k + idx_shift[2]) % lattice_width,
            )
            # get name of neighbor atom:
            neighbor_name = whole_lattice_species[neighbor_index]
            one_TM_neighbor_shift = nmr_shifts_dict_inlayer.get(neighbor_name, 0)
            tot_nmr_shift_this_Li += one_TM_neighbor_shift

        all_nmr_shifts.append(tot_nmr_shift_this_Li)

    return np.unique(all_nmr_shifts, return_counts=True)
