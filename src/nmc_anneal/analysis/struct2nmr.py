from pathlib import Path
import numpy as np

# NOTE: indexes getting shifted by integers as you traverse the structure are different than NMR CHEMICAL shifts which come from frequency shifts for an atom in a chemical versus in a standard chemical
# Therefore must be careful with the word shift that is used in two ways


def get_all_nmr_shifts(
    whole_lattice_charges: np.ndarray, whole_lattice_species
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads in a structure and calculates the chemical shift of every Li from the table of values. Many atoms will have the same nmr shift, so these are
    grouped and counted in the output

    May have to play around with the exact chemical shifts, because these are temperature dependent.

    Args:
        whole_lattice_charges (np.ndarray): Array containing all charges in structure in the correct geometry. Formatted as in initialize_lattice.py.
        whole_lattice_species (_type_): Array containing all ion names in structure in the correct geometry. Formatted as in initialize_lattice.py.

    Returns:
        tuple[np.ndarray, np.ndarray]: 1st array is list of all the nmr chemical shifts present, 2nd array is their frequency of occurance
    """

    # Table of 7Li nmr chemical shifts in ppm from nearest neighbor TM atoms
    # Bonded through an oxygen atom along a straight (180 deg) or bent (90 deg) bath
    # Geometry encoded as index shifts below in this function
    nmr_shifts_dict_90s = {
        "Mn": 255,
        "Ni2+": -25,
    }

    nmr_shifts_dict_180s = {
        "Mn": -52,
        "Ni2+": 120,
    }

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

    # Get indices of Li locations from the species list (should be the only +1, but being careful to use only atom names in NMR code)
    # NOTE: this includes both nominal Li layer and nominal TM layer
    indices = np.where(whole_lattice_species == "Li")

    all_nmr_shifts = []
    for i, j, k in zip(*indices):

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

        all_nmr_shifts.append(tot_nmr_shift_this_Li)

    return np.unique(all_nmr_shifts, return_counts=True)
