
import numpy as np
from numpy.typing import NDArray

import nmc_anneal.core.energy_calculations as encalc
from nmc_anneal.core.config import SimulationConfig

# Type alias for species lattices (dtype="<U4")
SpeciesLattice = NDArray[np.str_]
# Type alias for charges lattices (dtype=np.int8)
ChargesLattice = NDArray[np.int8]


def delithiate(
    config: SimulationConfig,
    whole_lattice_charges: ChargesLattice,
    whole_lattice_species: SpeciesLattice,
    frac_li_to_remove: float,
):
    """
    Remove lithium atoms and oxidize transition metals to maintain charge balance.

    Removes a specified fraction of lithium atoms from the structure and oxidizes the corresponding
    number of transition metals (Ni, Co) according to the oxidation model specified in config.
    The Li layer is assumed to rearrange deterministically to find the lowest energy configuration.

    Args:
        config (SimulationConfig): Data class containing simulation parameters and oxidation model selection.
        whole_lattice_charges (ChargesLattice): 3D array of all charges in structure with proper geometry.
        whole_lattice_species (SpeciesLattice): 3D array of all ion names in structure with proper geometry.
        frac_li_to_remove (float): Fraction of lithium atoms to remove (between 0 and 1).

    Raises:
        ValueError: If requested lithium removal exceeds the number of Li atoms present.
        ValueError: If requested redox stoichiometry cannot be satisfied by available metals.
    """

    if config.random_seed is not None:
        rng = np.random.default_rng(config.random_seed)
    else:
        rng = np.random.default_rng()

    # Convert fraction of Li to remove to integer number to remove
    # (while forcing to be even number in case methods require one TM per two Li)
    start_num_li = np.count_nonzero(whole_lattice_species == "Li")
    num_li_to_remove = int(np.ceil(start_num_li * frac_li_to_remove / 2) * 2)

    if num_li_to_remove > start_num_li:
        raise ValueError(
            f"Not enough Li atoms for requested amount of redox. Requested {num_li_to_remove} but only {start_num_li} Li are present"
        )

    # This model oxidizes ni2+ straight to ni4+, while removing two Li for each Ni oxidation
    # start by building lists of oxidation energy at the current configuration
    if config.oxidation_model == "ni_2to4_co_3to4":
        original_charge = 2
        redox_charge = 4
        energies_2plus = _get_energy_list(
            whole_lattice_charges,
            whole_lattice_species,
            original_charge,
            redox_charge,
            list(range(1, 2 * config.n_layers)),
        )

        original_charge = 1
        redox_charge = 0
        energies_1plus = _get_energy_list(
            whole_lattice_charges,
            whole_lattice_species,
            original_charge,
            redox_charge,
            list(range(0, 2 * config.n_layers)),
        )

        # Stoichiometry calculations:
        start_num_ni2 = int(np.count_nonzero(whole_lattice_species == "Ni2+"))

        start_num_co3 = int(np.count_nonzero(whole_lattice_species == "Co3+"))

        num_ni2_to_ni4 = int(num_li_to_remove / 2)

        if (num_li_to_remove / 2) > start_num_ni2:
            num_ni2_to_ni4 = start_num_ni2
            num_co3_to_co4 = num_li_to_remove - (start_num_ni2 * 2)
        else:
            num_ni2_to_ni4 = int(num_li_to_remove / 2)
            num_co3_to_co4 = 0

        if num_co3_to_co4 > start_num_co3:
            raise ValueError(
                f"Not enough Ni atoms + Co atoms for requested amount of redox. Requested {num_ni2_to_ni4} Ni2 and {num_co3_to_co4} Co3 atoms oxidized, whereas {start_num_ni2} Ni2 and {start_num_co3 } Co3 are present"
            )

        for _step in range(0, num_ni2_to_ni4):
            # oxidize Ni2+ to Ni4+ and update energies of all 2+Ni and Li sites
            # chooses lowest energy site (or randomly selects between several if equal energies)
            _redox_and_update2lists(
                whole_lattice_charges,
                whole_lattice_species,
                list_to_oxidize=energies_2plus,
                redox_starting_charge=2,
                redox_ending_charge=4,
                redox_ending_name="Ni4+",
                list_to_update=energies_1plus,
                update_list_redox_starting_charge=1,
                update_list_redox_ending_charge=0,
                rng=rng,
            )

            # remove the two Li atoms to charge balance and update Li and Ni2+ oxidation energy lists
            # chooses lowest energy site (or randomly selects between several if equal energies)
            for _i in {1, 2}:
                _redox_and_update2lists(
                    whole_lattice_charges,
                    whole_lattice_species,
                    list_to_oxidize=energies_1plus,
                    redox_starting_charge=1,
                    redox_ending_charge=0,
                    redox_ending_name="Vac",
                    list_to_update=energies_2plus,
                    update_list_redox_starting_charge=2,
                    update_list_redox_ending_charge=4,
                    rng=rng,
                )

            energies_2plus.sort(key=lambda row: row[3], reverse=True)
            energies_1plus.sort(key=lambda row: row[3], reverse=True)

        # Now switching to oxidizing Co3+ -> 4+
        original_charge = 3
        redox_charge = 4
        energies_3plus = _get_energy_list(
            whole_lattice_charges,
            whole_lattice_species,
            original_charge,
            redox_charge,
            list(range(1, 2 * config.n_layers)),
        )

        co_energies_3plus = [row for row in energies_3plus if row[-1] == "Co3+"]

        for _step in range(0, num_co3_to_co4):
            # oxidize Co3+ to Co4+ and update energies of all Co3+ and Li sites
            _redox_and_update2lists(
                whole_lattice_charges,
                whole_lattice_species,
                list_to_oxidize=co_energies_3plus,
                redox_starting_charge=3,
                redox_ending_charge=4,
                redox_ending_name="Co4+",
                list_to_update=energies_1plus,
                update_list_redox_starting_charge=1,
                update_list_redox_ending_charge=0,
                rng=rng,
            )

            # remove one Li atom to charge balance and update Li and Co3+ oxidation energy lists
            _redox_and_update2lists(
                whole_lattice_charges,
                whole_lattice_species,
                list_to_oxidize=energies_1plus,
                redox_starting_charge=1,
                redox_ending_charge=0,
                redox_ending_name="Vac",
                list_to_update=co_energies_3plus,
                update_list_redox_starting_charge=3,
                update_list_redox_ending_charge=4,
                rng=rng,
            )

            co_energies_3plus.sort(key=lambda row: row[3], reverse=True)
            energies_1plus.sort(key=lambda row: row[3], reverse=True)

    #
    # This model oxidizes Ni2+ to Ni4+ in two steps, then oxidizes Co3+ to Co4+
    # (1) oxidizes Ni2+ to Ni3+ while removing one Li in each Ni oxidation step
    # (2) only after all converted to Ni3+, oxidizes Ni3+ to Ni4+, removing one Li in each Ni oxidation step
    # (3) only after all Ni2+ converted to Ni4+, begin converting Co3+ to Co4+
    if config.oxidation_model == "ni_2to3_ni_3to4_co_3to4":
        # Stoichiometry calculations:
        start_num_ni2 = int(np.count_nonzero(whole_lattice_species == "Ni2+"))
        start_num_co3 = int(np.count_nonzero(whole_lattice_species == "Co3+"))

        if num_li_to_remove <= start_num_ni2:
            num_ni2_to_ni3 = num_li_to_remove
            num_ni3_to_ni4 = 0
            num_co3_to_co4 = 0

        elif num_li_to_remove <= 2 * start_num_ni2:
            num_ni2_to_ni3 = start_num_ni2
            num_ni3_to_ni4 = num_li_to_remove - start_num_ni2
            num_co3_to_co4 = 0

        else:
            num_ni2_to_ni3 = start_num_ni2
            num_ni3_to_ni4 = start_num_ni2
            num_co3_to_co4 = num_li_to_remove - 2 * start_num_ni2

        if num_co3_to_co4 > start_num_co3:
            raise ValueError(
                f"Not enough Ni atoms + Co atoms for requested amount of redox. Requested {num_ni2_to_ni3} Ni2 and {num_co3_to_co4} Co3 atoms oxidized, whereas {start_num_ni2} Ni2 and {start_num_co3 } Co3 are present"
            )

        original_charge = 2
        redox_charge = 3
        energies_2plus = _get_energy_list(
            whole_lattice_charges,
            whole_lattice_species,
            original_charge,
            redox_charge,
            list(range(1, 2 * config.n_layers)),
        )

        original_charge = 1
        redox_charge = 0
        energies_1plus = _get_energy_list(
            whole_lattice_charges,
            whole_lattice_species,
            original_charge,
            redox_charge,
            list(range(0, 2 * config.n_layers)),
        )

        for _step in range(0, num_ni2_to_ni3):
            # oxidize Ni2+ to Ni3+ and update energies of all Ni2+ and Li sites
            _redox_and_update2lists(
                whole_lattice_charges,
                whole_lattice_species,
                list_to_oxidize=energies_2plus,
                redox_starting_charge=2,
                redox_ending_charge=3,
                redox_ending_name="Ni3+",
                list_to_update=energies_1plus,
                update_list_redox_starting_charge=1,
                update_list_redox_ending_charge=0,
                rng=rng,
            )

            # remove one Li atom to charge balance and update Li and Ni2+ oxidation energy lists
            _redox_and_update2lists(
                whole_lattice_charges,
                whole_lattice_species,
                list_to_oxidize=energies_1plus,
                redox_starting_charge=1,
                redox_ending_charge=0,
                redox_ending_name="Vac",
                list_to_update=energies_2plus,
                update_list_redox_starting_charge=2,
                update_list_redox_ending_charge=3,
                rng=rng,
            )

            energies_2plus.sort(key=lambda row: row[3], reverse=True)
            energies_1plus.sort(key=lambda row: row[3], reverse=True)

        # Now switching to oxidizing Ni3+ -> 4+
        original_charge = 3
        redox_charge = 4
        energies_3plus = _get_energy_list(
            whole_lattice_charges,
            whole_lattice_species,
            original_charge,
            redox_charge,
            list(range(1, 2 * config.n_layers)),
        )

        ni_energies_3plus = [row for row in energies_3plus if row[-1] == "Ni3+"]

        for _step in range(0, num_ni3_to_ni4):
            # oxidize Ni3+ to Ni4+ and update energies of all Ni3+ and Li sites
            _redox_and_update2lists(
                whole_lattice_charges,
                whole_lattice_species,
                list_to_oxidize=ni_energies_3plus,
                redox_starting_charge=3,
                redox_ending_charge=4,
                redox_ending_name="Ni4+",
                list_to_update=energies_1plus,
                update_list_redox_starting_charge=1,
                update_list_redox_ending_charge=0,
                rng=rng,
            )

            # remove one Li atom to charge balance and update Li and Ni3+ oxidation energy lists
            _redox_and_update2lists(
                whole_lattice_charges,
                whole_lattice_species,
                list_to_oxidize=energies_1plus,
                redox_starting_charge=1,
                redox_ending_charge=0,
                redox_ending_name="Vac",
                list_to_update=ni_energies_3plus,
                update_list_redox_starting_charge=3,
                update_list_redox_ending_charge=4,
                rng=rng,
            )

            ni_energies_3plus.sort(key=lambda row: row[3], reverse=True)
            energies_1plus.sort(key=lambda row: row[3], reverse=True)

        # Now switching to oxidizing Co3+ -> 4+
        original_charge = 3
        redox_charge = 4
        energies_3plus = _get_energy_list(
            whole_lattice_charges,
            whole_lattice_species,
            original_charge,
            redox_charge,
            list(range(1, 2 * config.n_layers)),
        )

        co_energies_3plus = [row for row in energies_3plus if row[-1] == "Co3+"]

        for _step in range(0, num_co3_to_co4):
            # oxidize Co3+ to Co4+ and update energies of all Co3+ and Li sites
            _redox_and_update2lists(
                whole_lattice_charges,
                whole_lattice_species,
                list_to_oxidize=co_energies_3plus,
                redox_starting_charge=3,
                redox_ending_charge=4,
                redox_ending_name="Co4+",
                list_to_update=energies_1plus,
                update_list_redox_starting_charge=1,
                update_list_redox_ending_charge=0,
                rng=rng,
            )

            # remove one Li atom to charge balance and update Li and Co3+ oxidation energy lists
            _redox_and_update2lists(
                whole_lattice_charges,
                whole_lattice_species,
                list_to_oxidize=energies_1plus,
                redox_starting_charge=1,
                redox_ending_charge=0,
                redox_ending_name="Vac",
                list_to_update=co_energies_3plus,
                update_list_redox_starting_charge=3,
                update_list_redox_ending_charge=4,
                rng=rng,
            )

            co_energies_3plus.sort(key=lambda row: row[3], reverse=True)
            energies_1plus.sort(key=lambda row: row[3], reverse=True)

    #
    # This model oxidizes in two distinct steps involving Ni and Co
    # (1) oxidizes Ni2+ to Ni3+ while removing one Li in each Ni oxidation step
    # (2) only after all converted to Ni3+, oxidizes EITHER Ni3+ to Ni4+ or Co3+ to Co4+ (by oxygen energy), removing one Li in each Ni oxidation step
    if config.oxidation_model == "ni_2to3_any_3to4":
        original_charge = 2
        redox_charge = 3
        energies_2plus = _get_energy_list(
            whole_lattice_charges,
            whole_lattice_species,
            original_charge,
            redox_charge,
            list(range(1, 2 * config.n_layers)),
        )

        original_charge = 1
        redox_charge = 0
        energies_1plus = _get_energy_list(
            whole_lattice_charges,
            whole_lattice_species,
            original_charge,
            redox_charge,
            list(range(0, 2 * config.n_layers)),
        )

        # Stoichiometry calculations:
        start_num_ni2 = int(np.count_nonzero(whole_lattice_species == "Ni2+"))

        start_num_co3 = int(np.count_nonzero(whole_lattice_species == "Co3+"))

        num_ni2_to_ni3 = np.minimum(num_li_to_remove, start_num_ni2)

        if num_ni2_to_ni3 > start_num_ni2 or num_li_to_remove > start_num_li:
            raise ValueError(
                f"Not enough Li or Ni atoms for requested amount of redox. Requested {num_li_to_remove} Li removed and {num_ni2_to_ni3} Ni2 oxidized, whereas {start_num_li} Li and {start_num_ni2 } Ni2 are present"
            )

        # One lithium removed per Ni2 -> Ni3 event, so remainder is:
        num_li_to_remove_stage2 = num_li_to_remove - num_ni2_to_ni3

        if num_li_to_remove_stage2 > (start_num_ni2 + start_num_co3):
            raise ValueError(
                "Not enough Ni + Co atoms for the second redox stage of Ni3+ -> Ni4+ / Co3+ to Co4+ to equate to the Li removal requested."
            )

        for _step in range(0, num_ni2_to_ni3):
            # oxidize Ni2+ to Ni3+ and update energies of all 2+Ni and Li sites
            # chooses lowest energy site (or randomly selects between several if equal energies)
            _redox_and_update2lists(
                whole_lattice_charges,
                whole_lattice_species,
                list_to_oxidize=energies_2plus,
                redox_starting_charge=2,
                redox_ending_charge=3,
                redox_ending_name="Ni3+",
                list_to_update=energies_1plus,
                update_list_redox_starting_charge=1,
                update_list_redox_ending_charge=0,
                rng=rng,
            )

            # remove the two Li atoms to charge balance and update Li and Ni2+ oxidation energy lists
            # chooses lowest energy site (or randomly selects between several if equal energies)
            _redox_and_update2lists(
                whole_lattice_charges,
                whole_lattice_species,
                list_to_oxidize=energies_1plus,
                redox_starting_charge=1,
                redox_ending_charge=0,
                redox_ending_name="Vac",
                list_to_update=energies_2plus,
                update_list_redox_starting_charge=2,
                update_list_redox_ending_charge=3,
                rng=rng,
            )

            energies_2plus.sort(key=lambda row: row[3], reverse=True)
            energies_1plus.sort(key=lambda row: row[3], reverse=True)

        # Now switching to oxidizing Ni3+ OR Co3+ -> 4+
        original_charge = 3
        redox_charge = 4
        energies_3plus = _get_energy_list(
            whole_lattice_charges,
            whole_lattice_species,
            original_charge,
            redox_charge,
            list(range(1, 2 * config.n_layers)),
        )

        for _step in range(0, num_li_to_remove_stage2):
            # oxidize Ni2+ to Ni4+ and update energies of all 2+Ni and Li sites
            # chooses lowest energy site (or randomly selects between several if equal energies)
            _redox_and_update2lists(
                whole_lattice_charges,
                whole_lattice_species,
                list_to_oxidize=energies_3plus,
                redox_starting_charge=3,
                redox_ending_charge=4,
                redox_ending_name="Co4+ or Ni4+",
                list_to_update=energies_1plus,
                update_list_redox_starting_charge=1,
                update_list_redox_ending_charge=0,
                rng=rng,
            )

            # remove the two Li atoms to charge balance and update Li and Ni2+ oxidation energy lists
            # chooses lowest energy site (or randomly selects between several if equal energies)
            _redox_and_update2lists(
                whole_lattice_charges,
                whole_lattice_species,
                list_to_oxidize=energies_1plus,
                redox_starting_charge=1,
                redox_ending_charge=0,
                redox_ending_name="Vac",
                list_to_update=energies_3plus,
                update_list_redox_starting_charge=3,
                update_list_redox_ending_charge=4,
                rng=rng,
            )

            energies_3plus.sort(key=lambda row: row[3], reverse=True)
            energies_1plus.sort(key=lambda row: row[3], reverse=True)

    return


def _get_energy_list(
    whole_lattice_charges: ChargesLattice,
    whole_lattice_species: SpeciesLattice,
    charge_to_tabulate: int,
    proposed_new_charge: int,
    layers_to_tabulate: list,
) -> list:
    """
    Generate a sorted list of redox energies for metal sites with a given charge state.

    Identifies all metal sites with the specified charge and calculates the redox energy
    for each site if the charge were changed to the proposed new charge. Returns a list
    sorted by redox energy (lowest energy first).

    Args:
        whole_lattice_charges (ChargesLattice): 3D array of all charges in structure with proper geometry.
        whole_lattice_species (SpeciesLattice): 3D array of all ion names in structure with proper geometry.
        charge_to_tabulate (int): Current charge state to search for.
        proposed_new_charge (int): Target charge state for redox energy calculation.
        layers_to_tabulate (list[int]): Layer indices to include in the tabulation.

    Returns:
        list[list]: Each element is [layer_idx, x, y, redox_energy, atom_name], sorted by redox_energy descending.
    """

    energies = []

    for layer in layers_to_tabulate:
        mask = whole_lattice_charges[layer] == charge_to_tabulate

        xs, ys = np.where(mask)

        for x, y in zip(xs, ys, strict=True):
            energy = encalc.single_metal_redox(
                proposed_new_charge,
                whole_lattice_charges,
                (layer, x, y),
            )
            atom_name = whole_lattice_species[layer, x, y]
            energies.append([layer, int(x), int(y), float(energy), atom_name])

    energies.sort(key=lambda row: row[3], reverse=True)
    return energies


def _rndm_idx_lowestE(
    energylist: list,
    rng: np.random.Generator,
) -> int:
    """
    Randomly select one site from a list of energetically degenerate lowest-energy sites.

    When multiple sites have the same lowest redox energy, this function randomly selects one
    of them to prevent bias in site selection. Assumes the list is pre-sorted with lowest
    energies first.

    Args:
        energylist (list[list]): Sorted list of [layer_idx, x, y, energy, atom_name] entries.

    Returns:
        int: Index of the randomly selected entry in the energy list.
    """

    top_energy = energylist[0][3]

    last_index = 0
    for i, row in enumerate(energylist):
        if row[3] != top_energy:
            break
        last_index = i

    if last_index == 0:
        return 0
    else:
        selected_row = int(rng.integers(0, last_index))
        return selected_row


def _redox_and_update2lists(
    whole_lattice_charges: ChargesLattice,
    whole_lattice_species: SpeciesLattice,
    list_to_oxidize: list,
    redox_starting_charge: int,
    redox_ending_charge: int,
    redox_ending_name: str,
    list_to_update: list,
    update_list_redox_starting_charge: int,
    update_list_redox_ending_charge: int,
    rng: np.random.Generator,
):
    """
    Perform a redox reaction at the lowest-energy site and update affected energy lists.

    Selects the lowest-energy site from list_to_oxidize, performs the redox change, updates
    the species and charge lattices, and recalculates redox energies for all neighboring sites
    that may be affected by the charge change.

    Args:
        whole_lattice_charges (ChargesLattice): 3D array of all charges in structure with proper geometry.
        whole_lattice_species (SpeciesLattice): 3D array of all ion names in structure with proper geometry.
        list_to_oxidize (list[list]): Energy list for sites to oxidize (entry removed after selection).
        redox_starting_charge (int): Initial charge state of the site being oxidized.
        redox_ending_charge (int): Final charge state of the site being oxidized.
        redox_ending_name (str): Ion name for the oxidized species (e.g., "Ni4+", "Co4+", "Vac").
        list_to_update (list[list]): Energy list for sites whose energies must be recalculated.
        update_list_redox_starting_charge (int): Initial charge state for energies in list_to_update.
        update_list_redox_ending_charge (int): Final charge state for energies in list_to_update.
    """

    width = whole_lattice_charges.shape[1]
    num_li_plus_tm_layers = whole_lattice_charges.shape[0]
    # randomly select one entry equal to the lowest energy in the redox list:
    list_row_to_remove = _rndm_idx_lowestE(list_to_oxidize, rng)
    indices_of_removed = list_to_oxidize[list_row_to_remove][0:3]

    whole_lattice_charges[tuple(indices_of_removed)] = redox_ending_charge

    # change name of ion in lattice of species, usually using function argument, but check if simultaneous Co3+/Ni3+
    if redox_ending_name == "Co4+ or Ni4+":
        redox_starting_name = whole_lattice_species[tuple(indices_of_removed)]
        if redox_starting_name == "Ni3+":
            redox_ending_name = "Ni4+"
        if redox_starting_name == "Co3+":
            redox_ending_name = "Co4+"
    whole_lattice_species[tuple(indices_of_removed)] = redox_ending_name

    del list_to_oxidize[list_row_to_remove]

    in_layer_idx_shfts = _get_neighbor_indices(
        "same layer", indices_of_removed, width, num_li_plus_tm_layers
    )
    _update_en_list(
        whole_lattice_charges,
        redox_starting_charge,
        redox_ending_charge,
        list_to_oxidize,
        locations_to_check=in_layer_idx_shfts,
    )

    up_layer_idx_shfts = _get_neighbor_indices(
        "up layer", indices_of_removed, width, num_li_plus_tm_layers
    )
    _update_en_list(
        whole_lattice_charges,
        update_list_redox_starting_charge,
        update_list_redox_ending_charge,
        list_to_update,
        locations_to_check=up_layer_idx_shfts,
    )

    down_layer_idx_shfts = _get_neighbor_indices(
        "down layer", indices_of_removed, width, num_li_plus_tm_layers
    )
    _update_en_list(
        whole_lattice_charges,
        update_list_redox_starting_charge,
        update_list_redox_ending_charge,
        list_to_update,
        locations_to_check=down_layer_idx_shfts,
    )

    list_to_oxidize.sort(key=lambda row: row[3], reverse=True)
    list_to_update.sort(key=lambda row: row[3], reverse=True)

    return


def _get_neighbor_indices(
    layer_name: str,
    start_index,
    lattice_width: int,
    num_li_plus_tm_layers: int,
) -> list:
    """
    Generate neighbor indices for a metal site accounting for periodic boundary conditions.

    Returns the addresses of all neighboring metal sites in a specified layer relative to
    start_index. Accounts for periodicity in both vertical (layer) and horizontal (x,y) directions.

    Args:
        layer_name (str): Which layer to get neighbors from: "same layer", "up layer", or "down layer".
        start_index (tuple[int, int, int]): Reference position (layer_idx, x, y).
        lattice_width (int): Width of the square 2D layers.
        num_li_plus_tm_layers (int): Total number of layers in the 3D structure.

    Returns:
        list[tuple[int, int, int]]: List of neighbor coordinates with periodic boundaries applied.
    """

    shifts_by_layer = {
        "same layer": [
            [0, 0, -1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, -1],
            [0, -1, 1],
            [0, -1, 0],
        ],
        "up layer": [
            [1, 0, 0],
            [1, 0, -1],
            [1, 0, -2],
            [1, -1, 0],
            [1, -2, 0],
            [1, -1, -1],
        ],
        "down layer": [
            [-1, 0, 0],
            [-1, 0, 1],
            [-1, 1, 0],
            [-1, 2, 0],
            [-1, 1, 1],
            [-1, 0, 2],
        ],
    }

    shifts = shifts_by_layer[layer_name]

    final_indices = [
        [
            (start_index[0] + dz) % num_li_plus_tm_layers,
            (start_index[1] + dx) % lattice_width,
            (start_index[2] + dy) % lattice_width,
        ]
        for dz, dx, dy in shifts
    ]

    return final_indices


def _update_en_list(
    whole_lattice_charges: ChargesLattice,
    redox_starting_charge: int,
    redox_ending_charge: int,
    energies_list: list,
    locations_to_check: list,
):
    """
    Update redox energies for specific sites in an energy list.

    Recalculates redox energies for sites at specified locations and updates their entries
    in the energy list. Used after a redox event to account for changed charge environments.

    Args:
        whole_lattice_charges (ChargesLattice): 3D array of all charges in structure with proper geometry.
        redox_starting_charge (int): Initial charge state for redox energy calculation.
        redox_ending_charge (int): Final charge state for redox energy calculation.
        energies_list (list[list]): Mutable list of [layer_idx, x, y, energy] entries to update in-place.
        locations_to_check (list[tuple]): Coordinates of sites to recalculate (indices already corrected for periodicity).
    """

    new_energies_list = []
    for index in locations_to_check:
        if (
            whole_lattice_charges[(index[0], index[1], index[2])]
            == redox_starting_charge
        ):
            energy = encalc.single_metal_redox(
                redox_ending_charge,
                whole_lattice_charges,
                index,
            )
            new_energies_list.append([index[0], index[1], index[2], float(energy)])

    # merge new subset list with existing full list
    for new_energy in new_energies_list:
        for idx in range(0, len(energies_list)):
            if energies_list[idx][0:3] == new_energy[0:3]:
                energies_list[idx][3] = new_energy[3]

    return
