import math
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.typing import NDArray

import nmc_anneal.core.energy_calculations as encalc
from nmc_anneal.core.config import SimulationConfig

# Type alias for species lattices (dtype="<U4")
SpeciesLattice = NDArray[np.str_]
# Type alias for charges lattices (dtype=np.int8)
ChargesLattice = NDArray[np.int8]


def anneal_3Dlattice(
    config: SimulationConfig,
    whole_lattice_charges: ChargesLattice,
    whole_lattice_species: SpeciesLattice,
    anneal_type: str,
    graph_energy: bool = False,
) -> list[float]:
    """
    Handles passing lattice and parameters to parallel set of anneal_2D() processes and collating results
    - passes one 3-layer-sandwich of lattice to each process
    - each annealing process returns one layer, so these are interweaved with 'spectator' layers
    results are stored directly into whole_lattice_XXX rather than returned

    Args:
        config (SimulationConfig): Dict style data class containing most simulation parameters. See parser.py
        whole_lattice_charges (np.ndarray): Array containing all charges in structure in the correct geometry. Formatted as in initialize_lattice.py.
        whole_lattice_species (np.ndarray): Array containing all ion names in structure in the correct geometry. Formatted as in initialize_lattice.py.
        anneal_type (str): Define which layer to anneal and if intermediate (convergence check) or long run
        graph_energy (bool): At the expense of moderate extra computational time, track the energy change and record it in a 1D array. Defaults to false

    Returns:
        np.ndarray: optional return of energy trajectory as numpy array if graph_energy flag is true
    """
    energies = []
    # this list is written explicitly inside the slice_up_lattice function
    VALID_ANNEAL_TYPES = {
        "Initialize TM",
        "Anneal Li",
        "TM Convergence Check",
        "Li Convergence Check",
    }
    if anneal_type not in VALID_ANNEAL_TYPES:
        raise ValueError(
            f"Invalid anneal_type: '{anneal_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_ANNEAL_TYPES))}"
        )

    # Call helper function to generate list of 3-layer-sandwiches and control parameters needed for each parallel job
    task_list = _slice_up_lattice(
        config,
        whole_lattice_charges,
        whole_lattice_species,
        anneal_type,
        graph_energy,
    )

    results: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []

    with ProcessPoolExecutor(max_workers=config.n_layers) as executor:
        for result in executor.map(_anneal_2D_worker, task_list):
            results.append(result)

    # First order the results according to idx_anneal_layer in case child jobs don't come back in same order submitted
    # then pull species names and charges out of the large list:
    results.sort(key=lambda x: x[0])
    annealed_layers_species = [r[1] for r in results]
    annealed_layers_charges = [r[2] for r in results]

    # interweave the annealed layers into the original not-annealed layers
    # if anneal_type is init, the odd layers are annealed, if anneal_type is redox state, the evens are
    if anneal_type == "Initialize TM" or anneal_type == "TM Convergence Check":
        for idx_orig_layer, idx_annealed_layer in zip(
            range(1, 2 * config.n_layers, 2),
            range(0, len(annealed_layers_species), 1),
            strict=True,
        ):
            whole_lattice_charges[idx_orig_layer] = annealed_layers_charges[
                idx_annealed_layer
            ]
            whole_lattice_species[idx_orig_layer] = annealed_layers_species[
                idx_annealed_layer
            ]

    if anneal_type == "Anneal Li" or anneal_type == "Li Convergence Check":
        for idx_orig_layer, idx_annealed_layer in zip(
            range(0, 2 * config.n_layers - 1, 2),
            range(0, len(annealed_layers_species), 1),
            strict=True,
        ):
            whole_lattice_charges[idx_orig_layer] = annealed_layers_charges[
                idx_annealed_layer
            ]
            whole_lattice_species[idx_orig_layer] = annealed_layers_species[
                idx_annealed_layer
            ]

    if graph_energy:
        energy_list_per_layer = [r[3] for r in results]
        energies = np.mean(np.stack(energy_list_per_layer), axis=0)

    return energies


def _slice_up_lattice(
    config: SimulationConfig,
    whole_lattice_charges: ChargesLattice,
    whole_lattice_species: SpeciesLattice,
    anneal_type: str,
    graph_energy: bool = False,
) -> list[tuple[int, int, int, float, float, ChargesLattice, SpeciesLattice, bool]]:
    """
        Take entire lattice and slice into 3-layer sandwiches with the layer "idx_anneal_layer" as center layer
    anneal_type string sets whether to anneal the Li layers (those with even indices) or the TM layers (those with odd indices)

     NOTE: this function is where number of steps and simulation temp parameters are set (either from input text or manual control with "curr_conv_check" variables)

    Args:
        config (SimulationConfig): Dict style data class containing most simulation parameters. See parser.py
        whole_lattice_charges (np.ndarray): Array containing all charges in structure in the correct geometry. Formatted as in initialize_lattice.py.
        whole_lattice_species (np.ndarray): Array containing all ion names in structure in the correct geometry. Formatted as in initialize_lattice.py.
        anneal_type (str): Define which layer to anneal and if intermediate (convergence check) or long run
        graph_energy (bool): At the expense of moderate extra computational time, track the energy change and record it in a 1D array

    Returns:
        tuple[int, int, int, float, float, np.ndarray, np.ndarray, np.ndarray]: A complex data package describing a run on a 3-layer subset of the whole structure.
        see list_of_sandwiches.append command at end of this function for a description
    """

    # Set number of steps and temperature parameters for annealing
    if anneal_type == "Initialize TM":
        n_steps = int(config.initialize_anneal_steps / config.n_layers)
        sim_temp_hot = config.initialize_anneal_hot_temp
        sim_temp_cold = config.initialize_anneal_cold_temp

    elif anneal_type == "Anneal Li":
        n_steps = int(config.mid_delithiation_anneal_steps / config.n_layers)
        sim_temp_hot = config.mid_delithiation_anneal_hot_temp
        sim_temp_cold = config.mid_delithiation_anneal_cold_temp

    elif anneal_type == "TM Convergence Check" or anneal_type == "Li Convergence Check":
        # extra error checking only on this one since others checked at inpyt file load time
        try:
            n_steps = int(config.curr_conv_check_n_steps / config.n_layers)
            sim_temp_hot = config.curr_conv_check_hot_temp
            sim_temp_cold = config.curr_conv_check_cold_temp
        except ValueError as exc:
            raise ValueError(
                f"Anneal 2D called as convergence check without controlling parameters: {exc}"
            ) from exc
    else:
        raise ValueError(
            "anneal_type must be one of: Initialize TM, Anneal Li, TM Convergence Check, Li Convergence Check"
        )

    # TM layers are odd numbered vertical layers
    if anneal_type == "Initialize TM" or anneal_type == "TM Convergence Check":
        indices_center_layers = range(1, (2 * config.n_layers), 2)

    # Li layers are odd numbered vertical layers
    if anneal_type == "Anneal Li" or anneal_type == "Li Convergence Check":
        indices_center_layers = range(0, (2 * config.n_layers - 1), 2)

    # Make the three-layer lattice sandwiches, accounting for vertical periodicity
    list_of_sandwiches = []
    for idx_center_layer in indices_center_layers:
        idx_bottom_layer = idx_center_layer - 1
        if idx_bottom_layer == -1:
            idx_bottom_layer = 2 * config.n_layers - 1

        idx_top_layer = idx_center_layer + 1
        if idx_top_layer == (2 * config.n_layers):
            idx_top_layer = 0

        # Get 3-layer sandwich to work on for both the charges and atom names lattices:
        # note: the notation [3:4] gets a single slice but keeps shape (dimension) of pieces
        lattice_3L_charges = np.concatenate(
            (
                whole_lattice_charges[idx_bottom_layer : idx_bottom_layer + 1, :, :],
                whole_lattice_charges[idx_center_layer : idx_center_layer + 1, :, :],
                whole_lattice_charges[idx_top_layer : idx_top_layer + 1, :, :],
            ),
            axis=0,
        )

        lattice_3L_species = np.concatenate(
            (
                whole_lattice_species[idx_bottom_layer : idx_bottom_layer + 1, :, :],
                whole_lattice_species[idx_center_layer : idx_center_layer + 1, :, :],
                whole_lattice_species[idx_top_layer : idx_top_layer + 1, :, :],
            ),
            axis=0,
        )
        list_of_sandwiches.append(
            (
                idx_center_layer,
                n_steps,
                config.width,
                sim_temp_hot,
                sim_temp_cold,
                lattice_3L_charges,
                lattice_3L_species,
                graph_energy,
            )
        )

    return list_of_sandwiches


def _anneal_2D_worker(args):
    """
    Helper function just to dereference arguments before passing on to _anneal_2D
    """
    return _anneal_2D(*args)


def _anneal_2D(
    idx_anneal_layer: int,
    n_steps: int,
    lattice_width: int,
    sim_temp_hot: float,
    sim_temp_cold: float,
    lattice_charges: ChargesLattice,
    lattice_species: SpeciesLattice,
    graph_energy: bool = False,
) -> tuple[int, SpeciesLattice, ChargesLattice, np.ndarray]:
    """
    Reads in a 3-layer segment of whole lattice and anneals the center layer
    - returns just the annealed layer
    - also returns idx_anneal_layer for keeping track of where the returned annealed layer should be placed when rebuilding the whole stack

    Args:
        idx_anneal_layer (int): Center layer corresponds to layer with index idx_anneal_layer in whole lattice
        n_steps (int): Total number of steps in annealing
        lattice_width (int): Width of the square 2D layers
        sim_temp_hot (float): Simulated annealing start temperature (follows profile in _temp_ramp_shape)
        sim_temp_cold (float): Simulated annealing ending temperature (follows profile in _temp_ramp_shape)
        whole_lattice_charges (np.ndarray): Array containing all charges in structure in the correct geometry. Formatted as in initialize_lattice.py.
        whole_lattice_species (np.ndarray): Array containing all ion names in structure in the correct geometry. Formatted as in initialize_lattice.py.
        graph_energy (bool, optional): At the expense of moderate extra computational time, track the energy change and record it in a 1D array. Defaults to False.


    Returns:
        tuple[int, np.ndarray, np.ndarray, np.ndarray]: return just the center layer (the annealed one) with its original index and optionally the energy trajectory
    """

    if graph_energy and n_steps < 100:
        raise ValueError(
            "Cannot generate graph of energies each 1% of run if n_steps is less than 100!"
        )

    energies = np.zeros(101)
    checkpoint_interval = int(max(1, n_steps // 100) + 1)

    layer_to_anneal = 1  # always central layer of sandwich
    for curr_step in range(0, n_steps + 1):
        atom1 = (
            layer_to_anneal,
            random.randint(0, lattice_width - 1),
            random.randint(0, lattice_width - 1),
        )

        # Pick a random atom two but only accept if atomic charge is different from atom1
        num_failed_selections = 0
        while True:
            atom2 = (
                layer_to_anneal,
                random.randint(0, lattice_width - 1),
                random.randint(0, lattice_width - 1),
            )
            num_failed_selections += 1
            if num_failed_selections > int(0.05 * n_steps):
                print(
                    f"Warning, having difficulty finding an an atom whose charge isn't +{lattice_charges[atom1]} to swap with, skipping to a new site for now."
                )
                break
            if lattice_charges[atom1] != lattice_charges[atom2]:
                break

        # calc energy change if the two atoms are swapped
        swap_energy = -1 * encalc.energy_swap_two_metals(
            lattice_width, lattice_charges, atom1, atom2
        )

        # Determine if swap accepted or not
        sim_temp = _temp_ramp_shape(n_steps, curr_step, sim_temp_cold, sim_temp_hot)

        if swap_energy >= 0.0:  # always accept if downhill
            p_swap = 1.0

        elif sim_temp <= 0.0:
            p_swap = 0.0

        # If uphill, probability falls exponentially as energy cost of uphill step increases
        elif swap_energy < 0.0:
            p_swap = math.exp(swap_energy / sim_temp)

        # Uphill step accepted and charge swap saved to lattice of charges IF probability bests random number generator:
        # Note that exp of a negative has output range of 1 -> 0 and this is range of random.random()
        if p_swap > random.random():
            lattice_charges[atom1], lattice_charges[atom2] = (
                lattice_charges[atom2],
                lattice_charges[atom1],
            )
            lattice_species[atom1], lattice_species[atom2] = (
                lattice_species[atom2],
                lattice_species[atom1],
            )

        if graph_energy:
            if curr_step % checkpoint_interval == 0:
                current_percent_idx = int(curr_step / checkpoint_interval)
                if curr_step != n_steps:
                    energies[current_percent_idx] = (
                        encalc.one_metal_layer_oxygen_energies(lattice_charges)
                    )
                else:
                    energies[(len(energies) - 1)] = (
                        encalc.one_metal_layer_oxygen_energies(lattice_charges)
                    )

    return (
        idx_anneal_layer,
        lattice_species[1:2, :, :],
        lattice_charges[1:2, :, :],
        energies,
    )


def _temp_ramp_shape(
    n_steps: int, curr_step: int, sim_temp_cold: float, sim_temp_hot: float
) -> float:
    """
    Just using a simple ramp shape, can change if needed
    Shape: flat at T_hot for first 10 percent, then linear down to T_cold during next 80% and then hold

    :param n_steps: Total number of steps in annealing
    :type n_steps: int
    :param curr_step: Current step number
    :type curr_step: int
    :param sim_temp_cold: Starting temp. of annealing simulation
    :type sim_temp_cold: float
    :param sim_temp_hot: Ending temp. of annealing simulation
    :type sim_temp_hot: float
    :return: Temperature to conduct curr_step step at
    :rtype: float
    """
    flat_until = 0.10
    reaches_min = 0.80
    if (curr_step / n_steps) < flat_until:
        sim_temp = sim_temp_hot

    elif (curr_step / n_steps) < reaches_min:
        sim_temp = ((sim_temp_hot - sim_temp_cold) / (flat_until - reaches_min)) * (
            (curr_step / n_steps) - flat_until
        ) + sim_temp_hot

    else:
        sim_temp = sim_temp_cold

    return sim_temp
