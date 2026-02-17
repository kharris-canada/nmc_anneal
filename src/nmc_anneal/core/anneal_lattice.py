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
    Perform parallel annealing on layers of a 3D lattice structure.

    Slices the lattice into 3-layer sandwiches and anneals the center layer of each via
    parallel processes. Results are stored directly into the input lattice arrays.

    Args:
        config (SimulationConfig): Data class containing simulation parameters and annealing settings.
        whole_lattice_charges (ChargesLattice): 3D array of all charges to anneal in-place.
        whole_lattice_species (SpeciesLattice): 3D array of all ion names to anneal in-place.
        anneal_type (str): Type of anneal: "Initialize TM", "Anneal Li", "TM Convergence Check", or "Li Convergence Check".
        graph_energy (bool): If True, track and return energy trajectory. Defaults to False.

    Returns:
        list[float]: Energy trajectory (one value per 1% of steps) if graph_energy=True, else empty list.

    Raises:
        ValueError: If anneal_type is not one of the valid options.
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
    Slice lattice into 3-layer sandwiches and prepare annealing job parameters.

    Partitions the lattice into 3-layer sandwiches with the target layer in the center.
    Sets annealing steps and temperature parameters based on anneal_type and config.

    Args:
        config (SimulationConfig): Data class containing simulation parameters.
        whole_lattice_charges (ChargesLattice): 3D array of all charges with proper geometry.
        whole_lattice_species (SpeciesLattice): 3D array of all ion names with proper geometry.
        anneal_type (str): Type of anneal: "Initialize TM", "Anneal Li", "TM Convergence Check", or "Li Convergence Check".
        graph_energy (bool): If True, record energy trajectory. Defaults to False.

    Returns:
        list[tuple]: Each tuple contains (idx_center_layer, n_steps, lattice_width, sim_temp_hot, sim_temp_cold, lattice_charges_3L, lattice_species_3L, graph_energy).
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
    Wrapper function to dereference tuple arguments for _anneal_2D.

    Unpacks a single tuple argument containing all parameters and passes them to _anneal_2D.
    Enables use of executor.map() in ProcessPoolExecutor.

    Args:
        args (tuple): Tuple containing all parameters for _anneal_2D.

    Returns:
        tuple[int, SpeciesLattice, ChargesLattice, np.ndarray]: Result from _anneal_2D.
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
    Perform simulated annealing on the center layer of a 3-layer sandwich.

    Uses Metropolis algorithm with temperature ramp to optimize ion positions. Randomly swaps
    ions of different charges and accepts/rejects based on energy change and current temperature.

    Args:
        idx_anneal_layer (int): Index of the center (annealed) layer in the original lattice.
        n_steps (int): Total number of Monte Carlo steps.
        lattice_width (int): Width of the square 2D layers.
        sim_temp_hot (float): Initial simulated annealing temperature.
        sim_temp_cold (float): Final simulated annealing temperature.
        lattice_charges (ChargesLattice): 3D array of charges for the 3-layer sandwich.
        lattice_species (SpeciesLattice): 3D array of ion names for the 3-layer sandwich.
        graph_energy (bool): If True, record energy at each checkpoint. Defaults to False.

    Returns:
        tuple[int, SpeciesLattice, ChargesLattice, np.ndarray]: (idx_anneal_layer, annealed_species_1L, annealed_charges_1L, energy_trajectory).

    Raises:
        ValueError: If graph_energy=True but n_steps < 100.
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
    Calculate the current temperature for simulated annealing at a given step.

    Implements a temperature schedule: flat at hot temperature for first 10%, linear ramp down
    to cold temperature over next 80%, then flat at cold temperature for final 10%.

    Args:
        n_steps (int): Total number of annealing steps.
        curr_step (int): Current step number (0 to n_steps).
        sim_temp_cold (float): Target (minimum) temperature.
        sim_temp_hot (float): Initial (maximum) temperature.

    Returns:
        float: Temperature to use for the current step.
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
