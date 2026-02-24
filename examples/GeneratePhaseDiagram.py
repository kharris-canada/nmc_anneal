from pathlib import Path

import nmc_anneal as nmc
from nmc_anneal.analysis import get_phase_diagram


def main() -> None:
    # Load stoichiometry and simulation parameters from file
    config = nmc.parse_input_file(Path("examples/ex1_parameters.txt"))

    # Generate lattice of charges and equivalent lattice of atomic names with randomized positions
    whole_lattice_species, whole_lattice_charges = nmc.initialize_lattice(config)

    # Settle the structure into a very low energy (set with low temp. parameter in input.txt)
    nmc.anneal_3Dlattice(
        config,
        whole_lattice_charges,
        whole_lattice_species,
        anneal_type="Initialize TM",
        graph_energy=False,
    )

    get_phase_diagram(
        config,
        whole_lattice_charges,
        whole_lattice_species,
        output_filename="examples/phase_diagram.png",
        anneal_type="TM Convergence Check",
        n_steps_perT=1e4,
        sim_start_temp=0,
        sim_end_temp=3,
    )


if __name__ == "__main__":
    main()
