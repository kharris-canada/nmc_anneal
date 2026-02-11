from pathlib import Path
import numpy as np
import nmc_anneal as nmc


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

    # Remove 0.20 units of Li (from the empirical formula Li_a[Ni_x Mn_y Co_z O_2])
    # (and oxidize 1/2 that amount of Ni2+ atoms to Ni4+)
    nmc.delithiate(config, whole_lattice_charges, whole_lattice_species, 0.20)

    np.savez(
        "examples/example4_nmc_sim.npz",
        array1=whole_lattice_charges,
        array2=whole_lattice_species,
        config=config,
    )


if __name__ == "__main__":
    main()
