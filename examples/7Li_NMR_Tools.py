from pathlib import Path
import numpy as np
import nmc_anneal as nmc

from nmc_anneal.analysis.struct2nmr import get_all_nmr_shifts
import nmc_anneal.viz.nmr_gui as NMRplot


def main() -> None:
    # Load stoichiometry and simulation parameters from file
    config = nmc.parse_input_file(Path("examples/ex2_parameters.txt"))

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

    # Generate a list of 7Li NMR peak positions and their intensity (site multiplicity)
    nmr_ppm_shifts = get_all_nmr_shifts(whole_lattice_charges, whole_lattice_species)

    # Load an experiment for comparison (formatted as two arrays: one for axis positions in ppm, the other signal intensity at those positions)
    data = np.load("examples/artifical_experimental_7LiNMR.npz")
    exp_ppm_axis = data["ppm_axis"]
    exp_intensities = data["intensities"]

    datasets = {
        "Simulation": (nmr_ppm_shifts[0], nmr_ppm_shifts[1]),
        "Experiment": (exp_ppm_axis, exp_intensities),
    }

    NMRplot.run_peak_gui(datasets)


if __name__ == "__main__":
    main()
