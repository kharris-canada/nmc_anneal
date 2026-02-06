from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=False)
class SimulationConfig:
    # Lattice is tilted square of width^2 atoms, n_layers of TM (oxygen and li layers for periodicity implied)
    width: int
    n_layers: int

    # Li layer Composition as stoichiometry fractions between 0 and 1
    li_fraction_li_layer: float
    mn_fraction_li_layer: float
    ni2_fraction_li_layer: float
    ni3_fraction_li_layer: float
    vac_fraction_li_layer: float

    # TM layer Composition as stoichiometry fractions between 0 and 1
    li_fraction_tm_layer: float
    mn_fraction_tm_layer: float
    ni2_fraction_tm_layer: float
    ni3_fraction_tm_layer: float
    co_fraction_tm_layer: float
    vac_fraction_tm_layer: float

    # Lattice initialization annealing
    initialize_anneal_steps: float  # float for sci. notation, converted to int later
    initialize_anneal_hot_temp: float
    initialize_anneal_cold_temp: float

    # Electrochemistry steps and atomic numbers as stoichiometry fractions between 0 and 1
    delithiation_steps: float  # float for sci. notation, converted to int later
    delithiation_fraction_to_remove: float
    oxidation_model: str

    # Lattice mid-delithiation annealing
    mid_delithiation_anneal_steps: float
    mid_delithiation_anneal_hot_temp: float
    mid_delithiation_anneal_cold_temp: float

    # Output / control
    output_file: Path
    random_seed: int | None

    # Internally used variables, not for use in input text file:
    stoich_string = None
    curr_conv_check_n_steps = None
    curr_conv_check_max_n_steps = None
    curr_conv_check_hot_temp = None
    curr_conv_check_cold_temp = None
