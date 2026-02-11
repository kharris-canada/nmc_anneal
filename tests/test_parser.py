from pathlib import Path

import pytest

import nmc_anneal as nmc
from nmc_anneal import SimulationConfig


def test_parse_valid_input(tmp_path: Path) -> None:
    input_text = """
    # Lattice
    width = 50
    n_layers = 5

    #Li layer Composition
    li_fraction_li_layer = 1.0
    mn_fraction_li_layer = 0
    ni2_fraction_li_layer = 0
    ni3_fraction_li_layer = 0
    vac_fraction_li_layer = 0

    # TM layer Composition
    li_fraction_tm_layer = 0
    mn_fraction_tm_layer = 0.3333
    ni2_fraction_tm_layer = 0.3333
    ni3_fraction_tm_layer = 0
    co_fraction_tm_layer = 0.3333
    vac_fraction_tm_layer = 0

    #Lattice initialization by annealing TM layer
    initialize_anneal_steps = 1e5
    initialize_anneal_hot_temp = 0.2
    initialize_anneal_cold_temp = 0.06

    # Electrochemistry
    delithiation_steps = 10
    oxidation_model = ni_2to4

    #Lattice mid-delithiation annealing
    mid_delithiation_anneal_steps=1e4
    mid_delithiation_anneal_hot_temp=0.0
    mid_delithiation_anneal_cold_temp=0.0

    # Output
    output_file = lattice_final.npy
    random_seed = 769
    """

    input_file = tmp_path / "input.txt"
    input_file.write_text(input_text)

    config = nmc.parse_input_file(input_file)

    assert config.width == 50
    assert config.n_layers == 5

    assert config.li_fraction_li_layer == 1.0
    assert config.mn_fraction_li_layer == 0
    assert config.ni2_fraction_li_layer == 0
    assert config.ni3_fraction_li_layer == 0
    assert config.vac_fraction_li_layer == 0

    assert config.li_fraction_tm_layer == 0
    assert config.mn_fraction_tm_layer == 0.3333
    assert config.ni2_fraction_tm_layer == 0.3333
    assert config.co_fraction_tm_layer == 0.3333
    assert config.ni3_fraction_tm_layer == 0
    assert config.vac_fraction_tm_layer == 0

    assert config.initialize_anneal_steps == 1e5
    assert config.initialize_anneal_hot_temp == 0.2
    assert config.initialize_anneal_cold_temp == 0.06

    assert config.delithiation_steps == 10
    assert config.oxidation_model == "ni_2to4"

    assert config.mid_delithiation_anneal_steps == 1e4
    assert config.mid_delithiation_anneal_hot_temp == 0.0
    assert config.mid_delithiation_anneal_cold_temp == 0.0

    assert config.output_file.name == "lattice_final.npy"
    assert config.random_seed == 769


def test_missing_required_key_raises_error(tmp_path: Path) -> None:
    input_text = """
    # Lattice
    width = 50
    n_layers = 5

    #Li layer Composition
    li_fraction_li_layer = 1.0
    mn_fraction_li_layer = 0
    ni2_fraction_li_layer = 0
    ni3_fraction_li_layer = 0
    vac_fraction_li_layer = 0

    # TM layer Composition
    li_fraction_tm_layer = 0
    mn_fraction_tm_layer = 0.3333
    ni2_fraction_tm_layer = 0.3333
    ni3_fraction_tm_layer = 0
    co_fraction_tm_layer = 0.3333

    #Lattice initialization by annealing TM layer
    initialize_anneal_steps = 1e5
    initialize_anneal_hot_temp = 0.2
    initialize_anneal_cold_temp = 0.06

    # Electrochemistry
    delithiation_steps = 10
    oxidation_model = ni_2to4

    #Lattice mid-delithiation annealing
    mid_delithiation_anneal_steps=1e4
    mid_delithiation_anneal_hot_temp=0.0
    mid_delithiation_anneal_cold_temp=0.0

    # Output
    output_file = lattice_final.npy
    random_seed = 769
    """

    input_file = tmp_path / "input.txt"
    input_file.write_text(input_text)

    with pytest.raises(KeyError):
        nmc.parse_input_file(input_file)


def test_random_seed_optional(tmp_path: Path) -> None:
    input_text = """
    # Lattice
    width = 50
    n_layers = 5

    #Li layer Composition
    li_fraction_li_layer = 1.0
    mn_fraction_li_layer = 0
    ni2_fraction_li_layer = 0
    ni3_fraction_li_layer = 0
    vac_fraction_li_layer = 0

    # TM layer Composition
    li_fraction_tm_layer = 0
    mn_fraction_tm_layer = 0.3333
    ni2_fraction_tm_layer = 0.3333
    ni3_fraction_tm_layer = 0
    co_fraction_tm_layer = 0.3333
    vac_fraction_tm_layer = 0

    #Lattice initialization by annealing TM layer
    initialize_anneal_steps = 1e5
    initialize_anneal_hot_temp = 0.2
    initialize_anneal_cold_temp = 0.06

    # Electrochemistry
    delithiation_steps = 10
    oxidation_model = ni_2to4

    #Lattice mid-delithiation annealing
    mid_delithiation_anneal_steps=1e4
    mid_delithiation_anneal_hot_temp=0.0
    mid_delithiation_anneal_cold_temp=0.0

    # Output
    output_file = lattice_final.npy
    """

    input_file = tmp_path / "input.txt"
    input_file.write_text(input_text)

    config = nmc.parse_input_file(input_file)

    assert config.random_seed is None


def test_invalid_tm_layer_contents(tmp_path: Path) -> None:
    input_text = """
    # Lattice
    width = 50
    n_layers = 5

    #Li layer Composition
    li_fraction_li_layer = 1.0
    mn_fraction_li_layer = 0
    ni2_fraction_li_layer = 0
    ni3_fraction_li_layer = 0
    vac_fraction_li_layer = 0

    # TM layer Composition
    li_fraction_tm_layer = 0
    mn_fraction_tm_layer = 0.4
    ni2_fraction_tm_layer = 0.4
    ni3_fraction_tm_layer = 0
    co_fraction_tm_layer = 0.1
    vac_fraction_tm_layer = 0

    #Lattice initialization by annealing TM layer
    initialize_anneal_steps = 1e5
    initialize_anneal_hot_temp = 0.2
    initialize_anneal_cold_temp = 0.06

    # Electrochemistry
    delithiation_steps = 10
    oxidation_model = ni_2to4

    #Lattice mid-delithiation annealing
    mid_delithiation_anneal_steps=1e4
    mid_delithiation_anneal_hot_temp=0.0
    mid_delithiation_anneal_cold_temp=0.0

    # Output
    output_file = lattice_final.npy
    random_seed = 769
    """

    input_file = tmp_path / "input.txt"
    input_file.write_text(input_text)

    with pytest.raises(ValueError):
        nmc.parse_input_file(input_file)


def test_invalid_li_layer_contents(tmp_path: Path) -> None:
    input_text = """
    # Lattice
    width = 50
    n_layers = 5

    #Li layer Composition
    li_fraction_li_layer = 1.0
    mn_fraction_li_layer = 0
    ni2_fraction_li_layer = 0.0
    ni3_fraction_li_layer = 0
    vac_fraction_li_layer = 0.1

    # TM layer Composition
    li_fraction_tm_layer = 0
    mn_fraction_tm_layer = 0.4
    ni2_fraction_tm_layer = 0.4
    ni3_fraction_tm_layer = 0
    co_fraction_tm_layer = 0.2
    vac_fraction_tm_layer = 0

    #Lattice initialization by annealing TM layer
    initialize_anneal_steps = 1e5
    initialize_anneal_hot_temp = 0.2
    initialize_anneal_cold_temp = 0.06

    # Electrochemistry
    delithiation_steps = 10
    oxidation_model = ni_2to4

    #Lattice mid-delithiation annealing
    mid_delithiation_anneal_steps=1e4
    mid_delithiation_anneal_hot_temp=0.0
    mid_delithiation_anneal_cold_temp=0.0

    # Output
    output_file = lattice_final.npy
    random_seed = 769
    """

    input_file = tmp_path / "input.txt"
    input_file.write_text(input_text)

    with pytest.raises(ValueError):
        nmc.parse_input_file(input_file)


def test_invalid_method(tmp_path: Path) -> None:
    input_text = """
    # Lattice
    width = 50
    n_layers = 5

    #Li layer Composition
    li_fraction_li_layer = 1.0
    mn_fraction_li_layer = 0
    ni2_fraction_li_layer = 0.0
    ni3_fraction_li_layer = 0
    vac_fraction_li_layer = 0.0

    # TM layer Composition
    li_fraction_tm_layer = 0
    mn_fraction_tm_layer = 0.4
    ni2_fraction_tm_layer = 0.4
    ni3_fraction_tm_layer = 0
    co_fraction_tm_layer = 0.2
    vac_fraction_tm_layer = 0

    #Lattice initialization by annealing TM layer
    initialize_anneal_steps = 1e5
    initialize_anneal_hot_temp = 0.2
    initialize_anneal_cold_temp = 0.06

    # Electrochemistry
    delithiation_steps = 10
    oxidation_model = co_2to4

    #Lattice mid-delithiation annealing
    mid_delithiation_anneal_steps=1e4
    mid_delithiation_anneal_hot_temp=0.0
    mid_delithiation_anneal_cold_temp=0.0

    # Output
    output_file = lattice_final.npy
    random_seed = 769
    """

    input_file = tmp_path / "input.txt"
    input_file.write_text(input_text)

    with pytest.raises(ValueError):
        nmc.parse_input_file(input_file)
