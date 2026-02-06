from __future__ import annotations
from pathlib import Path
from typing import Dict
from nmc_anneal import SimulationConfig


def parse_input_file(path: Path) -> SimulationConfig:
    """
    Parse a key = value input file and store result as SimulationConfig class.

    All lines beginning with '#' or empty lines are ignored.
    """
    raw_values = _read_key_value_file(path)
    config = _build_config(raw_values)
    _validate_config(config)
    return config


# Read in input text file and store values as key/value pairs in a dict
def _read_key_value_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    values: Dict[str, str] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            if "=" not in stripped:
                raise ValueError(
                    f"Invalid line {line_number}: '{line.strip()}' "
                    "(expected key = value)"
                )

            key, value = stripped.split("=", maxsplit=1)
            values[key.strip()] = value.strip()

    return values


# Convert the key/value dict pairs to proper types and store in SimulationConfig class
def _build_config(values: Dict[str, str]) -> SimulationConfig:
    try:
        return SimulationConfig(
            # Lattice
            width=int(values["width"]),
            n_layers=int(values["n_layers"]),
            # Li layer Composition
            li_fraction_li_layer=float(values["li_fraction_li_layer"]),
            mn_fraction_li_layer=float(values["mn_fraction_li_layer"]),
            ni2_fraction_li_layer=float(values["ni2_fraction_li_layer"]),
            ni3_fraction_li_layer=float(values["ni3_fraction_li_layer"]),
            vac_fraction_li_layer=float(values["vac_fraction_li_layer"]),
            # TM layer Composition
            li_fraction_tm_layer=float(values["li_fraction_tm_layer"]),
            mn_fraction_tm_layer=float(values["mn_fraction_tm_layer"]),
            ni2_fraction_tm_layer=float(values["ni2_fraction_tm_layer"]),
            ni3_fraction_tm_layer=float(values["ni3_fraction_tm_layer"]),
            co_fraction_tm_layer=float(values["co_fraction_tm_layer"]),
            vac_fraction_tm_layer=float(values["vac_fraction_tm_layer"]),
            # Lattice initialization annealing
            initialize_anneal_steps=int(float(values["initialize_anneal_steps"])),
            initialize_anneal_hot_temp=float(values["initialize_anneal_hot_temp"]),
            initialize_anneal_cold_temp=float(values["initialize_anneal_cold_temp"]),
            # Electrochemistry
            delithiation_steps=int(float(values["delithiation_steps"])),
            delithiation_fraction_to_remove=float(
                values["delithiation_fraction_to_remove"]
            ),
            oxidation_model=values["oxidation_model"],
            # Lattice mid-delithiation annealing
            mid_delithiation_anneal_steps=int(
                float(values["mid_delithiation_anneal_steps"])
            ),
            mid_delithiation_anneal_hot_temp=float(
                values["mid_delithiation_anneal_hot_temp"]
            ),
            mid_delithiation_anneal_cold_temp=float(
                values["mid_delithiation_anneal_cold_temp"]
            ),
            # Output / control
            output_file=Path(values["output_file"]),
            random_seed=_parse_optional_int(values.get("random_seed")),
        )
    except KeyError as exc:
        raise KeyError(f"Missing required configuration key: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid value in configuration: {exc}") from exc


# helper to make sure no error if optional keys aren't present
def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value)


###Section to validate system described in input file
def _validate_config(config: SimulationConfig) -> None:
    _validate_lattice(config)
    _validate_composition(config)
    _validate_electrochemistry(config)
    _validate_output(config)


# Make sure lattice physically possible
def _validate_lattice(config: SimulationConfig) -> None:
    if config.width <= 0:
        raise ValueError(
            "Lattice horizontal size is (width x width), so they must be positive integers"
        )
    if config.n_layers <= 0:
        raise ValueError("Need at least one layer")


# Make sure occupations physically possible (any amount of vacancies currently allowed in code, maybe not in real life)
def _validate_composition(config: SimulationConfig) -> None:

    li_layer_fractional_vars = [
        config.li_fraction_li_layer,
        config.mn_fraction_li_layer,
        config.ni2_fraction_li_layer,
        config.ni3_fraction_li_layer,
        config.vac_fraction_li_layer,
    ]
    tm_layer_fractional_vars = [
        config.li_fraction_tm_layer,
        config.mn_fraction_tm_layer,
        config.ni2_fraction_tm_layer,
        config.ni3_fraction_tm_layer,
        config.co_fraction_tm_layer,
        config.vac_fraction_tm_layer,
    ]

    if any(f < 0.0 or f > 1.0 for f in li_layer_fractional_vars):
        raise ValueError(
            "All fractional occupations in Li layer must be between 0 and 1"
        )

    if any(f < 0.0 or f > 1.0 for f in tm_layer_fractional_vars):
        raise ValueError(
            "All fractional occupations in TM layer must be between 0 and 1"
        )

    # accept tiny roundoff errors here for convenience, populate with Co3+ or Li or something when populating lattice
    if not (0.9999 <= sum(li_layer_fractional_vars) <= 1.0):
        raise ValueError(
            "sum of Li layer atom fractions must be closer to 1.0, explicitly include vacancies if present"
        )

    # accept tiny roundoff errors here for convenience, populate with Co3+ or Li or something when populating lattice
    if not (0.9999 <= sum(tm_layer_fractional_vars) <= 1.0):
        raise ValueError(
            "sum of TM layer atom fractions must be closer to 1.0, is, explicitly include vacancies if present"
        )

    # Check for charge balance
    total_li_charge = (
        config.li_fraction_li_layer * 1
        + config.mn_fraction_li_layer * 4
        + config.ni2_fraction_li_layer * 2
        + config.ni3_fraction_li_layer * 3
    )
    total_tm_charge = (
        config.li_fraction_tm_layer * 1
        + config.mn_fraction_tm_layer * 4
        + config.ni2_fraction_tm_layer * 2
        + config.ni3_fraction_tm_layer * 3
        + config.co_fraction_tm_layer * 3
    )
    if not (3.999 <= (total_li_charge + total_tm_charge) <= 4.001):
        raise ValueError(
            "Material not charge balanced. Metals must sum to +4 to equal the two -2 oxygen atoms."
        )


# Make sure delithiation procedure is sensible and defined
def _validate_electrochemistry(config: SimulationConfig) -> None:
    if config.delithiation_steps <= 0:
        raise ValueError("delithiation_steps must be a positive integer")

    if not (0.0 <= config.delithiation_fraction_to_remove <= 1.0):
        raise ValueError("delithiation_fraction_to_remove must be between 0 and 1.0")

    allowed_models = {
        "ni_2to4",
        "ni_2to3_3to4",
        "ni_2to4_co3to4",
        "ni_2to3_ni_3to4_co3to4",
    }
    if config.oxidation_model not in allowed_models:
        raise ValueError(f"oxidation_model must be one of {sorted(allowed_models)}")


def _validate_output(config: SimulationConfig) -> None:
    if config.random_seed is not None and config.random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")

    if config.output_file.suffix != ".npy":
        raise ValueError("output_file must have .npy extension")
