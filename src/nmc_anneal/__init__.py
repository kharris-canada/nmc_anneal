"""NMC annealing simulation package.

Provides tools for initializing NMC (LiNi_xMn_yCo_zO2) crystal structures, simulating
lithium deintercalation via simulated annealing, calculating redox chemistry, and
analysing structural features through NMR spectroscopy simulations.
"""

__version__ = "0.6.1"

from .core.anneal_lattice import anneal_3Dlattice
from .core.charging_methods import delithiate
from .core.config import SimulationConfig
from .core.initialize_lattice import initialize_lattice
from .io.parser import parse_input_file

__all__ = [
    "SimulationConfig",
    "anneal_3Dlattice",
    "initialize_lattice",
    "parse_input_file",
    "delithiate",
]
