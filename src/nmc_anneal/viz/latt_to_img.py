"""Visualize 2D lattice structures as atoms on a hexagonal lattice.

Renders 2D lattice configurations as hexagonal lattice diagrams with atoms
represented by colored circles, where color encodes atomic charge values for
quick visual inspection of charge distributions.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def plot_2Dlattice(
    data_array: np.ndarray,
    filename: str = "2Dlattice.pdf",
    atom_radius: float = 0.45,
):
    """
    Visualize a 2D lattice of atomic charges as colored circles.

    Creates a hexagonal lattice visualization where each atom is represented by a colored
    circle. Colors are mapped based on the atomic charge values.

    Args:
        data_array (np.ndarray): 2D array of integer charges to visualize.
        filename (str): Output image filename (.pdf or other matplotlib format). Defaults to "2Dlattice.pdf".
        atom_radius (float): Radius of each atom circle in plot coordinates. Defaults to 0.45.
    """
    color_map = {
        -2: "firebrick",
        0: "black",
        1: "lightskyblue",
        2: "blue",
        3: "green",
        4: "darkviolet",
    }

    fig, ax = plt.subplots(figsize=(10, 10))

    rows, cols = data_array.shape

    for i in range(rows):
        for j in range(cols):
            atomic_charge = data_array[i, j]

            # Calculate position with 60-degree twist (0.5 shift per row)
            x = j + (i * 0.5)
            y = i * np.sqrt(3) / 2  # vertical spacing for hexagonal lattice

            # Get color from the color_map
            color = color_map.get(
                atomic_charge, "gray"
            )  # default to gray if not in map

            circle = Circle((x, y), atom_radius, color=color, linewidth=0.5)
            ax.add_patch(circle)

    ax.set_aspect("equal")
    ax.autoscale()
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
