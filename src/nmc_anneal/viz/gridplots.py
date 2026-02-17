import matplotlib.pyplot as plt
import numpy as np

from nmc_anneal.core.config import SimulationConfig


def plot_energy_convergence_grid(
    config: SimulationConfig,
    anneal_type: str,
    trajectories: list,
    step_counts: list[int],
    final_avg_energies: list[float],
    outfile: str,
):
    """
    Plot energy convergence trajectories in a 3x3 grid with run parameters.

    Creates a 3x3 grid of subplots showing energy evolution for up to 8 annealing runs.
    The 9th panel displays key simulation parameters.

    Args:
        config (SimulationConfig): Configuration object with simulation parameters.
        anneal_type (str): Label for the type of annealing performed.
        trajectories (list[np.ndarray]): Energy trajectories (sampled at 1% intervals).
        step_counts (list[int]): Total Monte Carlo steps for each trajectory.
        final_avg_energies (list[float]): Average energy over final 5% for each trajectory.
        outfile (str): Path to output PNG or PDF file.
    """
    nrows = 3
    ncols = 3

    global_ymin = min(traj.min() for traj in trajectories)
    global_ymax = max(traj.max() for traj in trajectories)

    pad = 0.05 * (global_ymax - global_ymin)
    global_ymin -= pad
    global_ymax += pad

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(20, 10),
        constrained_layout=True,
    )

    axes = axes.flatten()

    info_ax = axes[-1]
    for i, ax in enumerate(axes):
        if ax is info_ax:
            ax.axis("off")
            continue

        if i >= len(trajectories):
            # Keep empty slots but hide ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(True)
            continue

        traj = trajectories[i]
        steps = step_counts[i]
        avg_E = final_avg_energies[i]

        x = np.linspace(0, 100, len(traj), endpoint=False)

        ax.plot(
            x,
            traj,
            linestyle="None",
            marker="o",
            markersize=3,
            color="black",
        )

        ax.set_xlim(0, 100)
        ax.set_xlabel("% of simulation completed")
        ylabel = r"$\mathbf{\langle}\mathbf{E}_\mathrm{oxy}\mathbf{\rangle}$"
        ax.set_ylabel(
            ylabel,
            fontsize=14,  # makes ⟨E⟩ large
            fontweight="bold",
            labelpad=8,
        )

        # Grid: exactly 5 lines in each direction
        ax.set_ylim(global_ymin, global_ymax)
        ax.set_xticks(np.linspace(0, 100, 5))
        ax.set_yticks(np.linspace(0, global_ymax, 5))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Annotation
        ax.text(
            0.97,
            0.95,
            f"Steps: {steps: 0.1e}\n" f"Avg last 5%: {avg_E:.4f}",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    run_info = {
        "Formula": config.stoich_string,
        "Simulation Box Width": config.width,
        "Simulation Box Number of Layers": config.n_layers * 2,
        "Anneal type": anneal_type,
        "Simulation Hot Start Temperature": config.curr_conv_check_hot_temp,
        "Simulation Cold Start Temperature": config.curr_conv_check_cold_temp,
    }

    info_ax.axis("off")

    text = "\n".join(f"{k}: {v}" for k, v in run_info.items())

    info_ax.text(
        0.05,
        0.95,
        text,
        transform=info_ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    fig.savefig(outfile, dpi=300)
    plt.close(fig)
