"""Generate and plot NMR spectra from lithium shift distributions.

Provides functions to synthesize NMR spectra by convolving discrete lithium shifts
with line shape profiles (pseudo-Voigt), supporting both experimental spectral
generation and interactive GUI-based spectrum fitting.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _pseudo_voigt(x, x0, fwhm, eta):
    """
    Calculate the pseudo-Voigt lineshape at given x-coordinates.

    Blends Gaussian and Lorentzian profiles: (eta * Gaussian) + ((1-eta) * Lorentzian).

    Args:
        x (np.ndarray): X-coordinates where lineshape is evaluated.
        x0 (float): Center position of the peak.
        fwhm (float): Full-width at half-maximum of the peak.
        eta (float): Blending parameter (0=pure Lorentzian, 1=pure Gaussian).

    Returns:
        np.ndarray: Lineshape values at the given x-coordinates.
    """

    if fwhm <= 0:
        return np.zeros_like(x)

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gaussian = np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    gamma = fwhm / 2
    lorentz = 1 / (1 + ((x - x0) / gamma) ** 2)

    return eta * gaussian + (1 - eta) * lorentz


def generate_spectrum(
    shifts: np.ndarray,
    intensities: np.ndarray,
    percent_gaussian: float,
    fwhm_at_zero: float,
    fwhm_linear_scale: float,
    n_points: int,
    xmin: float,
    xmax: float,
):
    """
    Generate full 7Li MAS-NMR spectrum from a list of the chemical shifts in ppm and
    their relative intensities (site multiplicities)

    Args:
        shifts (np.ndarray): list of ppm shift of all sites
        intensities (np.ndarray): intensity (site multiplicity) of all peaks listed in "shifts"
        percent_gaussian (float): percent gaussian (vs. Lorenztian) of all peaks (0 to 100%)
        fwhm_at_zero (float): linewidth of all peaks (peaks not at 0 ppm may be increased with next parameter)
        fwhm_linear_scale (float): peaks shifted far from 0 ppm are probably subject to more ABMS broadening, so
                                    increase linewidth by this # of ppm per ppm the peak/site is centered at
        n_points (int): # of points in digital representation
        xmin (float): make simulation from this low value
        xmax (float): make simulation up to this high value

    Returns:
        tuple: x, y (ppm shift, intensity) of the 7Li NMR spectrum
    """
    eta = percent_gaussian / 100.0
    x = np.linspace(xmin, xmax, int(n_points))
    y = np.zeros_like(x)

    for s, amp in zip(shifts, intensities, strict=True):
        fwhm = fwhm_at_zero + fwhm_linear_scale * abs(s)
        y += amp * _pseudo_voigt(x, s, fwhm, eta)

    return x, y


def image_from_peaklist(
    data,
    percent_gaussian,
    fwhm_at_zero,
    fwhm_linear_scale,
    output_filename,
    n_points=4000,
    xmin=-300,
    xmax=700,
):
    """
    Generate a 7Li NMR spectrum and save it as an image file.

    Creates a pseudo-Voigt NMR spectrum from peak data and saves it as a high-quality
    image with inverted x-axis per NMR convention.

    Args:
        data (tuple[np.ndarray, np.ndarray]): (shifts, intensities) tuple.
        percent_gaussian (float): Gaussian fraction of lineshape (0-100).
        fwhm_at_zero (float): Linewidth (FWHM) at zero shift in ppm.
        fwhm_linear_scale (float): Additional broadening per ppm of offset.
        output_filename (str): Output filename (.png or .pdf).
        n_points (int): Resolution of generated spectrum. Defaults to 4000.
        xmin (float): Lower bound in ppm. Defaults to -300.
        xmax (float): Upper bound in ppm. Defaults to 700.

    Returns:
        tuple[np.ndarray, np.ndarray]: (x_ppm, intensity) arrays of the generated spectrum.

    Raises:
        ValueError: If shifts and intensities have different lengths or output filename has wrong extension.
    """

    shifts, intensities = data
    shifts = np.asarray(shifts, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    if len(shifts) != len(intensities):
        raise ValueError("shifts and intensities must have same length")

    x, y = generate_spectrum(
        data[0],
        data[1],
        percent_gaussian,
        fwhm_at_zero,
        fwhm_linear_scale,
        n_points,
        xmin,
        xmax,
    )

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y)
    ax.xaxis.set_inverted(
        True
    )  # Convention in NMR is to plot decreasing ppm to the right (backwards of normal graphs)
    ax.set_xlabel("Chemical Shift (ppm)")
    ax.set_title("Predicted 7Li MAS NMR Spectrum")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.yticks([])  # Hide y-axis ticks since they are arbitrary units

    plt.tight_layout()

    # Save
    ext = Path(output_filename).suffix.lower()
    if ext not in [".png", ".pdf"]:
        raise ValueError("Output filename must end with .png or .pdf")

    plt.savefig(output_filename, dpi=300)
    plt.close(fig)

    return x, y
