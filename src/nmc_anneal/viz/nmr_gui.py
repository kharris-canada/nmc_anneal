"""Interactive GUI for NMR spectrum analysis and fitting.

Provides a PyQt5-based graphical interface for interactive NMR spectrum fitting,
allowing users to adjust lineshape parameters (linewidth, Lorentzian/Gaussian blend),
view region-specific spectra, and interactively refine spectral models through
rectangle-selector-based region selection.
"""

import sys

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QWidget,
)

from nmc_anneal.viz.nmr_simpleplot import generate_spectrum

# ============================================================
# ---------------------  MAIN WINDOW  -------------------------
# ============================================================


class PeakFitGUI(QWidget):
    # -------- Default values ---------
    gauss_perc = 50
    gauss_perc_delta = 5
    fwhm = 5
    fwhm_delta = 0.25
    fwhm_lin = 0
    fwhm_lin_delta = 0.0025
    npoints = 4096
    npoints_delta = 2048
    xmin = -300
    xmax = 1200
    x_delta = 50
    v_scale = 1
    v_scale_delta = 0.25
    v_offset = 0
    v_offset_delta = 0.05

    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.init_ui()
        self.update_plot()

    """
    Initialize the interactive peak fitting GUI.

    Sets up the UI with control panels for spectrum parameters and a matplotlib canvas
    for visualization.

    Args:
        datasets (dict[str, tuple]): Dictionary mapping dataset names to (shifts, intensities) tuples.
    """

    # ---------------- UI ----------------

    def init_ui(self):
        """
        Construct the user interface for the peak fitting tool.

        Creates parameter adjustment widgets (Gaussian %, FWHM, resolution, zoom controls)
        and a matplotlib canvas displaying the NMR spectrum.
        """
        main_layout = QHBoxLayout(self)

        grid = QGridLayout()
        row = 0

        # ---------- column headers ----------
        grid.addWidget(QLabel("Name"), row, 0)
        grid.addWidget(QLabel("Parameter"), row, 1)
        grid.addWidget(QLabel("Step Size"), row, 2)
        row += 1

        # ---------- helpers ----------
        def set_width_digits(widget, digits):
            fm = widget.fontMetrics()
            px = fm.averageCharWidth() * digits + 24
            widget.setFixedWidth(px)

        def make_double_pair(value, step, decimals=3, minv=-1e9, maxv=1e9):
            box = QDoubleSpinBox()
            box.setDecimals(decimals)
            box.setRange(minv, maxv)
            box.setSingleStep(step)
            box.setValue(value)
            box.valueChanged.connect(self.update_plot)
            set_width_digits(box, 10)

            step_box = QDoubleSpinBox()
            step_box.setDecimals(decimals)
            step_box.setRange(1e-12, 1e9)
            step_box.setSingleStep(step)
            step_box.setValue(step)
            set_width_digits(step_box, 5)

            def update_step(val):
                box.setSingleStep(val)

            step_box.valueChanged.connect(update_step)
            return box, step_box

        def make_int_pair(value, step):
            box = QSpinBox()
            box.setRange(1, 1_000_000_000)
            box.setSingleStep(step)
            box.setValue(value)
            box.valueChanged.connect(self.update_plot)
            set_width_digits(box, 10)

            step_box = QSpinBox()
            step_box.setRange(1, 1_000_000_000)
            step_box.setSingleStep(step)
            step_box.setValue(step)
            set_width_digits(step_box, 5)

            def update_step(val):
                box.setSingleStep(val)

            step_box.valueChanged.connect(update_step)
            return box, step_box

        # ---------- parameters ----------
        def add_row(label, box, step_box):
            nonlocal row
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(box, row, 1)
            grid.addWidget(step_box, row, 2)
            row += 1

        self.gauss_box, self.gauss_step = make_double_pair(
            self.gauss_perc, self.gauss_perc_delta, 0
        )
        add_row("Percent Gaussian", self.gauss_box, self.gauss_step)

        self.fwhm0_box, self.fwhm0_step = make_double_pair(
            self.fwhm, self.fwhm_delta, 3
        )
        add_row("FWHM @ 0 ppm", self.fwhm0_box, self.fwhm0_step)

        self.fwhm_lin_box, self.fwhm_lin_step = make_double_pair(
            self.fwhm_lin, self.fwhm_lin_delta, 4
        )
        add_row("FWHM Linear", self.fwhm_lin_box, self.fwhm_lin_step)

        self.npoints_box, self.npoints_step = make_int_pair(
            self.npoints, self.npoints_delta
        )
        add_row("Resolution", self.npoints_box, self.npoints_step)

        self.xmin_box, self.xmin_step = make_double_pair(
            self.xmin, self.x_delta, 0, -1000, 1e9
        )
        add_row("xmin", self.xmin_box, self.xmin_step)

        self.xmax_box, self.xmax_step = make_double_pair(self.xmax, self.x_delta, 0)
        add_row("xmax", self.xmax_box, self.xmax_step)

        self.scale_box, self.scale_step = make_double_pair(
            self.v_scale, self.v_scale_delta, 3
        )
        add_row("Vertical Scale", self.scale_box, self.scale_step)

        self.offset_box, self.offset_step = make_double_pair(
            self.v_offset, self.v_offset_delta, 3
        )
        add_row("Vertical Offset", self.offset_box, self.offset_step)

        # ---------- reset button ----------
        reset_btn = QPushButton("Reset Zoom")
        reset_btn.clicked.connect(self.reset_zoom)
        grid.addWidget(reset_btn, row, 0, 1, 3)
        row += 1

        # ---- lock control panel width ----
        # ---- lock control panel size and pin to top ----
        control_panel = QWidget()
        control_panel.setLayout(grid)

        control_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        control_panel.setFixedWidth(control_panel.sizeHint().width())
        control_panel.setFixedHeight(control_panel.sizeHint().height())

        # Keep grid aligned to top
        grid.setAlignment(Qt.AlignTop)

        main_layout.addWidget(control_panel)

        # ---- Plot ----
        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim([self.xmax, self.xmin])

        names = list(self.datasets.keys())
        if len(names) > 1:
            comparison_xy = self.datasets[names[1]]
            max_y_experiment = np.max(comparison_xy[1])
        else:
            max_y_experiment = 1
        self.ax.set_ylim([0, max_y_experiment])

        self.zoom_selector = RectangleSelector(
            self.ax,
            self.on_zoom_select,
            useblit=True,
            button=[1],
            minspanx=1e-6,
            minspany=1e-6,
            spancoords="data",
            interactive=False,
        )

        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)

    # ----- Save Zoom state ----

    def _get_current_view(self):
        """
        Get current plot axis limits.

        Returns:
            tuple: (xlim, ylim) where each is a tuple of (min, max) for that axis.
        """
        return self.ax.get_xlim(), self.ax.get_ylim()

    def _restore_view(self, xlim, ylim):
        """
        Restore previous plot axis limits.

        Args:
            xlim (tuple): (xmin, xmax) limits for x-axis.
            ylim (tuple): (ymin, ymax) limits for y-axis.
        """
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

    # ---------------- Plot ----------------
    def update_plot(self):
        """
        Regenerate and redraw the NMR spectrum plot.

        Recalculates the spectrum using current parameter values, applies vertical scaling
        and offset, and redisplays with preserved zoom level.
        """
        xlim, ylim = self._get_current_view()
        names = list(self.datasets.keys())
        shifts, intensities = self.datasets[names[0]]
        comparison_xy = self.datasets[names[1]] if len(names) > 1 else None

        x, y = generate_spectrum(
            shifts,
            intensities,
            self.gauss_box.value(),
            self.fwhm0_box.value(),
            self.fwhm_lin_box.value(),
            self.npoints_box.value(),
            self.xmin_box.value(),
            self.xmax_box.value(),
        )

        y = y * self.scale_box.value() + self.offset_box.value()

        self.ax.clear()
        self.ax.plot(x, y, label="Simulation")

        if comparison_xy is not None:
            self.ax.plot(*comparison_xy, label="Experiment", alpha=0.7)

        self.ax.legend()

        # Reverse X axis
        self.ax.set_xlim(self.ax.get_xlim()[::-1])
        self._restore_view(xlim, ylim)
        self.canvas.draw_idle()

    # ---------------- Zoom ----------------
    def on_zoom_select(self, eclick, erelease):
        """
        Handle rectangular zoom selection on the plot.

        Args:
            eclick (matplotlib.backend_bases.MouseEvent): Mouse click event at corner 1.
            erelease (matplotlib.backend_bases.MouseEvent): Mouse release event at corner 2.
        """
        if eclick.xdata is None or erelease.xdata is None:
            return

        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        y1, y2 = sorted([eclick.ydata, erelease.ydata])

        self.ax.set_xlim(x2, x1)  # keep reversed axis
        self.ax.set_ylim(y1, y2)
        self.canvas.draw_idle()

    def on_mouse_press(self, event):
        """
        Handle mouse press events (double-click resets zoom).

        Args:
            event (matplotlib.backend_bases.MouseEvent): Mouse event data.
        """
        if event.dblclick:
            self.reset_zoom()

    def reset_zoom(self):
        """
        Reset the plot axes to show full spectrum range.

        Autoscales both axes to fit all data and respects the parameter bounds.
        """
        self.ax.autoscale()
        curr_xmax = np.maximum(self.xmax, self.xmax_box.value())
        curr_xmin = np.minimum(self.xmin, self.xmin_box.value())
        self.ax.set_xlim([curr_xmax, curr_xmin])
        self.canvas.draw_idle()


# does a quick rough scaling of experimental dataset get close to first simulation
def _scale_data(datasets):
    """
    Automatically scale experimental data to roughly match simulation intensities.

    Computes vertical scaling factor from the simulation and applies it to experimental data
    for better visual comparison.

    Args:
        datasets (dict[str, tuple]): Mutable dictionary of datasets. Modifies the experimental dataset in-place.
    """
    dataset_names = list(datasets.keys())
    data_to_model = datasets[dataset_names[0]]
    num_of_sites = sum(data_to_model[1])

    if len(dataset_names) > 1:
        comparison_xy = datasets[dataset_names[1]]
        x_c, y_c = comparison_xy
        init_v_scale = num_of_sites / (np.max(y_c) + 1e-12)
        init_v_offset = -np.min(y_c) * init_v_scale

    tempxy = [comparison_xy[0], comparison_xy[1] * init_v_scale + init_v_offset]
    datasets[dataset_names[1]] = tempxy


# ============================================================
# -----------------------  RUN APP  ---------------------------
# ============================================================


def run_peak_gui(datasets):
    app = QApplication(sys.argv)
    _scale_data(datasets)
    win = PeakFitGUI(datasets)
    win.setWindowTitle("Interactive Peak Fitting")
    win.resize(900, 500)
    win.show()
    sys.exit(app.exec_())
