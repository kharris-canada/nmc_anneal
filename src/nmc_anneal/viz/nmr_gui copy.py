import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from nmc_anneal.viz.nmr_simpleplot import generate_spectrum


# ============================================================
# ---------------------  MAIN WINDOW  -------------------------
# ============================================================


class PeakFitGUI(QWidget):

    # -------- Default values ---------
    gauss_perc = 50
    gauss_perc_delta = 5
    fwhm = 5
    fwhm_delta = 0.5
    fwhm_lin = 0
    fwhm_lin_delta = 0.005
    npoints = 4096
    npoints_delta = 2048
    xmin = -300
    xmax = 1200
    x_delta = 50
    v_scale = 1
    v_scale_delta = 2
    v_offset = 0
    v_offset_delta = 0.05

    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.init_ui()
        self.update_plot()

    # ---------------- UI ----------------
    def init_ui(self):

        main_layout = QHBoxLayout(self)

        # ---- Controls ----
        form = QFormLayout()

        def spin_double(value, step, dec=3):
            box = QDoubleSpinBox()
            box.setDecimals(dec)
            box.setSingleStep(step)
            box.setRange(-1e6, 1e6)
            box.setValue(value)
            box.valueChanged.connect(self.update_plot)
            return box

        def spin_int(value, step=100):
            box = QSpinBox()
            box.setRange(200, 200000)
            box.setSingleStep(step)
            box.setValue(value)
            box.valueChanged.connect(self.update_plot)
            return box

        self.gauss_box = spin_double(self.gauss_perc, self.gauss_perc_delta)
        self.fwhm0_box = spin_double(self.fwhm, self.fwhm_delta, 4)
        self.fwhm_lin_box = spin_double(self.fwhm_lin, self.fwhm_lin_delta, 4)
        self.npoints_box = spin_int(self.npoints)

        self.xmin_box = spin_double(self.xmin, self.x_delta)
        self.xmax_box = spin_double(self.xmax, self.x_delta)

        self.scale_box = spin_double(self.v_scale, self.v_scale_delta, 4)
        self.offset_box = spin_double(self.v_offset, self.v_offset_delta, 4)

        form.addRow("Percent Gaussian", self.gauss_box)
        form.addRow("FWHM @ 0", self.fwhm0_box)
        form.addRow("FWHM Linear", self.fwhm_lin_box)
        form.addRow("Resolution", self.npoints_box)
        form.addRow("xmin", self.xmin_box)
        form.addRow("xmax", self.xmax_box)
        form.addRow("Vertical Scale", self.scale_box)
        form.addRow("Vertical Offset", self.offset_box)

        reset_btn = QPushButton("Reset Zoom")
        reset_btn.clicked.connect(self.reset_zoom)
        form.addRow(reset_btn)

        main_layout.addLayout(form)

        # ---- Plot ----
        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim([self.xmax, self.xmin])
        names = list(self.datasets.keys())
        if len(names) > 1:
            comparison_xy = self.datasets[names[1]]
            max_y_experiment = np.max(comparison_xy[1])
        else:
            max_y_experiment = 1
        self.ax.set_ylim([0, max_y_experiment])

        # Rectangle zoom
        self.zoom_selector = RectangleSelector(
            self.ax,
            self.on_zoom_select,
            # drawtype="box",
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
        """Return current axis limits (used to preserve zoom)."""
        return self.ax.get_xlim(), self.ax.get_ylim()

    def _restore_view(self, xlim, ylim):
        """Restore axis limits after redraw."""
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

    # ---------------- Plot ----------------
    def update_plot(self):
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
        if eclick.xdata is None or erelease.xdata is None:
            return

        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        y1, y2 = sorted([eclick.ydata, erelease.ydata])

        self.ax.set_xlim(x2, x1)  # keep reversed axis
        self.ax.set_ylim(y1, y2)
        self.canvas.draw_idle()

    def on_mouse_press(self, event):
        if event.dblclick:
            self.reset_zoom()

    def reset_zoom(self):
        self.ax.autoscale()
        curr_xmax = np.maximum(self.xmax, self.xmax_box.value())
        curr_xmin = np.minimum(self.xmin, self.xmin_box.value())
        self.ax.set_xlim([curr_xmax, curr_xmin])
        self.canvas.draw_idle()


# does a quick rough scaling of experimental dataset get close to first simulation
def _scale_data(datasets):
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
