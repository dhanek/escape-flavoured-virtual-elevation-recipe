from PySide6.QtCore import Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QLabel, QSizePolicy

from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

class VEPlotLabel(QLabel):
    sizeChanged = Signal()

    def __init__(self, screen):
        super().__init__()
        self.setScaledContents(True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.updateGeometry()
        self.screen = screen
        self.setMinimumSize(self.size())
        self.screen.logicalDotsPerInchChanged.connect(self.on_dpi_changed)

    def save(self, plot_path):
        self.pixmap().save(plot_path, "PNG")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.sizeChanged.emit()

    def on_dpi_changed(self, dpi):
        self.sizeChanged.emit()

    def get_size_info(self):
        self.pixel_ratio = self.screen.devicePixelRatio()
        return {"logical_dpi": self.screen.logicalDotsPerInch(),
                "width": self.width(),
                "height": self.height(),
                "pixel_ratio": self.pixel_ratio}

    def set_fig(self, res):
        image = QImage(res["buf"].data, res["width"], res["height"], QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(image)
        pixmap.setDevicePixelRatio(self.pixel_ratio)
        self.setPixmap(pixmap)

class VEFigure:
    def __init__(self, size_info):
        logical_dpi = size_info["logical_dpi"]
        width = size_info["width"]
        height = size_info["height"]
        pixel_ratio = size_info["pixel_ratio"]

        fig_width_in = width / logical_dpi
        fig_height_in = height / logical_dpi

        effective_dpi = logical_dpi * pixel_ratio

        self.fig = Figure(figsize=(fig_width_in, fig_height_in), dpi=effective_dpi)
        self.canvas = FigureCanvas(self.fig)
        gs = GridSpec(2, 1, height_ratios=[3, 1], figure=self.fig)
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1])

    def get_fig_axes(self):
        return self.fig, self.ax1, self.ax2

    def draw(self):
        self.fig.tight_layout()
        self.canvas.draw()
        width, height = self.canvas.get_width_height()

        fig_buf = np.frombuffer(self.canvas.buffer_rgba(),
                                dtype=np.uint8).reshape((width, height, 4))
        return {"buf": fig_buf,
                "width": width,
                "height": height}
