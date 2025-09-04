import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PySide6.QtCore import Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


class VEPlotLabel(QLabel):
    sizeChanged = Signal()

    def __init__(self, screen):
        super().__init__()
        self.setScaledContents(True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.updateGeometry()
        self.screen = screen
        self.pixel_ratio = screen.devicePixelRatio()
        self.setMinimumSize(self.size())
        self.screen.logicalDotsPerInchChanged.connect(self.on_dpi_changed)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.sizeChanged.emit()

    def on_dpi_changed(self, dpi):
        self.pixel_ratio = self.screen.devicePixelRatio()
        self.sizeChanged.emit()

    def get_size_info(self):
        self.pixel_ratio = self.screen.devicePixelRatio()
        return {
            "logical_dpi": self.screen.logicalDotsPerInch(),
            "width": self.width(),
            "height": self.height(),
            "pixel_ratio": self.pixel_ratio,
        }

    def set_fig(self, res):
        image = QImage(
            res["buf"].data, res["width"], res["height"], QImage.Format_RGBA8888
        )
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

        fig_buf = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            (width, height, 4)
        )
        return {"buf": fig_buf, "width": width, "height": height}


class VEPlotSaver:
    def __init__(self, worker, thread):
        self.worker = worker
        worker.moveToThread(thread)
        worker.resultReady.connect(self.on_ve_result_ready)

    def save(self, values, plot_path, width=3840, height=2160, dpi=200):
        self.plot_path = plot_path

        values["plot_size_info"] = {
            "logical_dpi": dpi,
            "width": width,
            "height": height,
            "pixel_ratio": 1.0,
        }
        self.worker.set_values(values)

    def on_ve_result_ready(self, res):
        fig_res = res["fig_res"]
        image = QImage(
            fig_res["buf"].data,
            fig_res["width"],
            fig_res["height"],
            QImage.Format_RGBA8888,
        )
        pixmap = QPixmap.fromImage(image)
        pixmap.save(self.plot_path, "PNG")
