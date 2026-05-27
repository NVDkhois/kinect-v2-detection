"""
LossChart — widget vẽ loss curve realtime bằng matplotlib FigureCanvas.

Fallback hoàn toàn qua matplotlib (pyqtgraph không có trong venv).
`add_epoch()` phải gọi từ Qt main thread (signal queued connection từ
TrainingThread → safe).
"""

from __future__ import annotations

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget


class LossChart(QWidget):
    """
    Widget vẽ box_loss (xanh dương) và cls_loss (cam) theo epoch.

    Interface:
        add_epoch(epoch, metrics): append và redraw
        reset(): xoá data về blank state
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._epochs: list[int] = []
        self._box_loss: list[float] = []
        self._cls_loss: list[float] = []

        self._fig = Figure(figsize=(5, 2.8), dpi=90, tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._line_box, = self._ax.plot([], [], color="#4C72B0",
                                         linestyle="-",  linewidth=1.5,
                                         label="box_loss")
        self._line_cls, = self._ax.plot([], [], color="#DD8452",
                                         linestyle="--", linewidth=1.5,
                                         label="cls_loss")

        self._ax.set_xlabel("Epoch", fontsize=8)
        self._ax.set_ylabel("Loss", fontsize=8)
        self._ax.tick_params(labelsize=7)
        self._ax.legend(loc="upper right", fontsize=7)
        self._ax.grid(True, alpha=0.3, linewidth=0.5)
        self._ax.set_xlim(0, 1)
        self._ax.set_ylim(0, 1)
        self._fig.patch.set_facecolor("#f8f8f8")
        self._ax.set_facecolor("#ffffff")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    # ---------------------------------------------------------------- API
    def add_epoch(self, epoch: int, metrics: dict) -> None:
        """Append 1 data point và redraw. Gọi từ Qt main thread."""
        self._epochs.append(epoch)
        self._box_loss.append(float(metrics.get("box_loss", 0.0)))
        self._cls_loss.append(float(metrics.get("cls_loss", 0.0)))
        self._redraw()

    def reset(self) -> None:
        """Xoá tất cả data về blank state."""
        self._epochs.clear()
        self._box_loss.clear()
        self._cls_loss.clear()
        self._line_box.set_data([], [])
        self._line_cls.set_data([], [])
        self._ax.set_xlim(0, 1)
        self._ax.set_ylim(0, 1)
        self._canvas.draw_idle()

    # --------------------------------------------------------------- draw
    def _redraw(self) -> None:
        if not self._epochs:
            return
        xs = self._epochs
        self._line_box.set_data(xs, self._box_loss)
        self._line_cls.set_data(xs, self._cls_loss)

        # Auto-scale X
        self._ax.set_xlim(1, max(xs[-1], 2))

        # Auto-scale Y với chút margin
        all_y = self._box_loss + self._cls_loss
        if all_y:
            ymin, ymax = min(all_y), max(all_y)
            margin = (ymax - ymin) * 0.1 + 0.01
            self._ax.set_ylim(max(0, ymin - margin), ymax + margin)

        self._canvas.draw_idle()
