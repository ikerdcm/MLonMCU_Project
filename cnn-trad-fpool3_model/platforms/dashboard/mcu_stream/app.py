"""PyQt (PySide6) dashboard for monitoring 1–3 MCUs simultaneously.

v1 = monitor only: pick MCU + port + firmware mode per slot, Connect, and watch
a normalized live inference table with per-stream stats. Cross-MCU sync is by
host arrival timestamp (time.monotonic()).
"""
from __future__ import annotations

import time
from collections import deque, Counter
from functools import partial
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QProcess
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    QHeaderView, QDialogButtonBox, QPlainTextEdit,
)

import pyqtgraph as pg

from .protocol import PROFILES, InferenceRecord, label_in_set
from .reader import SerialReader, list_serial_ports, detect_ports
from .flasher import (flash_command, flash_available, FLASH_MODES, REPORT_MODES,
                      MODELS, MODEL_LABEL_SET)

# Selectable plot sources: label -> (record attribute, only-on-inference?)
PLOT_SOURCES = {
    "Mic level": ("level", False),
    "Confidence %": ("confidence", True),
    "Infer µs": ("infer_us", True),
}


class RollingPlot(pg.PlotWidget):
    """Fixed-height rolling time-series plot (keeps the last `window_s` seconds)."""

    def __init__(self, window_s: float = 12.0):
        super().__init__()
        self.window_s = window_s
        self.setFixedHeight(130)
        self.setMouseEnabled(x=False, y=False)
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setLabel("bottom", "t (s)")
        self.curve = self.plot(pen=pg.mkPen("#2ecc40", width=1))
        # Adaptive voice-trigger threshold (dotted), overlaid on the Mic level plot.
        self.curve_thr = self.plot(
            pen=pg.mkPen("#e67e22", width=1, style=Qt.PenStyle.DashLine))
        self.t: deque[float] = deque()
        self.v: deque[float] = deque()
        self.vt: deque[float] = deque()   # threshold aligned with self.t (NaN = none)
        self.t0: float | None = None

    def add(self, t_host: float, value: float, thr: float | None = None) -> None:
        if self.t0 is None:
            self.t0 = t_host
        t = t_host - self.t0
        self.t.append(t)
        self.v.append(value)
        self.vt.append(float(thr) if thr is not None else float("nan"))
        cutoff = t - self.window_s
        while self.t and self.t[0] < cutoff:
            self.t.popleft()
            self.v.popleft()
            self.vt.popleft()
        self.curve.setData(list(self.t), list(self.v))
        # connect="finite" leaves gaps where thr is NaN (e.g. non-level sources).
        self.curve_thr.setData(list(self.t), list(self.vt), connect="finite")

    def clear_data(self) -> None:
        self.t.clear()
        self.v.clear()
        self.vt.clear()
        self.t0 = None
        self.curve.setData([], [])
        self.curve_thr.setData([], [])

MAX_ROWS = 300              # cap table rows per panel
RATE_WINDOW_S = 5.0         # window for lines/s and inf/s


class McuPanel(QGroupBox):
    """One MCU slot: connection controls + live normalized inference table."""

    def __init__(self, slot_idx: int, on_flash=None, parent=None):
        super().__init__(f"MCU slot {slot_idx + 1}", parent)
        self.on_flash = on_flash
        self.reader: SerialReader | None = None
        self._t0_wall: float | None = None
        self._line_times: deque[float] = deque()
        self._inf_times: deque[float] = deque()
        self._inf_count = 0
        self.hidden_labels: set[str] = set()   # classes hidden by the "report" selection
        self._build_ui()

    def mcu_key(self) -> str:
        return self.cb_mcu.currentData()

    def model_token(self) -> str:
        return MODELS.get(self.mcu_key(), {}).get(self.cb_model.currentText(), "dscnn")

    # ── UI ──────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.cb_mcu = QComboBox()
        for key, prof in PROFILES.items():
            self.cb_mcu.addItem(prof.name, key)
        self.cb_mcu.currentIndexChanged.connect(self._on_mcu_changed)

        self.cb_mode = QComboBox()
        self.cb_model = QComboBox()   # which model/firmware to flash for this MCU
        self.cb_model.setToolTip("Model/firmware to build & flash for this MCU")
        self.cb_port = QComboBox()
        self.cb_port.setMinimumWidth(180)
        self.btn_refresh = QPushButton("⟳")
        self.btn_refresh.setFixedWidth(30)
        self.btn_refresh.setToolTip("Refresh serial ports")
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self._toggle_connect)
        self.btn_flash = QPushButton("Flash")
        self.btn_flash.setToolTip("Build + flash this MCU in the window's flash mode")
        self.btn_flash.clicked.connect(
            lambda: self.on_flash and self.on_flash(self.mcu_key()))

        # Row 1: MCU + Mode
        controls.addWidget(QLabel("MCU:"))
        controls.addWidget(self.cb_mcu)
        controls.addWidget(QLabel("Mode:"))
        controls.addWidget(self.cb_mode)
        controls.addStretch(1)
        root.addLayout(controls)

        # Row 2: Model + Port (kept narrow so the panel doesn't grow too wide)
        controls2 = QHBoxLayout()
        controls2.addWidget(QLabel("Model:"))
        controls2.addWidget(self.cb_model)
        controls2.addWidget(QLabel("Port:"))
        controls2.addWidget(self.cb_port)
        controls2.addWidget(self.btn_refresh)
        controls2.addWidget(self.btn_connect)
        controls2.addWidget(self.btn_flash)
        controls2.addStretch(1)
        root.addLayout(controls2)

        self.lbl_status = QLabel("idle")
        self.lbl_status.setStyleSheet("color: gray;")
        self.lbl_stats = QLabel("—")
        stats_row = QHBoxLayout()
        stats_row.addWidget(self.lbl_status, 1)
        stats_row.addWidget(self.lbl_stats, 0)
        root.addLayout(stats_row)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["t_host (s)", "run", "label", "conf %", "infer µs", "cycles"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        root.addWidget(self.table, 1)

        # Live plot + source selector
        plot_row = QHBoxLayout()
        plot_row.addWidget(QLabel("Plot:"))
        self.cb_plot = QComboBox()
        self.cb_plot.addItems(list(PLOT_SOURCES.keys()))
        self.cb_plot.currentIndexChanged.connect(lambda: self.plot.clear_data())
        plot_row.addWidget(self.cb_plot)
        plot_row.addStretch(1)
        root.addLayout(plot_row)
        self.plot = RollingPlot()
        root.addWidget(self.plot)

        self._on_mcu_changed()
        self.refresh_ports()

    def _on_mcu_changed(self) -> None:
        prof = PROFILES[self.cb_mcu.currentData()]
        self.cb_mode.clear()
        self.cb_mode.addItems(prof.modes)
        self.cb_model.clear()
        self.cb_model.addItems(list(MODELS.get(self.cb_mcu.currentData(), {}).keys()))
        self._autoselect_port()

    def refresh_ports(self) -> None:
        current = self.cb_port.currentText()
        self.cb_port.clear()
        ports = list_serial_ports()
        self.cb_port.addItems(ports)
        if not ports:
            self.cb_port.addItem("(no ports found)")
        elif current in ports:
            self.cb_port.setCurrentText(current)
        self._autoselect_port()

    def _autoselect_port(self) -> None:
        """Pick this MCU's port automatically (matched by USB vendor id)."""
        port = detect_ports().get(self.mcu_key())
        if port:
            idx = self.cb_port.findText(port)
            if idx >= 0:
                self.cb_port.setCurrentIndex(idx)

    # ── connection ──────────────────────────────────────────────────────
    def _toggle_connect(self) -> None:
        if self.reader is None:
            self._connect()
        else:
            self._disconnect()

    def _connect(self) -> None:
        port = self.cb_port.currentText()
        if not port or port.startswith("("):
            self.lbl_status.setText("no port selected")
            return
        mcu = self.cb_mcu.currentData()
        mode = self.cb_mode.currentText()
        baud = PROFILES[mcu].default_baud

        self._reset_stats()
        self.reader = SerialReader(mcu, mode, port, baud)
        self.reader.record.connect(self._on_record)
        self.reader.raw.connect(self._on_raw)
        self.reader.status.connect(self._on_status)
        self.reader.start()

        self.btn_connect.setText("Disconnect")
        self._set_controls_enabled(False)

    def _disconnect(self) -> None:
        if self.reader is not None:
            self.reader.stop()
            self.reader.wait(2000)
            self.reader = None
        self.btn_connect.setText("Connect")
        self.lbl_status.setText("idle")
        self.lbl_status.setStyleSheet("color: gray;")
        self._set_controls_enabled(True)

    def _set_controls_enabled(self, on: bool) -> None:
        for w in (self.cb_mcu, self.cb_mode, self.cb_port, self.btn_refresh):
            w.setEnabled(on)

    # ── slots ───────────────────────────────────────────────────────────
    def _on_status(self, text: str) -> None:
        self.lbl_status.setText(text)
        color = "green" if text.startswith("connected") else (
            "orange" if "retry" in text or "disconnected" in text else "gray")
        self.lbl_status.setStyleSheet(f"color: {color};")

    def _on_raw(self, line: str) -> None:
        self._line_times.append(time.monotonic())

    def _on_record(self, rec: InferenceRecord) -> None:
        # Feed the live plot according to the selected source.
        attr, inf_only = PLOT_SOURCES[self.cb_plot.currentText()]
        if (not inf_only) or rec.is_inference:
            val = getattr(rec, attr, None)
            if val is not None:
                # On the Mic level plot, overlay the adaptive trigger threshold.
                thr = rec.thr if attr == "level" else None
                self.plot.add(rec.t_host, float(val), thr)

        if not rec.is_inference:
            return
        lbl = (label_in_set(rec.pred_idx, MODEL_LABEL_SET.get(self.model_token(), "dscnn"))
               if rec.pred_idx is not None else rec.label)
        if lbl in self.hidden_labels:      # honor the "report" selection in the view
            return
        if self._t0_wall is None:
            self._t0_wall = rec.t_host
        self._inf_times.append(rec.t_host)
        self._inf_count += 1

        t_rel = rec.t_host - self._t0_wall
        row = self.table.rowCount()
        if row >= MAX_ROWS:
            self.table.removeRow(0)
            row = self.table.rowCount()
        self.table.insertRow(row)
        vals = [
            f"{t_rel:8.3f}",
            "" if rec.run is None else str(rec.run),
            lbl or "?",
            "" if rec.confidence is None else f"{rec.confidence:.0f}",
            "" if rec.infer_us is None else str(rec.infer_us),
            "" if rec.cycles is None else str(rec.cycles),
        ]
        for col, v in enumerate(vals):
            item = QTableWidgetItem(v)
            if col in (0, 1, 3, 4, 5):
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(row, col, item)
        # highlight confident detections (Coral live) / non-unknown
        if rec.extra.get("confident") or (lbl not in (None, "unknown", "silence")):
            for col in range(self.table.columnCount()):
                it = self.table.item(row, col)
                if it:
                    it.setBackground(QColor(40, 90, 40))
        self.table.scrollToBottom()

    # ── stats ───────────────────────────────────────────────────────────
    def _reset_stats(self) -> None:
        self._t0_wall = None
        self._line_times.clear()
        self._inf_times.clear()
        self._inf_count = 0
        self.table.setRowCount(0)
        self.plot.clear_data()

    def tick_stats(self) -> None:
        now = time.monotonic()
        cutoff = now - RATE_WINDOW_S
        while self._line_times and self._line_times[0] < cutoff:
            self._line_times.popleft()
        while self._inf_times and self._inf_times[0] < cutoff:
            self._inf_times.popleft()
        if self.reader is None:
            self.lbl_stats.setText("—")
            return
        lps = len(self._line_times) / RATE_WINDOW_S
        ips = len(self._inf_times) / RATE_WINDOW_S
        self.lbl_stats.setText(
            f"inferences: {self._inf_count}   |   {ips:.1f} inf/s   |   "
            f"{lps:.1f} lines/s")

    def shutdown(self) -> None:
        self._disconnect()


# Ground-truth words offered in the eval mode (the keywords shared by both families).
PROMPT_WORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]


class EvalWindow(QWidget):
    """Guided live confusion matrix: pick a word, listen 5 s, tally ONE majority
    prediction per MCU into a per-MCU confusion matrix that updates live."""

    def __init__(self, panels):
        super().__init__()
        self.setWindowTitle("Live confusion matrix")
        self.panels = [p for p in panels if p.reader is not None]
        self.resize(440 * max(1, len(self.panels)), 540)
        self._collecting = False
        self._buf: dict[int, list[str]] = {}
        self.conf: list[dict] = [{} for _ in self.panels]   # conf[i][true][col] = count

        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Word:"))
        self.cb_word = QComboBox(); self.cb_word.addItems(PROMPT_WORDS)
        bar.addWidget(self.cb_word)
        self.btn_start = QPushButton("Listen 5 s")
        self.btn_start.clicked.connect(self.start_round)
        bar.addWidget(self.btn_start)
        self.lbl_count = QLabel(""); bar.addWidget(self.lbl_count)
        bar.addStretch(1)
        b_reset = QPushButton("Reset"); b_reset.clicked.connect(self.reset); bar.addWidget(b_reset)
        b_csv = QPushButton("Export CSV"); b_csv.clicked.connect(self.export_csv); bar.addWidget(b_csv)
        root.addLayout(bar)

        if not self.panels:
            root.addWidget(QLabel("Connect at least one MCU first, then reopen Evaluate."))
            return

        mats = QHBoxLayout()
        self.tables: list[QTableWidget] = []
        self.cols: list[list[str]] = []
        self.acc_lbls: list[QLabel] = []
        for p in self.panels:
            box = QVBoxLayout()
            box.addWidget(QLabel(f"<b>{p.cb_mcu.currentText()}</b> · {p.cb_model.currentText()}"))
            cols = PROMPT_WORDS + ["unknown", "other", "(none)"]
            self.cols.append(cols)
            t = QTableWidget(len(PROMPT_WORDS), len(cols))
            t.setHorizontalHeaderLabels(cols)
            t.setVerticalHeaderLabels(PROMPT_WORDS)
            t.setEditTriggers(QTableWidget.NoEditTriggers)
            self.tables.append(t)
            box.addWidget(t)
            acc = QLabel("accuracy: —"); self.acc_lbls.append(acc); box.addWidget(acc)
            mats.addLayout(box)
        root.addLayout(mats, 1)

        for i, p in enumerate(self.panels):
            p.reader.record.connect(partial(self._collect, i))
        self._refresh()

    def _collect(self, i: int, rec: InferenceRecord) -> None:
        if self._collecting and rec.is_inference:
            lbl = label_in_set(rec.pred_idx,
                               MODEL_LABEL_SET.get(self.panels[i].model_token(), "dscnn"))
            if lbl:
                self._buf[i].append(lbl)

    def start_round(self) -> None:
        if self._collecting or not self.panels:
            return
        self._buf = {i: [] for i in range(len(self.panels))}
        self._collecting = True
        self._remaining = 5
        self.btn_start.setEnabled(False)
        self.lbl_count.setText(f"listening… 5s — say '{self.cb_word.currentText()}'")
        self._tick = QTimer(self); self._tick.timeout.connect(self._countdown); self._tick.start(1000)

    def _countdown(self) -> None:
        self._remaining -= 1
        if self._remaining > 0:
            self.lbl_count.setText(f"listening… {self._remaining}s — say '{self.cb_word.currentText()}'")
        else:
            self._tick.stop()
            self._finish()

    def _finish(self) -> None:
        self._collecting = False
        true_w = self.cb_word.currentText()
        for i in range(len(self.panels)):
            preds = self._buf.get(i, [])
            maj = Counter(preds).most_common(1)[0][0] if preds else "(none)"
            if maj in PROMPT_WORDS:
                col = maj
            elif maj in ("unknown", "silence"):
                col = "unknown"
            elif maj == "(none)":
                col = "(none)"
            else:
                col = "other"
            self.conf[i].setdefault(true_w, {})
            self.conf[i][true_w][col] = self.conf[i][true_w].get(col, 0) + 1
        self.btn_start.setEnabled(True)
        self.lbl_count.setText("done")
        self._refresh()

    def _refresh(self) -> None:
        for i, t in enumerate(self.tables):
            cols = self.cols[i]
            correct = total = 0
            for r, w in enumerate(PROMPT_WORDS):
                rowd = self.conf[i].get(w, {})
                for c, cl in enumerate(cols):
                    n = rowd.get(cl, 0)
                    it = QTableWidgetItem(str(n) if n else "")
                    it.setTextAlignment(Qt.AlignCenter)
                    if n and cl == w:
                        it.setBackground(QColor(40, 90, 40))
                    elif n:
                        it.setBackground(QColor(90, 40, 40))
                    t.setItem(r, c, it)
                    total += n
                    if cl == w:
                        correct += n
            self.acc_lbls[i].setText(
                f"accuracy: {100 * correct / total:.0f}%  (n={total})" if total else "accuracy: —")

    def reset(self) -> None:
        self.conf = [{} for _ in self.panels]
        self._refresh()

    def export_csv(self) -> None:
        import csv
        ts = time.strftime("%Y%m%d_%H%M%S")
        for i, p in enumerate(self.panels):
            out = Path.cwd() / f"confusion_{p.mcu_key()}_{ts}.csv"
            with open(out, "w", newline="") as f:
                wri = csv.writer(f)
                wri.writerow(["true\\pred"] + self.cols[i])
                for w in PROMPT_WORDS:
                    rowd = self.conf[i].get(w, {})
                    wri.writerow([w] + [rowd.get(cl, 0) for cl in self.cols[i]])
        self.lbl_count.setText(f"exported CSV(s) to {Path.cwd()}")


AUTO_WORDS = ["yes", "no", "on", "down", "up"]   # all present in DS-CNN and kws20 sets
AUTO_REPS = 5
AUTO_WIN_S = 5
AUTO_READY_S = 2


class AutoEvalWindow(QWidget):
    """Deterministic automated sweep: for each AUTO_WORDS entry, AUTO_REPS capture
    windows of AUTO_WIN_S seconds. Each window's majority prediction per MCU fills
    a live, colour-graded confusion matrix (one per MCU)."""

    def __init__(self, panels):
        super().__init__()
        self.setWindowTitle("Automated confusion-matrix analysis")
        self.panels = [p for p in panels if p.reader is not None]
        self.resize(460 * max(1, len(self.panels)), 560)
        self._collecting = False
        self._running = False
        self._buf: dict[int, list[str]] = {}
        self.conf: list[dict] = [{} for _ in self.panels]
        self.cols = AUTO_WORDS + ["unknown", "other", "(none)"]

        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.btn_start = QPushButton("Start automated analysis")
        self.btn_start.clicked.connect(self.start)
        bar.addWidget(self.btn_start)
        self.lbl = QLabel(f"Press Start. Words: {', '.join(AUTO_WORDS)} "
                          f"({AUTO_REPS}× {AUTO_WIN_S}s each).")
        self.lbl.setStyleSheet("font-size: 15px;")
        bar.addWidget(self.lbl, 1)
        b_csv = QPushButton("Export CSV"); b_csv.clicked.connect(self.export_csv)
        bar.addWidget(b_csv)
        root.addLayout(bar)

        if not self.panels:
            root.addWidget(QLabel("Connect at least one MCU first, then reopen."))
            return

        mats = QHBoxLayout()
        self.tables: list[QTableWidget] = []
        self.acc_lbls: list[QLabel] = []
        for p in self.panels:
            box = QVBoxLayout()
            box.addWidget(QLabel(f"<b>{p.cb_mcu.currentText()}</b> · {p.cb_model.currentText()}"))
            t = QTableWidget(len(AUTO_WORDS), len(self.cols))
            t.setHorizontalHeaderLabels(self.cols)
            t.setVerticalHeaderLabels(AUTO_WORDS)
            t.setEditTriggers(QTableWidget.NoEditTriggers)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.tables.append(t); box.addWidget(t)
            acc = QLabel("accuracy: —"); self.acc_lbls.append(acc); box.addWidget(acc)
            mats.addLayout(box)
        root.addLayout(mats, 1)

        for i, p in enumerate(self.panels):
            p.reader.record.connect(partial(self._collect, i))
        self._refresh()

    def _collect(self, i: int, rec: InferenceRecord) -> None:
        if self._collecting and rec.is_inference:
            lbl = label_in_set(rec.pred_idx,
                               MODEL_LABEL_SET.get(self.panels[i].model_token(), "dscnn"))
            if lbl:
                self._buf[i].append(lbl)

    def start(self) -> None:
        if self._running or not self.panels:
            return
        self._seq = [(w, r) for w in AUTO_WORDS for r in range(1, AUTO_REPS + 1)]
        self._step = 0
        self._running = True
        self.btn_start.setEnabled(False)
        self._timer = QTimer(self); self._timer.timeout.connect(self._tick)
        self._begin_ready()

    def _begin_ready(self) -> None:
        self._phase = "ready"; self._rem = AUTO_READY_S
        self._show_phase()
        self._timer.start(1000)

    def _show_phase(self) -> None:
        w, r = self._seq[self._step]
        if self._phase == "ready":
            self.lbl.setText(f"Get ready… '{w.upper()}'  (rep {r}/{AUTO_REPS})  in {self._rem}s")
            self.lbl.setStyleSheet("font-size: 16px; color: #e0b020;")
        else:
            word_n = 1 + self._step // AUTO_REPS
            self.lbl.setText(f"SAY '{w.upper()}' — {self._rem}s   "
                             f"(rep {r}/{AUTO_REPS}, word {word_n}/{len(AUTO_WORDS)})")
            self.lbl.setStyleSheet("font-size: 18px; color: #40c040; font-weight: bold;")

    def _tick(self) -> None:
        self._rem -= 1
        if self._rem > 0:
            self._show_phase(); return
        if self._phase == "ready":
            self._phase = "listen"; self._rem = AUTO_WIN_S
            self._buf = {i: [] for i in range(len(self.panels))}
            self._collecting = True
            self._show_phase()
        else:
            self._collecting = False
            self._timer.stop()
            self._finalize_step()

    def _finalize_step(self) -> None:
        w, _ = self._seq[self._step]
        for i in range(len(self.panels)):
            preds = self._buf.get(i, [])
            maj = Counter(preds).most_common(1)[0][0] if preds else "(none)"
            if maj in AUTO_WORDS:
                col = maj
            elif maj in ("unknown", "silence"):
                col = "unknown"
            elif maj == "(none)":
                col = "(none)"
            else:
                col = "other"
            self.conf[i].setdefault(w, {})
            self.conf[i][w][col] = self.conf[i][w].get(col, 0) + 1
        self._refresh()
        self._step += 1
        if self._step < len(self._seq):
            self._begin_ready()
        else:
            self._running = False
            self.btn_start.setEnabled(True)
            self.lbl.setText(f"Done — {len(self._seq)} captures complete.")
            self.lbl.setStyleSheet("font-size: 16px; color: #40c040;")

    def _refresh(self) -> None:
        for i, t in enumerate(self.tables):
            correct = total = 0
            for ri, w in enumerate(AUTO_WORDS):
                rowd = self.conf[i].get(w, {})
                for ci, cl in enumerate(self.cols):
                    n = rowd.get(cl, 0)
                    it = QTableWidgetItem(str(n) if n else "")
                    it.setTextAlignment(Qt.AlignCenter)
                    if n:
                        ratio = min(1.0, n / AUTO_REPS)
                        if cl == w:
                            it.setBackground(QColor(20, 70 + int(ratio * 150), 20))
                        else:
                            it.setBackground(QColor(70 + int(ratio * 150), 25, 25))
                        it.setForeground(QColor(255, 255, 255))
                    t.setItem(ri, ci, it)
                    total += n
                    if cl == w:
                        correct += n
            self.acc_lbls[i].setText(
                f"accuracy: {100 * correct / total:.0f}%  (n={total})" if total else "accuracy: —")

    def export_csv(self) -> None:
        import csv
        ts = time.strftime("%Y%m%d_%H%M%S")
        for i, p in enumerate(self.panels):
            out = Path.cwd() / f"auto_confusion_{p.mcu_key()}_{ts}.csv"
            with open(out, "w", newline="") as f:
                wri = csv.writer(f)
                wri.writerow(["true\\pred"] + self.cols)
                for w in AUTO_WORDS:
                    rowd = self.conf[i].get(w, {})
                    wri.writerow([w] + [rowd.get(cl, 0) for cl in self.cols])
        self.lbl.setText(f"Exported CSV(s) to {Path.cwd()}")


class MonitorWindow(QWidget):
    def __init__(self, n_slots: int):
        super().__init__()
        self.setWindowTitle(f"MCU KWS Dashboard — {n_slots} slot(s)")
        self.resize(560 * min(n_slots, 2), 740)
        self.flash_proc: QProcess | None = None

        root = QVBoxLayout(self)

        # --- Flash bar ---
        flash_bar = QHBoxLayout()
        flash_bar.addWidget(QLabel("<b>Flash</b>  mode:"))
        self.flash_mode = QComboBox()
        self.flash_mode.addItems(list(FLASH_MODES))
        flash_bar.addWidget(self.flash_mode)
        flash_bar.addWidget(QLabel("report:"))
        self.flash_report = QComboBox()           # applies to all 3 boards at flash time
        self.flash_report.addItems(list(REPORT_MODES.keys()))
        flash_bar.addWidget(self.flash_report)
        self.btn_flash_all = QPushButton("Flash All (Coral + MAX + STM32)")
        self.btn_flash_all.clicked.connect(lambda: self.start_flash(None))
        flash_bar.addWidget(self.btn_flash_all)
        self.btn_connect_all = QPushButton("Connect All")
        self.btn_connect_all.setToolTip("Open the serial connection on every slot that has a port")
        self.btn_connect_all.clicked.connect(self.connect_all)
        flash_bar.addWidget(self.btn_connect_all)
        self.btn_eval = QPushButton("Evaluate")
        self.btn_eval.setToolTip("Guided 5 s-per-word capture → live per-MCU confusion matrix")
        self.btn_eval.clicked.connect(self.open_eval)
        flash_bar.addWidget(self.btn_eval)
        self.btn_auto = QPushButton("Auto Eval")
        self.btn_auto.setToolTip("Deterministic sweep: yes/no/on/down/up × 5 × 5 s → confusion matrices")
        self.btn_auto.clicked.connect(self.open_auto_eval)
        flash_bar.addWidget(self.btn_auto)
        flash_bar.addStretch(1)
        if not flash_available():
            self.btn_flash_all.setEnabled(False)
            self.btn_flash_all.setToolTip("flash_all.sh not found")
        root.addLayout(flash_bar)

        # --- MCU panels ---
        grid = QGridLayout()
        self.panels: list[McuPanel] = []
        for i in range(n_slots):
            panel = McuPanel(i, on_flash=self.start_flash)
            if i < len(PROFILES):          # default each slot to a distinct MCU
                panel.cb_mcu.setCurrentIndex(i)
            self.panels.append(panel)
            grid.addWidget(panel, 0, i)
        root.addLayout(grid, 1)

        # Apply the "report" selection to the live view (hide unknown/silence rows)
        # so it works for every model, including firmware that can't gate them.
        self.flash_report.currentIndexChanged.connect(self._apply_report_filter)
        self._apply_report_filter()

        # --- Flash output log ---
        root.addWidget(QLabel("Flash output:"))
        self.flash_log = QPlainTextEdit()
        self.flash_log.setReadOnly(True)
        self.flash_log.setMaximumBlockCount(5000)
        self.flash_log.setFixedHeight(150)
        root.addWidget(self.flash_log)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(500)

    def connect_all(self) -> None:
        """Open the serial connection on every slot that has a port selected."""
        for p in self.panels:
            if p.reader is None and p.cb_port.currentText():
                p._connect()

    def open_eval(self) -> None:
        """Open the guided confusion-matrix window for the connected MCUs."""
        self._eval = EvalWindow(self.panels)   # keep a ref so Qt doesn't GC it
        self._eval.show()

    def open_auto_eval(self) -> None:
        """Open the deterministic automated confusion-matrix sweep."""
        self._auto = AutoEvalWindow(self.panels)   # keep a ref so Qt doesn't GC it
        self._auto.show()

    def _apply_report_filter(self) -> None:
        """Hide unknown/silence rows in the live view per the 'report' selection."""
        token = REPORT_MODES.get(self.flash_report.currentText(), "keywords")
        hidden = {"keywords": {"unknown", "silence"}, "unknown": {"silence"},
                  "silence": {"unknown"}, "both": set()}.get(token, set())
        for p in self.panels:
            p.hidden_labels = set(hidden)

    # ── flashing ────────────────────────────────────────────────────────
    def start_flash(self, mcu: str | None) -> None:
        if self.flash_proc is not None:
            self._log("[flash] busy — wait for the current flash to finish.")
            return
        mode = self.flash_mode.currentText()
        # Free serial ports / avoid probe contention: disconnect affected readers.
        affected = self.panels if mcu is None else [
            p for p in self.panels if p.mcu_key() == mcu]
        for p in affected:
            if p.reader is not None:
                p._disconnect()
        report = REPORT_MODES[self.flash_report.currentText()]
        max_model = next((p.model_token() for p in self.panels
                          if p.mcu_key() == "max78000"), "dscnn")
        cmd = flash_command(mode, mcu, report, max_model)
        self._log(f"\n=== flashing {mcu or 'ALL'} (mode={mode}) ===")
        self._set_flash_enabled(False)
        self.flash_proc = QProcess(self)
        self.flash_proc.setProcessChannelMode(QProcess.MergedChannels)
        self.flash_proc.readyRead.connect(self._flash_output)
        self.flash_proc.finished.connect(self._flash_finished)
        self.flash_proc.start(cmd[0], cmd[1:])

    def _flash_output(self) -> None:
        if self.flash_proc is None:
            return
        data = bytes(self.flash_proc.readAll()).decode("utf-8", "replace")
        for line in data.splitlines():
            self._log(line)

    def _flash_finished(self, code: int, _status) -> None:
        self._log(f"=== flash finished (exit {code}) ===")
        if code == 0:
            self._log("Reconnect the affected MCU(s) to resume monitoring.")
        self.flash_proc = None
        self._set_flash_enabled(True)

    def _set_flash_enabled(self, on: bool) -> None:
        self.btn_flash_all.setEnabled(on and flash_available())
        for p in self.panels:
            p.btn_flash.setEnabled(on)

    def _log(self, text: str) -> None:
        self.flash_log.appendPlainText(text)

    def _tick(self) -> None:
        for p in self.panels:
            p.tick_stats()

    def closeEvent(self, event) -> None:
        if self.flash_proc is not None:
            self.flash_proc.kill()
        for p in self.panels:
            p.shutdown()
        super().closeEvent(event)


class Launcher(QDialog):
    """Start screen: Individual (1 MCU) or Multiple (up to 3)."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MCU KWS Dashboard")
        self.n_slots = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Choose a monitoring mode</b>"))

        self.cb = QComboBox()
        self.cb.addItem("Individual — one MCU", 1)
        self.cb.addItem("Multiple — 2 MCUs", 2)
        self.cb.addItem("Multiple — 3 MCUs", 3)
        layout.addWidget(self.cb)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        self.n_slots = self.cb.currentData()
        self.accept()


def main() -> int:
    app = QApplication([])
    launcher = Launcher()
    if launcher.exec() != QDialog.Accepted:
        return 0
    win = MonitorWindow(launcher.n_slots)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
