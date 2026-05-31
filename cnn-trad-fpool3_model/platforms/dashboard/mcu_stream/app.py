"""PyQt (PySide6) dashboard for monitoring 1–3 MCUs simultaneously.

v1 = monitor only: pick MCU + port + firmware mode per slot, Connect, and watch
a normalized live inference table with per-stream stats. Cross-MCU sync is by
host arrival timestamp (time.monotonic()).
"""
from __future__ import annotations

import time
from collections import deque

from PySide6.QtCore import Qt, QTimer, QProcess
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    QHeaderView, QDialogButtonBox, QPlainTextEdit,
)

from .protocol import PROFILES, InferenceRecord
from .reader import SerialReader, list_serial_ports, detect_ports
from .flasher import flash_command, flash_available, FLASH_MODES

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
        self._build_ui()

    def mcu_key(self) -> str:
        return self.cb_mcu.currentData()

    # ── UI ──────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.cb_mcu = QComboBox()
        for key, prof in PROFILES.items():
            self.cb_mcu.addItem(prof.name, key)
        self.cb_mcu.currentIndexChanged.connect(self._on_mcu_changed)

        self.cb_mode = QComboBox()
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

        controls.addWidget(QLabel("MCU:"))
        controls.addWidget(self.cb_mcu)
        controls.addWidget(QLabel("Mode:"))
        controls.addWidget(self.cb_mode)
        controls.addWidget(QLabel("Port:"))
        controls.addWidget(self.cb_port)
        controls.addWidget(self.btn_refresh)
        controls.addWidget(self.btn_connect)
        controls.addWidget(self.btn_flash)
        controls.addStretch(1)
        root.addLayout(controls)

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

        self._on_mcu_changed()
        self.refresh_ports()

    def _on_mcu_changed(self) -> None:
        prof = PROFILES[self.cb_mcu.currentData()]
        self.cb_mode.clear()
        self.cb_mode.addItems(prof.modes)
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
        if not rec.is_inference:
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
            rec.label or "?",
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
        if rec.extra.get("confident") or (rec.label not in (None, "unknown", "silence")):
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
        self.btn_flash_all = QPushButton("Flash All (Coral + MAX + STM32)")
        self.btn_flash_all.clicked.connect(lambda: self.start_flash(None))
        flash_bar.addWidget(self.btn_flash_all)
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
        cmd = flash_command(mode, mcu)
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
