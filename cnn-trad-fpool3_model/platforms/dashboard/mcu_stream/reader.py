"""Background readers: one QThread per MCU.

SerialReader opens a serial port (asserting DTR for Coral, whose CDC console
drops bytes until the host raises DTR). ProcessReader spawns a subprocess and
reads its stdout (GAP9: docker/OpenOCD semihosting, see gap9.py). Both read
line by line, host-timestamp each line with time.monotonic() for cross-MCU
synchronization, normalize it via protocol.py, and emit the same Qt signals
consumed by the GUI thread.
"""
from __future__ import annotations

import glob
import re
import subprocess
import time
from typing import Dict, List, Optional, Sequence

import serial
from serial.tools import list_ports
from PySide6.QtCore import QThread, Signal

from .protocol import PROFILES, build_record, InferenceRecord


def list_serial_ports() -> List[str]:
    """Available USB serial call-out devices on macOS/Linux."""
    ports = sorted(set(
        glob.glob("/dev/cu.usbmodem*")
        + glob.glob("/dev/cu.usbserial*")
        + glob.glob("/dev/ttyACM*")
        + glob.glob("/dev/ttyUSB*")
    ))
    return ports


def detect_ports() -> Dict[str, str]:
    """Map each known MCU key -> its serial port, by USB vendor id.

    e.g. {"coral": "/dev/cu.usbmodem1201", "stm32u5": "/dev/cu.usbmodem13103",
          "max78000": "/dev/cu.usbmodem1102"}. Missing boards are omitted.
    """
    vid_to_mcu = {p.usb_vid: p.key for p in PROFILES.values() if p.usb_vid}
    found: Dict[str, str] = {}
    for info in list_ports.comports():
        mcu = vid_to_mcu.get(info.vid)
        if mcu and mcu not in found:
            found[mcu] = info.device
    return found


class SerialReader(QThread):
    record = Signal(object)   # InferenceRecord
    raw = Signal(str)         # every decoded line, including noise
    status = Signal(str)      # human-readable connection status

    def __init__(self, mcu: str, mode: str, port: str, baud: int,
                 parent=None):
        super().__init__(parent)
        self.mcu = mcu
        self.mode = mode
        self.port = port
        self.baud = baud
        self.assert_dtr = PROFILES[mcu].assert_dtr
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:  # noqa: C901 - simple state machine
        while not self._stop:
            ser = None
            try:
                self.status.emit(f"opening {self.port} @ {self.baud}…")
                ser = serial.Serial()
                ser.port = self.port
                ser.baudrate = self.baud
                # NOTE: 1200 baud would reset Coral to bootloader — never use it.
                ser.timeout = 0.3
                if self.assert_dtr:
                    ser.dtr = True
                    ser.rts = True
                ser.open()
                if self.assert_dtr:
                    ser.dtr = True  # re-assert after open for Coral CDC
                self.status.emit(f"connected: {self.port}")

                while not self._stop:
                    data = ser.readline()
                    if not data:
                        continue
                    t_host = time.monotonic()
                    t_wall = time.time()
                    line = data.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    self.raw.emit(line)
                    rec: InferenceRecord | None = build_record(
                        self.mcu, self.mode, line, t_host, t_wall)
                    if rec is not None:
                        self.record.emit(rec)

            except serial.SerialException as exc:
                self.status.emit(f"disconnected ({exc}); retrying…")
            except Exception as exc:  # pragma: no cover - defensive
                self.status.emit(f"error: {exc}")
            finally:
                try:
                    if ser is not None and ser.is_open:
                        ser.close()
                except Exception:
                    pass

            if self._stop:
                break
            # brief backoff before reconnect attempt (board re-enumerating)
            for _ in range(10):
                if self._stop:
                    break
                self.msleep(50)

        self.status.emit("stopped")


# docker -t allocates a pty, so lines may carry ANSI escapes — strip them.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class ProcessReader(QThread):
    """Stream inference records from a subprocess's stdout (GAP9 docker run).

    Same signal interface as SerialReader. No reconnect loop: when the process
    exits, the reader reports it and stops — for GAP9 the process IS the
    build+flash+run session, so a restart means a fresh Connect.
    """

    record = Signal(object)   # InferenceRecord
    raw = Signal(str)         # every decoded line, including noise
    status = Signal(str)      # human-readable connection status

    # A line containing this marks the firmware as up and streaming.
    READY_MARKER = "Microphone ready."

    def __init__(self, mcu: str, mode: str, command: Sequence[str],
                 pre_commands: Sequence[Sequence[str]] = (),
                 stop_command: Optional[Sequence[str]] = None,
                 parent=None):
        super().__init__(parent)
        self.mcu = mcu
        self.mode = mode
        self.command = list(command)
        self.pre_commands = [list(c) for c in pre_commands]
        self.stop_command = list(stop_command) if stop_command else None
        self._stop = False
        self._popen: Optional[subprocess.Popen] = None

    def stop(self) -> None:
        self._stop = True
        # The reader thread blocks in readline(); ending the process (docker
        # stop) is what unblocks it. Called from the GUI thread.
        if self.stop_command:
            try:
                subprocess.run(self.stop_command, capture_output=True,
                               timeout=10)
            except Exception:
                pass
        p = self._popen
        if p is not None and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass

    def run(self) -> None:
        for cmd in self.pre_commands:
            try:
                subprocess.run(cmd, capture_output=True, timeout=30)
            except Exception:
                pass

        self.status.emit("starting container (build + flash, ~1-2 min)…")
        try:
            self._popen = subprocess.Popen(
                self.command, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                errors="replace")
        except OSError as exc:
            self.status.emit(f"error: {exc}")
            return

        ready = False
        try:
            for raw_line in iter(self._popen.stdout.readline, ""):
                if self._stop:
                    break
                line = _ANSI_RE.sub("", raw_line).strip()
                if not line:
                    continue
                t_host = time.monotonic()
                t_wall = time.time()
                self.raw.emit(line)
                if not ready:
                    if self.READY_MARKER in line:
                        ready = True
                        self.status.emit(f"connected: {self.mcu} (board running)")
                    else:
                        # Surface build/flash progress in the status label.
                        self.status.emit(line[:90])
                rec: InferenceRecord | None = build_record(
                    self.mcu, self.mode, line, t_host, t_wall)
                if rec is not None:
                    self.record.emit(rec)
        finally:
            code = self._popen.wait() if self._popen is not None else -1
            self._popen = None

        if self._stop:
            self.status.emit("stopped")
        else:
            self.status.emit(f"disconnected (process exited, code {code})")
