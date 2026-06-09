"""Background serial reader: one QThread per MCU.

Opens the port (asserting DTR for Coral, whose CDC console drops bytes until the
host raises DTR), reads line by line, host-timestamps each line with
time.monotonic() for cross-MCU synchronization, normalizes it via protocol.py,
and emits Qt signals consumed by the GUI thread.
"""
from __future__ import annotations

import glob
import time
from typing import Dict, List

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
