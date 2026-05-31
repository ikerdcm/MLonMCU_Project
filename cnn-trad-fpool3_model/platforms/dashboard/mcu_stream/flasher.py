"""Build + flash orchestration for the dashboard.

Routes all flashing through platforms/flash_all.sh, which already knows each
MCU's mode mechanism (Coral script choice; MAX/STM32 kws20_mode_config.h edit)
and targets each board via its own debug probe (no serial-port ambiguity).
"""
from __future__ import annotations

from pathlib import Path

# mcu_stream/ -> dashboard/ -> platforms/
PLATFORMS_DIR = Path(__file__).resolve().parents[2]
FLASH_ALL = PLATFORMS_DIR / "flash_all.sh"

# Dashboard MCU key -> flash_all.sh --only token
ONLY_TOKEN = {"coral": "coral", "stm32u5": "stm32", "max78000": "max"}

# Modes flash_all.sh understands. Coral "bench" maps to offline there.
FLASH_MODES = ("live", "offline")


def flash_command(mode: str, mcu: str | None = None) -> list[str]:
    """Build the flash_all.sh command. mcu=None flashes all three."""
    cmd = ["bash", str(FLASH_ALL), "--mode", mode]
    if mcu is not None:
        cmd += ["--only", ONLY_TOKEN[mcu]]
    return cmd


def flash_available() -> bool:
    return FLASH_ALL.exists()
