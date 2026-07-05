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

# Report-class gating — applies to all 3 boards. label -> flash_all.sh --report token.
REPORT_MODES = {
    "Keywords only": "keywords",
    "+ unknown": "unknown",
    "+ silence": "silence",
    "+ unknown & silence": "both",
}


# Per-MCU flashable models: label -> flash_all.sh model token.
# Every board exposes the ledger's precision split — v0 = fp32-cpu (32-bit) and
# v1 (8-bit) — selected via --stm32-model / --max-model / --coral-model. The
# first entry is the dropdown default (kept as the working 8-bit/INT8 build)
MODELS = {
    "max78000": {"DS-CNN (INT8+accel, 8-bit)": "v1",
                 "DS-CNN (FP32, 32-bit)": "v0",
                 "kws20_demo": "kws20_demo",
                 "kws20_demo_w90": "kws20_demo_w90"},
    "stm32u5":  {"DS-CNN (INT8, 8-bit)": "v1",
                 "DS-CNN (FP32, 32-bit)": "v0"},
    # NOTE: the "int8" apps currently load the broken v1.0 edgetpu model
    # (4-block/random-calib, 23.6% acc — see models/MODELS.md). A FIXED-v2 entry
    # needs apps pointed at ds_cnn_l_static_v2_edgetpu.tflite (TODO with re-measure).
    "coral":    {"DS-CNN (INT8 v2, Edge-TPU)": "int8",
                 "DS-CNN (FP32, CPU)": "fp32"},
    # GAP9 is built+flashed by its own Connect button (gap9.py), never by
    # flash_all.sh — this entry only feeds the model dropdown / label set.
    "gap9":     {"DS-CNN-L (FP32, Deeploy)": "fp32"},
}

# Which label set each model's pred_idx uses (see protocol.LABEL_SETS).
# DS-CNN v0/v1 and Coral int8/fp32 are all the 12-class DS-CNN → dscnn labels.
MODEL_LABEL_SET = {
    "v0": "dscnn", "v1": "dscnn",
    "dscnn": "dscnn", "int8": "dscnn", "fp32": "dscnn",
    "kws20_demo": "kws20", "kws20_demo_w90": "kws20",
}


def flash_command(mode: str, mcu: str | None = None,
                  report: str = "keywords", max_model: str = "v1",
                  coral_model: str = "int8", stm32_model: str = "v1") -> list[str]:
    """Build the flash_all.sh command. mcu=None flashes all three."""
    cmd = ["bash", str(FLASH_ALL), "--mode", mode, "--report", report,
           "--max-model", max_model, "--coral-model", coral_model,
           "--stm32-model", stm32_model]
    if mcu is not None:
        cmd += ["--only", ONLY_TOKEN[mcu]]
    return cmd


def flash_available() -> bool:
    return FLASH_ALL.exists()
