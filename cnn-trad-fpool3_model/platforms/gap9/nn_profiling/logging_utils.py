
import logging
import os

import yaml

# ──────────────────────────────────────────────────────────────────────────────
# Colored logging
# ──────────────────────────────────────────────────────────────────────────────
class ColorFormatter(logging.Formatter):
    """Custom formatter with ANSI colors per log level."""

    RESET = "\033[0m"
    COLORS = {
        logging.DEBUG:    "\033[36m",      # cyan
        logging.INFO:     "\033[32m",      # green
        logging.WARNING:  "\033[33m",      # yellow
        logging.ERROR:    "\033[31m",      # red
        logging.CRITICAL: "\033[1;31m",    # bold red
    }
    LABEL_COLORS = {
        logging.DEBUG:    "\033[1;36m",    # bold cyan
        logging.INFO:     "\033[1;32m",    # bold green
        logging.WARNING:  "\033[1;33m",    # bold yellow
        logging.ERROR:    "\033[1;31m",    # bold red
        logging.CRITICAL: "\033[1;41m",    # bold red bg
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        label_color = self.LABEL_COLORS.get(record.levelno, self.RESET)
        levelname = record.levelname
        msg = record.getMessage()
        timestamp = self.formatTime(record, "%H:%M:%S")
        return f"\033[90m{timestamp}\033[0m {label_color}{levelname:<8}{self.RESET} {color}{msg}{self.RESET}"


_ANSI_RE = None

def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    global _ANSI_RE
    if _ANSI_RE is None:
        import re
        _ANSI_RE = re.compile(r"\033\[[0-9;]*m")
    return _ANSI_RE.sub("", text)


class PlainFormatter(logging.Formatter):
    """Formatter that strips ANSI codes for file output."""

    def __init__(self):
        super().__init__("%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record):
        record.msg = _strip_ansi(str(record.msg))
        if record.args:
            record.args = tuple(
                _strip_ansi(str(a)) if isinstance(a, str) else a
                for a in (record.args if isinstance(record.args, tuple) else (record.args,))
            )
        return super().format(record)


def setup_logging(log_dir: str = None) -> logging.Logger:
    """Configure the root logger with colored console output and optional file handler."""
    logger = logging.getLogger("nn_profiler")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler (colored)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(ColorFormatter())
    logger.addHandler(ch)

    # File handler (plain, inside experiment dir)
    if log_dir:
        fh = logging.FileHandler(os.path.join(log_dir, "profiler.log"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(PlainFormatter())
        logger.addHandler(fh)

    return logger

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def format_bytes(n):
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"

