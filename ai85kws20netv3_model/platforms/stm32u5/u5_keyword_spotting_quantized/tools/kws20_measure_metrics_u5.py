#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys


def main():
    this_dir = Path(__file__).resolve().parent
    shared_script = this_dir.parent.parent / "tools" / "kws20_measure_metrics_u5.py"
    default_config = this_dir / "kws20_measure_u5_quantized_config.json"

    argv = [str(shared_script)]
    if "--config" not in sys.argv[1:]:
        argv.extend(["--config", str(default_config)])
    argv.extend(sys.argv[1:])
    sys.argv = argv
    runpy.run_path(str(shared_script), run_name="__main__")


if __name__ == "__main__":
    main()
