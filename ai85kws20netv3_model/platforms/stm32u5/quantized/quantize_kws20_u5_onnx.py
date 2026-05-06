#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def detect_layout(model_path: Path) -> tuple[str, str]:
    model = onnx.load(str(model_path))
    inp = model.graph.input[0]
    name = inp.name
    dims = []
    for dim in inp.type.tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else None)
    if dims == [1, 1, 128, 128]:
        return name, "nchw"
    if dims == [1, 128, 128, 1]:
        return name, "nhwc"
    raise SystemExit(f"Unsupported input shape for quantization: {dims}")


class NpzCalibrationReader(CalibrationDataReader):
    def __init__(self, input_name: str, npz_path: Path):
        data = np.load(npz_path)
        self.samples = data["x_test"].astype(np.float32, copy=False)
        self.input_name = input_name
        self.index = 0

    def get_next(self):
        if self.index >= len(self.samples):
            return None
        sample = self.samples[self.index:self.index + 1]
        self.index += 1
        return {self.input_name: sample}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_path = Path(cfg["model_path"]).expanduser().resolve()
    cal_nhwc = Path(cfg["calibration_nhwc"]).expanduser().resolve()
    cal_nchw = Path(cfg["calibration_nchw"]).expanduser().resolve()
    out_dir = Path(cfg["output_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_name, layout = detect_layout(model_path)
    cal_path = cal_nchw if layout == "nchw" else cal_nhwc

    out_model = out_dir / f"{model_path.stem}_int8_qdq.onnx"
    out_report = out_dir / f"{model_path.stem}_int8_qdq_report.json"

    reader = NpzCalibrationReader(input_name, cal_path)
    quantize_static(
        model_input=str(model_path),
        model_output=str(out_model),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
        },
    )

    model_q = onnx.load(str(out_model))
    ops = {}
    for node in model_q.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1

    report = {
        "input_model": str(model_path),
        "output_model": str(out_model),
        "input_name": input_name,
        "input_layout": layout,
        "calibration_file": str(cal_path),
        "input_model_size_bytes": model_path.stat().st_size,
        "output_model_size_bytes": out_model.stat().st_size,
        "ops": ops,
    }
    out_report.write_text(json.dumps(report, indent=2))

    print(f"[OK] wrote {out_model}")
    print(f"[OK] wrote {out_report}")
    print(f"[INFO] input layout: {layout}")
    print(f"[INFO] size: {model_path.stat().st_size} -> {out_model.stat().st_size} bytes")


if __name__ == "__main__":
    main()
