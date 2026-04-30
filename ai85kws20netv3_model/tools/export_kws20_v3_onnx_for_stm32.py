#!/usr/bin/env python3
import copy
import importlib.util
import sys
from pathlib import Path

import torch
import onnx

REPO = Path.home() / "max78000/ai8x-training"
OUT_DIR = Path.home() / "max78000/stm32u5_kws20"

MODEL_FILE = REPO / "models/ai85net-kws20-v3.py"

# For STM32 export use the QAT checkpoint, not the ADI -q synthesis checkpoint.
CHECKPOINT = Path.home() / "max78000/ai8x-synthesis/trained/ai85-kws20_v3-qat8.pth.tar"

OUT_QAT = OUT_DIR / "kws20_v3_qat_onnx_for_stm32u5.onnx"
OUT_SIMPLE = OUT_DIR / "kws20_v3_simplified_onnx_for_stm32u5.onnx"

OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO))
import ai8x

ai8x.set_device(device=85, simulate=False, round_avg=False, verbose=True)

spec = importlib.util.spec_from_file_location("ai85net_kws20_v3", MODEL_FILE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def build_model():
    model = mod.ai85kws20netv3(
        pretrained=False,
        num_classes=21,
        num_channels=128,
        dimensions=(128, 1),
        bias=True,
    )

    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_state[k] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    ai8x.update_model(model)
    model.eval()
    return model


def export_one(model, out_path, simplify):
    dummy = torch.zeros(1, 128, 128, dtype=torch.float32)

    with torch.no_grad():
        y = model(dummy)
        print(f"{out_path.name} output shape before ONNX:", tuple(y.shape))

    # Important: removes/replaces ai8x export-unfriendly ops like torch.quantile.
    ai8x.onnx_export_prep(model, simplify=simplify, remove_clamp=simplify)
    model.eval()

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["kws_input"],
        output_names=["kws_logits"],
        opset_version=13,
        do_constant_folding=True,
    )

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    ops = {}
    for node in onnx_model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1

    print(f"\nONNX written: {out_path}")
    print(f"Size: {out_path.stat().st_size / 1024:.1f} KiB")
    print("Ops:")
    for k in sorted(ops):
        print(f"  {k}: {ops[k]}")


print("\n=== Export QAT-like ONNX ===")
model_qat = build_model()
export_one(model_qat, OUT_QAT, simplify=False)

print("\nDONE")
print(f"QAT-like: {OUT_QAT}")
print("Simplified export skipped because ai8x simplify=True causes bias/scale mismatch for this model.")
