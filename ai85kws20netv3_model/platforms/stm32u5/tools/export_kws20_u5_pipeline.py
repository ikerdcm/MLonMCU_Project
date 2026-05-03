#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper


def clean_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def get_attr(node: onnx.NodeProto, name: str):
    for attr in node.attribute:
        if attr.name == name:
            return attr
    return None


def get_attr_ints(node: onnx.NodeProto, name: str):
    attr = get_attr(node, name)
    if attr is None:
        return None
    return list(attr.ints)


def set_attr_ints(node: onnx.NodeProto, name: str, values: list[int]) -> None:
    attr = get_attr(node, name)
    if attr is None:
        attr = node.attribute.add()
        attr.name = name
        attr.type = onnx.AttributeProto.INTS
    while len(attr.ints) > 0:
        attr.ints.pop()
    attr.ints.extend(values)


def clear_shape_dims(shape) -> None:
    while len(shape.dim) > 0:
        shape.dim.pop()


def set_input_shape(graph: onnx.GraphProto, dims: list[int], clear_denotation: bool = False) -> None:
    inp = graph.input[0]
    shape = inp.type.tensor_type.shape
    clear_shape_dims(shape)
    for d in dims:
        dim = shape.dim.add()
        dim.dim_value = d
        if clear_denotation:
            dim.denotation = ""


def clear_value_info(graph: onnx.GraphProto) -> None:
    while len(graph.value_info) > 0:
        graph.value_info.pop()


def run_shape_inference(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        return onnx.shape_inference.infer_shapes(model)
    except Exception as exc:
        print("shape inference warning:", exc)
        return model


def print_model_ops(model: onnx.ModelProto) -> None:
    ops = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print("Ops:")
    for key in sorted(ops):
        print(f"  {key}: {ops[key]}")


def export_qat_onnx(repo_dir: Path, checkpoint: Path, out_path: Path) -> None:
    sys.path.insert(0, str(repo_dir))
    import ai8x  # pylint: disable=import-error

    model_file = repo_dir / "models/ai85net-kws20-v3.py"

    ai8x.set_device(device=85, simulate=False, round_avg=False, verbose=True)

    spec = importlib.util.spec_from_file_location("ai85net_kws20_v3", model_file)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    model = mod.ai85kws20netv3(
        pretrained=False,
        num_classes=21,
        num_channels=128,
        dimensions=(128, 1),
        bias=True,
    )

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    clean_state = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        clean_state[key] = value

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    ai8x.update_model(model)
    model.eval()

    dummy = torch.zeros(1, 128, 128, dtype=torch.float32)
    with torch.no_grad():
        y = model(dummy)
        print(f"{out_path.name} output shape before ONNX:", tuple(y.shape))

    ai8x.onnx_export_prep(model, simplify=False, remove_clamp=False)
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

    model_onnx = onnx.load(str(out_path))
    onnx.checker.check_model(model_onnx)
    print(f"\nONNX written: {out_path}")
    print(f"Size: {out_path.stat().st_size / 1024:.1f} KiB")
    print_model_ops(model_onnx)


def patch_qat_1d_to_2d(in_path: Path, out_path: Path) -> None:
    model = onnx.load(str(in_path))
    graph = model.graph

    set_input_shape(graph, [1, 128, 128, 1])
    init_map = {init.name: init for init in graph.initializer}

    new_nodes = []
    patched_conv_attr = 0
    patched_conv_init = 0
    inserted_unsqueeze = 0
    patched_pool = 0

    for node in graph.node:
        if node.op_type == "Conv":
            k = get_attr_ints(node, "kernel_shape")
            if k and len(k) == 1:
                set_attr_ints(node, "kernel_shape", [k[0], 1])
                patched_conv_attr += 1

            s = get_attr_ints(node, "strides")
            if s and len(s) == 1:
                set_attr_ints(node, "strides", [s[0], 1])

            d = get_attr_ints(node, "dilations")
            if d and len(d) == 1:
                set_attr_ints(node, "dilations", [d[0], 1])

            p = get_attr_ints(node, "pads")
            if p and len(p) == 2:
                set_attr_ints(node, "pads", [p[0], 0, p[1], 0])

            wname = node.input[1]
            if wname in init_map:
                w_init = init_map[wname]
                w = numpy_helper.to_array(w_init)
                if w.ndim == 3:
                    w2 = w[:, :, :, None].astype(w.dtype)
                    w_init.CopyFrom(numpy_helper.from_array(w2, name=w_init.name))
                    patched_conv_init += 1
            else:
                axes_name = f"{node.name or 'conv'}_weight_unsq_axes_{inserted_unsqueeze}"
                unsq_out = f"{node.input[1]}_unsq4d_for_{node.name or inserted_unsqueeze}"
                axes_init = numpy_helper.from_array(np.array([3], dtype=np.int64), name=axes_name)
                graph.initializer.append(axes_init)
                unsq_node = helper.make_node(
                    "Unsqueeze",
                    inputs=[node.input[1], axes_name],
                    outputs=[unsq_out],
                    name=f"{node.name or 'conv'}_weight_unsqueeze",
                )
                new_nodes.append(unsq_node)
                node.input[1] = unsq_out
                inserted_unsqueeze += 1
        elif node.op_type in ("MaxPool", "AveragePool"):
            k = get_attr_ints(node, "kernel_shape")
            if k and len(k) == 1:
                set_attr_ints(node, "kernel_shape", [k[0], 1])
                patched_pool += 1

            s = get_attr_ints(node, "strides")
            if s and len(s) == 1:
                set_attr_ints(node, "strides", [s[0], 1])

            d = get_attr_ints(node, "dilations")
            if d and len(d) == 1:
                set_attr_ints(node, "dilations", [d[0], 1])

            p = get_attr_ints(node, "pads")
            if p and len(p) == 2:
                set_attr_ints(node, "pads", [p[0], 0, p[1], 0])

        new_nodes.append(node)

    while len(graph.node) > 0:
        graph.node.pop()
    graph.node.extend(new_nodes)

    clear_value_info(graph)
    onnx.checker.check_model(model)
    model = run_shape_inference(model)
    onnx.save(model, str(out_path))

    print("\n=== Stage: patch_qat_1d_to_2d ===")
    print("input :", in_path)
    print("output:", out_path)
    print("patched conv attrs:", patched_conv_attr)
    print("patched direct conv initializers:", patched_conv_init)
    print("inserted conv weight unsqueeze:", inserted_unsqueeze)
    print("patched pool nodes:", patched_pool)
    print("size KiB:", out_path.stat().st_size / 1024)


def convert_to_true_nchw(in_path: Path, out_path: Path) -> None:
    model = onnx.load(str(in_path))
    graph = model.graph

    set_input_shape(graph, [1, 1, 128, 128])

    producer = {}
    for node in graph.node:
        for out in node.output:
            producer[out] = node

    conv_nodes = [n for n in graph.node if n.op_type == "Conv"]
    if not conv_nodes:
        raise SystemExit("No Conv nodes found")

    first_conv = conv_nodes[0]
    old_w_input = first_conv.input[1]
    if old_w_input not in producer:
        raise SystemExit(f"Could not find producer of first conv weight: {old_w_input}")

    old_unsq = producer[old_w_input]
    if old_unsq.op_type != "Unsqueeze":
        raise SystemExit(f"Expected first conv weight producer to be Unsqueeze, got {old_unsq.op_type}")

    pre_weight = old_unsq.input[0]
    w_transposed = pre_weight + "_true2d_transposed"
    w_true2d = pre_weight + "_true2d_weight"

    w_transpose_node = helper.make_node(
        "Transpose",
        inputs=[pre_weight],
        outputs=[w_transposed],
        perm=[0, 2, 1],
        name="first_conv_weight_transpose_true2d",
    )

    axis_name = "first_conv_weight_unsqueeze_axis_1"
    axis_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name=axis_name)
    graph.initializer.append(axis_init)

    w_unsqueeze_node = helper.make_node(
        "Unsqueeze",
        inputs=[w_transposed, axis_name],
        outputs=[w_true2d],
        name="first_conv_weight_unsqueeze_true2d",
    )

    first_conv.input[1] = w_true2d

    k = get_attr_ints(first_conv, "kernel_shape")
    if not k or len(k) != 2:
        raise SystemExit(f"Unexpected first conv kernel_shape: {k}")
    set_attr_ints(first_conv, "kernel_shape", [k[0], 128])

    s = get_attr_ints(first_conv, "strides")
    if s and len(s) == 2:
        set_attr_ints(first_conv, "strides", [s[0], 1])

    d = get_attr_ints(first_conv, "dilations")
    if d and len(d) == 2:
        set_attr_ints(first_conv, "dilations", [d[0], 1])

    p = get_attr_ints(first_conv, "pads")
    if p and len(p) == 4:
        set_attr_ints(first_conv, "pads", [p[0], 0, p[2], 0])

    new_nodes = []
    inserted = False
    for node in graph.node:
        if node is first_conv and not inserted:
            new_nodes.append(w_transpose_node)
            new_nodes.append(w_unsqueeze_node)
            inserted = True
        new_nodes.append(node)

    while len(graph.node) > 0:
        graph.node.pop()
    graph.node.extend(new_nodes)

    clear_value_info(graph)
    onnx.checker.check_model(model)
    model = run_shape_inference(model)
    onnx.save(model, str(out_path))

    print("\n=== Stage: convert_to_true_nchw ===")
    print("input :", in_path)
    print("output:", out_path)
    print("external input shape: [1,1,128,128]")
    print("first conv patched to kernel [K,128] with weight [O,1,K,128]")
    print("size KiB:", out_path.stat().st_size / 1024)


def _eval_fold_node(node: onnx.NodeProto, values: dict[str, np.ndarray]):
    op = node.op_type
    ins = []
    for name in node.input:
        if name == "":
            ins.append(None)
        elif name in values:
            ins.append(values[name])
        else:
            return None

    try:
        if op == "Constant":
            attr = get_attr(node, "value")
            if attr is not None:
                return [numpy_helper.to_array(attr.t)]
        if op == "Identity":
            return [ins[0]]
        if op == "Add":
            return [ins[0] + ins[1]]
        if op == "Sub":
            return [ins[0] - ins[1]]
        if op == "Mul":
            return [ins[0] * ins[1]]
        if op == "Div":
            return [ins[0] / ins[1]]
        if op == "Neg":
            return [-ins[0]]
        if op == "Abs":
            return [np.abs(ins[0])]
        if op == "Pow":
            return [np.power(ins[0], ins[1])]
        if op == "Reciprocal":
            return [1.0 / ins[0]]
        if op == "Log":
            return [np.log(ins[0])]
        if op == "Floor":
            return [np.floor(ins[0])]
        if op == "Round":
            return [np.round(ins[0])]
        if op == "Clip":
            lo = ins[1] if len(ins) > 1 and ins[1] is not None else -np.inf
            hi = ins[2] if len(ins) > 2 and ins[2] is not None else np.inf
            return [np.clip(ins[0], lo, hi)]
        if op == "ReduceMax":
            axes_attr = get_attr(node, "axes")
            keepdims_attr = get_attr(node, "keepdims")
            axes = tuple(axes_attr.ints) if axes_attr is not None else None
            keepdims = bool(keepdims_attr.i) if keepdims_attr is not None else True
            return [np.max(ins[0], axis=axes, keepdims=keepdims)]
        if op == "Reshape":
            return [np.reshape(ins[0], ins[1].astype(np.int64).tolist())]
        if op == "Transpose":
            perm_attr = get_attr(node, "perm")
            perm = list(perm_attr.ints) if perm_attr is not None else None
            return [np.transpose(ins[0], axes=perm)]
        if op == "Unsqueeze":
            axes_attr = get_attr(node, "axes")
            if axes_attr is not None:
                axes = list(axes_attr.ints)
            elif len(ins) >= 2 and ins[1] is not None:
                axes = ins[1].astype(np.int64).tolist()
            else:
                return None
            y = ins[0]
            for ax in sorted(axes):
                y = np.expand_dims(y, axis=ax)
            return [y]
        if op == "Cast":
            to_attr = get_attr(node, "to")
            if to_attr is None:
                return None
            to = to_attr.i
            mapping = {
                TensorProto.FLOAT: np.float32,
                TensorProto.DOUBLE: np.float64,
                TensorProto.INT64: np.int64,
                TensorProto.INT32: np.int32,
                TensorProto.UINT8: np.uint8,
                TensorProto.INT8: np.int8,
            }
            if to in mapping:
                return [ins[0].astype(mapping[to])]
    except Exception:
        return None
    return None


def fold_qat_weights(in_path: Path, out_path: Path) -> None:
    model = onnx.load(str(in_path))
    graph = model.graph

    values = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    graph_input_names = {x.name for x in graph.input}

    folded_nodes = 0
    for node in graph.node:
        outs = _eval_fold_node(node, values)
        if outs is not None and len(outs) == len(node.output):
            for name, val in zip(node.output, outs):
                values[name] = np.asarray(val)
            folded_nodes += 1

    new_initializers = []
    replaced = 0
    for node in graph.node:
        if node.op_type in ("Conv", "Gemm"):
            for idx in [1, 2]:
                if idx < len(node.input):
                    name = node.input[idx]
                    if name not in graph_input_names and name in values:
                        arr = np.asarray(values[name])
                        if arr.dtype == np.float64:
                            arr = arr.astype(np.float32)
                        new_name = clean_name(f"{node.name or node.op_type}_input{idx}_folded")
                        new_initializers.append(numpy_helper.from_array(arr, name=new_name))
                        node.input[idx] = new_name
                        replaced += 1

    graph.initializer.extend(new_initializers)

    needed = set(o.name for o in graph.output)
    kept_rev = []
    for node in reversed(graph.node):
        if any(out in needed for out in node.output):
            kept_rev.append(node)
            for inp in node.input:
                if inp:
                    needed.add(inp)
    kept_nodes = list(reversed(kept_rev))

    while len(graph.node) > 0:
        graph.node.pop()
    graph.node.extend(kept_nodes)

    needed_inputs = set()
    for node in graph.node:
        for inp in node.input:
            if inp:
                needed_inputs.add(inp)

    kept_inits = [init for init in graph.initializer if init.name in needed_inputs]
    while len(graph.initializer) > 0:
        graph.initializer.pop()
    graph.initializer.extend(kept_inits)

    clear_value_info(graph)
    onnx.checker.check_model(model)
    model = run_shape_inference(model)
    onnx.save(model, str(out_path))

    print("\n=== Stage: fold_qat_weights ===")
    print("input:", in_path)
    print("output:", out_path)
    print("constant-folded nodes:", folded_nodes)
    print("replaced Conv/Gemm weight/bias inputs:", replaced)
    print("remaining nodes:", len(graph.node))
    print("remaining initializers:", len(graph.initializer))
    print("size KiB:", out_path.stat().st_size / 1024)


def convert_to_opset11(in_path: Path, out_path: Path) -> None:
    model = onnx.load(str(in_path))
    graph = model.graph
    init_map = {i.name: i for i in graph.initializer}

    for node in graph.node:
        if node.op_type == "Unsqueeze" and len(node.input) == 2:
            axes_name = node.input[1]
            if axes_name not in init_map:
                raise SystemExit(f"Unsqueeze axes not initializer: {node.name} {axes_name}")
            axes = numpy_helper.to_array(init_map[axes_name]).astype("int64").tolist()
            while len(node.input) > 1:
                node.input.pop()
            keep_attrs = [a for a in node.attribute if a.name != "axes"]
            while len(node.attribute) > 0:
                node.attribute.pop()
            node.attribute.extend(keep_attrs)
            attr = node.attribute.add()
            attr.name = "axes"
            attr.ints.extend(axes)
            attr.type = onnx.AttributeProto.INTS

    set_input_shape(graph, [1, 1, 128, 128], clear_denotation=True)
    clear_value_info(graph)

    used = set()
    for node in graph.node:
        for name in node.input:
            if name:
                used.add(name)

    kept = [i for i in graph.initializer if i.name in used]
    while len(graph.initializer) > 0:
        graph.initializer.pop()
    graph.initializer.extend(kept)

    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            opset.version = 11

    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))

    print("\n=== Stage: convert_to_opset11 ===")
    print("wrote:", out_path)
    print("graph input:", graph.input[0].name, [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim])
    print("nodes:", len(graph.node))
    print("initializers:", len(graph.initializer))
    print("opsets:", [(o.domain, o.version) for o in model.opset_import])


def fold_all_constants(in_path: Path, out_path: Path) -> None:
    model = onnx.load(str(in_path))
    graph = model.graph

    values = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    graph_input_names = {x.name for x in graph.input}
    graph_output_names = {x.name for x in graph.output}

    foldable_outputs = set()
    folded_nodes = 0
    for node in graph.node:
        outs = _eval_fold_node(node, values)
        if outs is not None and len(outs) == len(node.output):
            for name, val in zip(node.output, outs):
                values[name] = np.asarray(val)
                foldable_outputs.add(name)
            folded_nodes += 1

    kept_nodes = []
    for node in graph.node:
        produces_graph_output = any(out in graph_output_names for out in node.output)
        all_outputs_foldable = all(out in foldable_outputs for out in node.output)
        if produces_graph_output or not all_outputs_foldable:
            kept_nodes.append(node)

    needed_inputs = set()
    for node in kept_nodes:
        for inp in node.input:
            if inp:
                needed_inputs.add(inp)

    new_initializers = []
    existing_names = set()
    for name in needed_inputs:
        if name in graph_input_names or name not in values:
            continue
        arr = np.asarray(values[name])
        if arr.shape == ():
            arr = arr.reshape(1)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        if arr.dtype == np.int32:
            arr = arr.astype(np.int64)

        safe_name = clean_name(name)
        if safe_name != name:
            for node in kept_nodes:
                for idx, inp in enumerate(node.input):
                    if inp == name:
                        node.input[idx] = safe_name
            name = safe_name

        if name not in existing_names:
            new_initializers.append(numpy_helper.from_array(arr, name=name))
            existing_names.add(name)

    while len(graph.node) > 0:
        graph.node.pop()
    graph.node.extend(kept_nodes)

    while len(graph.initializer) > 0:
        graph.initializer.pop()
    graph.initializer.extend(new_initializers)

    clear_value_info(graph)
    set_input_shape(graph, [1, 1, 128, 128], clear_denotation=True)

    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            opset.version = 11

    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))

    print("\n=== Stage: fold_all_constants ===")
    print("input:", in_path)
    print("output:", out_path)
    print("folded nodes:", folded_nodes)
    print("kept nodes:", len(graph.node))
    print("initializers:", len(graph.initializer))
    print("opsets:", [(o.domain, o.version) for o in model.opset_import])
    print("size KiB:", out_path.stat().st_size / 1024)


def parse_args() -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(
        description="Export and convert the MAX78000 KWS20 v3 QAT model into the final STM32U5 CubeAI ONNX."
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=home / "max78000/ai8x-training",
        help="Path to ai8x-training repo.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=home / "max78000/ai8x-synthesis/trained/ai85-kws20_v3-qat8.pth.tar",
        help="Path to QAT checkpoint.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=home / "max78000/stm32u5_kws20",
        help="Output directory for all intermediate and final ONNX files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_qat = args.out_dir / "kws20_v3_qat_onnx_for_stm32u5.onnx"
    out_2d = args.out_dir / "kws20_v3_qat_2d_patched_for_stm32u5.onnx"
    out_true = args.out_dir / "kws20_v3_qat_true2d_nchw_for_cubeai.onnx"
    out_folded_weights = args.out_dir / "kws20_v3_qat_true2d_nchw_folded_weights_for_cubeai.onnx"
    out_opset11 = args.out_dir / "kws20_v3_qat_true2d_nchw_folded_opset11_for_cubeai.onnx"
    out_final = args.out_dir / "kws20_v3_qat_true2d_nchw_allconst_folded_for_cubeai.onnx"

    print("\n=== Stage: export_qat_onnx ===")
    export_qat_onnx(args.repo_dir, args.checkpoint, out_qat)
    patch_qat_1d_to_2d(out_qat, out_2d)
    convert_to_true_nchw(out_2d, out_true)
    fold_qat_weights(out_true, out_folded_weights)
    convert_to_opset11(out_folded_weights, out_opset11)
    fold_all_constants(out_opset11, out_final)

    print("\nDONE")
    print("Final model:", out_final)


if __name__ == "__main__":
    main()
