
import argparse
import logging
import os
import shutil
import sys
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from nntool.api import NNGraph
from nntool.api.utils import quantization_options, model_settings

log = logging.getLogger("nn_profiler")

# ──────────────────────────────────────────────────────────────────────────────
# Preset quantization schemes
# ──────────────────────────────────────────────────────────────────────────────
PRESETS = {
    "sq8": {
        "graph_options": {
            "scheme": "SQ8",
            "allow_asymmetric_out": True,
        },
    },
    "ne16": {
        "graph_options": {
            "scheme": "SQ8",
            "use_ne16": True,
        },
    },
    "ne16_a16w8": {
        "graph_options": {
            "scheme": "SQ8",
            "use_ne16": True,
            "force_external_size": 16,
            "force_input_size": 16,
            "force_output_size": 16,
            "weight_bits": 8,
        },
    },
    "ne16_w4": {
        "graph_options": {
            "scheme": "SQ8",
            "use_ne16": True,
            "weight_bits": 4,
        },
    },
    "fp16": {
        "graph_options": {
            "scheme": "FLOAT",
            "float_type": "float16",
        },
    },
    "bfp16": {
        "graph_options": {
            "scheme": "FLOAT",
            "float_type": "bfloat16",
        },
    },
}

N_FAKE_SAMPLES = 1

def prepare_graph(
    model_path: str,
    graph_options: dict,
    node_options: dict = None,
    fusions: list[str] = None,
    load_tflite_quant: bool = False,
) -> NNGraph:
    """Load model, adjust order, apply fusions, fake-quantize, and return the graph."""
    # Load
    log.info("Loading model: %s", model_path)
    G = NNGraph.load_graph(model_path, load_quantization=load_tflite_quant)

    log.info("  Model name : %s", G.name)
    log.info("  Inputs     : %s", [n.name for n in G.input_nodes()])
    log.info("  Outputs    : %s", [n.name for n in G.output_nodes()])

    # Adjust order
    G.adjust_order()

    # Fusions
    fusion_names = fusions or ["scaled_match_group"]
    log.info("Applying fusions: %s", fusion_names)
    G.fusions(*fusion_names)

    # Fake quantize: collect stats from random data, then quantize
    scheme = graph_options.get("scheme", "SQ8").upper()
    q_opts = quantization_options(**graph_options)

    stats = None
    if scheme != "FLOAT" and not load_tflite_quant:
        input_nodes = G.input_nodes()
        shapes = [tuple(n.out_dims[0].shape) for n in input_nodes]

        def random_iter():
            for _ in range(N_FAKE_SAMPLES):
                yield [np.random.uniform(-1, 1, s).astype(np.float32) for s in shapes]

        log.info("Collecting fake statistics (%d random samples)", N_FAKE_SAMPLES)
        stats = G.collect_statistics(random_iter())

    n_opts = None
    if node_options:
        n_opts = {
            node_name: quantization_options(**opts)
            for node_name, opts in node_options.items()
        }

    log.info("Quantizing (scheme=%s, fake=True)", scheme)
    G.quantize(stats, graph_options=q_opts, node_options=n_opts)
    log.info("  Quantization done")

    return G


def execute_on_target(
        G: NNGraph,
        settings: dict,
        io_settings: dict = None,
        platform: str = "gvsoc",
        **extra_kwargs
    ):
    """Generate AT model, build, and run on target. Return profiling results."""
    ms = model_settings(**settings)

    # Random input tensors
    input_tensors = []
    for n in G.input_nodes():
        shape = tuple(n.out_dims[0].shape)
        input_tensors.append(np.random.uniform(-1, 1, shape).astype(np.float32))
        if n.name in io_settings:
            opts = io_settings[n.name]
            for k, v in opts.items():
                setattr(n, k, v)

    log.info("Running execute_on_target (platform=%s)", platform)
    result = G.execute_on_target(
        directory="/tmp/nn_profiler",
        input_tensors=input_tensors,
        settings=ms,
        platform=platform,
        performance=True,
        at_log=True,
        memory=True,
        **extra_kwargs,
    )
    return result
