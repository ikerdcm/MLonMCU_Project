#!/usr/bin/env python3
"""
Extract MAX78000 AI_INPUT_DUMP blocks from a UART log and generate
STM32U5-ready float headers plus a registry header for offline inference.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

KWS_INPUT_SIZE = 16384
LABEL_TO_INDEX = {
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "stop": 4,
    "go": 5,
    "yes": 6,
    "no": 7,
    "on": 8,
    "off": 9,
    "one": 10,
    "two": 11,
    "three": 12,
    "four": 13,
    "five": 14,
    "six": 15,
    "seven": 16,
    "eight": 17,
    "nine": 18,
    "zero": 19,
    "unknown": 20,
}


@dataclass
class DumpRecord:
    dump_index: int
    label: str
    values: list[int]

    @property
    def label_idx(self) -> int:
        return LABEL_TO_INDEX.get(self.label, 20)

    @property
    def symbol(self) -> str:
        return f"kws_max78000_dump_{self.dump_index:03d}_{self.label}"

    @property
    def header_name(self) -> str:
        return f"{self.symbol}.h"


def slugify_label(label: str) -> str:
    label = label.strip().lower()
    label = re.sub(r"[^a-z0-9]+", "_", label)
    label = label.strip("_")
    return label or "unknown"


def parse_log(text: str) -> list[DumpRecord]:
    token_re = re.compile(r"AI_INPUT_DUMP_BEGIN|AI_INPUT_DUMP_END|Detected word:\s*([A-Za-z0-9_]+)")
    pending_values: list[list[int]] = []
    records: list[DumpRecord] = []
    dump_index = 0
    pos = 0

    for match in token_re.finditer(text):
        token = match.group(0)
        if token == "AI_INPUT_DUMP_BEGIN":
            end_match = re.search(r"AI_INPUT_DUMP_END", text[match.end():])
            if not end_match:
                break
            body_end = match.end() + end_match.start()
            body = text[match.end():body_end]
            values = [int(v) for v in re.findall(r"-?\d+", body)]
            if len(values) == KWS_INPUT_SIZE:
                pending_values.append(values)
            pos = body_end
        elif token.startswith("Detected word:"):
            raw_label = match.group(1) or "unknown"
            label = slugify_label(raw_label)
            while pending_values:
                dump_index += 1
                records.append(DumpRecord(dump_index=dump_index,
                                          label=label,
                                          values=pending_values.pop(0)))
            pos = match.end()

    while pending_values:
        dump_index += 1
        records.append(DumpRecord(dump_index=dump_index,
                                  label="unknown",
                                  values=pending_values.pop(0)))

    return records


def write_header(out_dir: Path, source_rel: str, rec: DumpRecord) -> None:
    guard = rec.symbol.upper() + "_H"
    header = out_dir / rec.header_name
    lines = [
        f"/* Auto-generated from {source_rel}, dump #{rec.dump_index}, label '{rec.label}'. */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        f"static const float {rec.symbol}[{KWS_INPUT_SIZE}] = {{",
    ]

    row: list[str] = []
    for i, val in enumerate(rec.values, 1):
        signed_val = val - 256 if val > 127 else val
        row.append(f"{float(signed_val):.1f}f")
        if len(row) == 16 or i == len(rec.values):
            lines.append("    " + ", ".join(row) + ("," if i != len(rec.values) else ""))
            row = []

    lines += [
        "};",
        "",
        "#endif",
        "",
    ]
    header.write_text("\n".join(lines))


def write_registry(out_dir: Path, source_rel: str, records: Iterable[DumpRecord]) -> None:
    records = list(records)
    registry = out_dir / "kws_max78000_generated_vectors.h"
    lines = [
        "/* Auto-generated registry of MAX78000 AI input dumps for STM32U5 offline eval. */",
        "#ifndef KWS_MAX78000_GENERATED_VECTORS_H",
        "#define KWS_MAX78000_GENERATED_VECTORS_H",
        "",
    ]

    for rec in records:
        lines.append(f'#include "{rec.header_name}"')

    lines += [
        "",
        "#define KWS_MAX78000_GENERATED_CASES \\",
    ]

    if records:
        for idx, rec in enumerate(records):
            suffix = ", \\" if idx != len(records) - 1 else ""
            lines.append(
                f'    {{ "{rec.label} / dump #{rec.dump_index}", '
                f'"{source_rel}: AI_INPUT_DUMP #{rec.dump_index}", '
                f"{rec.symbol}, 1.0f, 0, {rec.label_idx}, {rec.label_idx} }}{suffix}"
            )
    else:
        lines.append("    /* no generated MAX78000 dumps found */")

    lines += [
        "",
        "#define KWS_MAX78000_GENERATED_CASE_COUNT " + str(len(records)),
        "",
        "#endif",
        "",
    ]
    registry.write_text("\n".join(lines))


def write_manifest(manifest_path: Path, source_rel: str, records: Iterable[DumpRecord]) -> None:
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "label", "label_idx", "header", "source"])
        for rec in records:
            writer.writerow([
                f"{rec.label} / dump #{rec.dump_index}",
                rec.label,
                rec.label_idx,
                rec.header_name,
                source_rel,
            ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-log", required=True, help="MAX78000 UART log with AI_INPUT_DUMP blocks")
    parser.add_argument("--out-inc-dir", required=True, help="Output directory for generated .h files")
    parser.add_argument("--manifest", required=True, help="CSV manifest output path")
    args = parser.parse_args()

    input_log = Path(args.input_log).resolve()
    out_inc_dir = Path(args.out_inc_dir).resolve()
    manifest = Path(args.manifest).resolve()
    out_inc_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    source_rel = input_log.as_posix()
    records = parse_log(input_log.read_text())

    for rec in records:
        write_header(out_inc_dir, source_rel, rec)

    write_registry(out_inc_dir, source_rel, records)
    write_manifest(manifest, source_rel, records)

    print(f"generated {len(records)} dump headers into {out_inc_dir}")
    print(f"registry: {out_inc_dir / 'kws_max78000_generated_vectors.h'}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
