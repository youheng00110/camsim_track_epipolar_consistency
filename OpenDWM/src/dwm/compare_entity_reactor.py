from __future__ import annotations

import argparse
from pathlib import Path

from dwm.analysis.psi import discover_capture_files, load_capture
from dwm.analysis.visualize import render_gate_comparison


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline and gated Entity Reactor captures"
    )
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("gated_root", type=Path)
    parser.add_argument("-o", "--output-path", required=True, type=Path)
    parser.add_argument("--stage", type=str, default="final")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--track-hash", type=int, default=None)
    return parser


def index_captures(root: Path) -> dict[tuple[int, float], Path]:
    indexed = {}
    for path in discover_capture_files(root):
        capture = load_capture(path)
        metadata = capture["metadata"]
        key = (
            int(metadata.get("sample_index", -1)),
            round(float(metadata.get("sigma", -1.0)), 6),
        )
        indexed[key] = path
    return indexed


def main() -> None:
    args = create_parser().parse_args()
    baseline_index = index_captures(args.baseline_root)
    gated_index = index_captures(args.gated_root)
    key = (int(args.sample_index), round(float(args.sigma), 6))
    if key not in baseline_index:
        raise KeyError(f"baseline capture {key} was not found")
    if key not in gated_index:
        raise KeyError(f"gated capture {key} was not found")
    render_gate_comparison(
        baseline_index[key],
        gated_index[key],
        args.stage,
        args.output_path,
        track_hash=args.track_hash,
    )
    print(f"[EntityReactor] wrote {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
