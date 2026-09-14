from __future__ import annotations

import argparse
from pathlib import Path

from dwm.analysis.visualize import (
    ENTITY_REACTOR_VIS_VERSION,
    discover_run_roots,
    render_model_reactor,
)

ENTITY_REACTOR_REPORT_VERSION = "v17-minimal-two-figures-report-20260816"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render exactly two Entity Reactor PNGs from cached CSV/JSON analysis outputs"
    )
    parser.add_argument("input_root", type=Path)
    parser.add_argument("-o", "--output-path", required=True, type=Path)
    parser.add_argument("--preferred-sigma", type=float, default=0.6)
    return parser


def relative_run_path(run_root: Path, input_root: Path) -> Path:
    resolved_run = Path(run_root).resolve()
    resolved_input = Path(input_root).resolve()
    try:
        relative = resolved_run.relative_to(resolved_input)
    except ValueError:
        relative = Path(resolved_run.name)
    if str(relative) in ("", "."):
        relative = Path(resolved_run.name)
    return relative


def main() -> None:
    args = create_parser().parse_args()
    run_roots = discover_run_roots(args.input_root)
    args.output_path.mkdir(parents=True, exist_ok=True)
    print(
        f"[EntityReactor] report={ENTITY_REACTOR_REPORT_VERSION} "
        f"visualizer={ENTITY_REACTOR_VIS_VERSION}",
        flush=True,
    )

    for run_root in run_roots:
        relative_path = relative_run_path(run_root, args.input_root)
        output_dir = args.output_path / relative_path
        causal_path, overview_path = render_model_reactor(
            run_root,
            output_dir=output_dir,
            preferred_sigma=float(args.preferred_sigma),
        )
        print(f"[EntityReactor] causal  -> {causal_path}", flush=True)
        print(f"[EntityReactor] overview -> {overview_path}", flush=True)


if __name__ == "__main__":
    main()
