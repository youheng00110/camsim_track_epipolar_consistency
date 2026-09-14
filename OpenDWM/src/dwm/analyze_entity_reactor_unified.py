from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-c", "--config-path", required=True, type=Path)
    parser.add_argument("--backend", choices=("auto", "legacy", "wan"), default="auto")
    return parser


def _detect_backend(config: dict) -> str:
    model_cfg = config.get("pipeline", {}).get("model", {})
    class_name = str(model_cfg.get("_class_name", ""))
    lowered = class_name.lower()
    if "wan" in lowered or "wantransformer" in lowered:
        return "wan"
    return "legacy"


def _strip_backend_argument(argv: list[str]) -> list[str]:
    output = []
    skip = False
    for index, value in enumerate(argv):
        if skip:
            skip = False
            continue
        if value == "--backend":
            skip = True
            continue
        if value.startswith("--backend="):
            continue
        output.append(value)
    return output


def main(argv=None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    bootstrap, _ = _bootstrap_parser().parse_known_args(raw_argv)
    config = json.loads(bootstrap.config_path.read_text(encoding="utf-8"))
    backend = bootstrap.backend
    if backend == "auto":
        backend = _detect_backend(config)
    forwarded = _strip_backend_argument(raw_argv)
    print(f"[UnifiedEntityReactor] backend={backend}", flush=True)
    if backend == "wan":
        from dwm.analysis.entity_reactor_unified_wan import main as backend_main
    else:
        try:
            from dwm.analysis.entity_reactor_unified_legacy import main as backend_main
        except ImportError as error:
            raise ImportError(
                "legacy backend requires the previous Entity Reactor core files "
                "dwm/analysis/entity_reactor.py, psi.py, tracks.py and "
                "dwm/analyze_entity_reactor.py. If this is the old baseline repo, "
                "copy the optional legacy_core files from the package first."
            ) from error
    backend_main(forwarded)


if __name__ == "__main__":
    main()
