from __future__ import annotations

import argparse
import copy
import gc
import glob
import json
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import Optional

import einops
import numpy as np
import torch

import dwm.common
from dwm.analysis.entity_reactor import (
    ENTITY_REACTOR_SELECTION_VERSION,
    EntityReactorController,
    NoCrossCameraTransitionError,
    entity_reactor_forward,
)
from dwm.analysis.tracks import NuPlanTrackResolver
from dwm.analysis.psi import (
    CaptureProbeResult,
    build_probe_events,
    build_runtime_atlas,
    describe_stages,
    evaluate_capture,
    layer_output_indices,
    merge_transport_reservoir,
    write_plot_data_cache,
)
from dwm.analysis.visualize import (
    ENTITY_REACTOR_VIS_VERSION,
    discover_run_roots,
    render_design_comparison,
    render_model_reactor,
)

ENTITY_REACTOR_ANALYSIS_VERSION = "v19.7.3-fullD-strict-resumable-analysis-20260819"
ENTITY_REACTOR_CAMERA_CAUSAL_VERSION = "v19.7.3-camera-separated-resumable-causal-20260819"
ENTITY_REACTOR_RESUME_VERSION = "v1-runtime-statistics-transport-reservoir-20260819"
AUTO_CAUSAL_SIGMA = 0.60


class ControlledModelInputBuilder:
    def __init__(self, pipeline, model_kind: str, seed: int) -> None:
        self.pipeline = pipeline
        self.model_kind = str(model_kind)
        self.seed = int(seed)

    def encode_real_video(self, batch: dict) -> torch.Tensor:
        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
        raw_images = batch["vae_images"].flatten(0, 2).to(self.pipeline.device)
        image_tensor = self.pipeline.image_processor.preprocess(raw_images)
        if self.model_kind == "bev":
            latents = self.pipeline.encode_images(image_tensor, use_mode=True)
            return einops.rearrange(
                latents,
                "(b t v) c h w -> b t v c h w",
                b=batch_size,
                t=sequence_length,
                v=view_count,
            )

        shift_factor = self.pipeline.vae.config.shift_factor
        if shift_factor is None:
            shift_factor = 0.0
        if getattr(self.pipeline, "is_temporal_vae", False):
            image_tensor = einops.rearrange(
                image_tensor,
                "(b t v) c h w -> (b v) c t h w",
                b=batch_size,
                t=sequence_length,
                v=view_count,
            )
        memory_batch = int(self.pipeline.common_config.get("memory_efficient_batch", 12))
        latent_chunks = []
        for start in range(0, image_tensor.shape[0], memory_batch):
            stop = min(start + memory_batch, image_tensor.shape[0])
            current_images = image_tensor[start:stop].to(
                device=self.pipeline.device,
                dtype=self.pipeline.vae.dtype,
            )
            posterior = self.pipeline.vae.encode(current_images).latent_dist
            current_latents = (
                posterior.mode() - shift_factor
            ) * self.pipeline.vae.config.scaling_factor
            latent_chunks.append(current_latents)
        latents = torch.cat(latent_chunks, dim=0)
        if getattr(self.pipeline, "is_temporal_vae", False):
            return einops.rearrange(
                latents,
                "(b v) c t h w -> b t v c h w",
                b=batch_size,
                v=view_count,
            )
        return einops.rearrange(
            latents,
            "(b t v) c h w -> b t v c h w",
            b=batch_size,
            t=sequence_length,
            v=view_count,
        )

    def make_noise(self, latent_shape: torch.Size, sample_index: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(sample_index) * 1009)
        noise = torch.randn(latent_shape, generator=generator)
        return noise.to(self.pipeline.device)

    def timestep_for_sigma(self, sigma: float) -> torch.Tensor:
        scheduler = self.pipeline.train_scheduler
        scheduler_sigmas = scheduler.sigmas.detach().float().cpu()
        scheduler_timesteps = scheduler.timesteps.detach().float().cpu()
        usable_count = min(scheduler_sigmas.shape[0], scheduler_timesteps.shape[0])
        distance = torch.abs(scheduler_sigmas[:usable_count] - float(sigma))
        index = int(torch.argmin(distance).item())
        timestep = scheduler_timesteps[index].to(self.pipeline.device)
        if not torch.isfinite(timestep):
            raise RuntimeError(f"scheduler returned an invalid timestep for sigma={sigma}")
        return timestep

    def decode_generated_video(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, view_count = latents.shape[:3]
        if self.model_kind == "bev":
            decoded = self.pipeline.decode_latents(latents.flatten(0, 2))
            images = self.pipeline.image_processor.postprocess(decoded, output_type="pt")
            return images.unflatten(0, (batch_size, sequence_length, view_count))

        shift_factor = self.pipeline.vae.config.shift_factor
        if shift_factor is None:
            shift_factor = 0.0
        is_temporal_vae = bool(getattr(self.pipeline, "is_temporal_vae", False))
        if is_temporal_vae:
            decode_input = einops.rearrange(latents, "b t v c h w -> (b v) c t h w")
        else:
            decode_input = latents.flatten(0, 2)
        memory_batch = int(self.pipeline.common_config.get("memory_efficient_batch", 12))
        decoded_chunks = []
        for start in range(0, decode_input.shape[0], memory_batch):
            stop = min(start + memory_batch, decode_input.shape[0])
            current = decode_input[start:stop].to(
                device=self.pipeline.device,
                dtype=self.pipeline.vae.dtype,
            )
            decoded = self.pipeline.vae.decode(
                current / self.pipeline.vae.config.scaling_factor + shift_factor,
                return_dict=False,
            )[0]
            decoded_chunks.append(decoded)
        decoded_tensor = torch.cat(decoded_chunks, dim=0)
        if is_temporal_vae:
            decoded_tensor = einops.rearrange(
                decoded_tensor,
                "(b v) c t h w -> (b t v) c h w",
                b=batch_size,
                v=view_count,
            )
        images = self.pipeline.image_processor.postprocess(decoded_tensor, output_type="pt")
        return images.unflatten(0, (batch_size, sequence_length, view_count))

    def prepare_conditions(self, batch: dict, latents: torch.Tensor) -> dict:
        if self.model_kind == "bev":
            condition_keep = torch.ones(latents.shape[0], device="cpu", dtype=torch.bool)
            model_conditions = self.pipeline.prepare_model_conditions(
                batch,
                latents.shape,
                condition_keep=condition_keep,
            )
            model_conditions["disable_temporal"] = torch.zeros(
                latents.shape[0],
                1,
                1,
                device=self.pipeline.device,
                dtype=torch.bool,
            )
            return model_conditions

        model_conditions = self.pipeline.get_conditions(
            self.pipeline.model,
            self.pipeline.text_encoders,
            self.pipeline.tokenizers,
            self.pipeline.common_config,
            batch["vae_images"].shape,
            batch,
            self.pipeline.device,
            self.pipeline.model_dtype,
            do_classifier_free_guidance=False,
            latents_shape=latents.shape,
        )
        return model_conditions


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-path", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help=(
            "Optional checkpoint file, checkpoint directory, output directory, or glob. "
            "If omitted, pipeline.model_checkpoint_path from the config is used."
        ),
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Analyze every checkpoint when --checkpoint resolves to multiple files.",
    )
    parser.add_argument("-o", "--output-path", required=True, type=Path)
    parser.add_argument(
        "--model-kind",
        choices=("auto", "bev", "pv"),
        default="auto",
        help="Backward-compatible option. The model kind is detected from the config.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Backward-compatible no-op. Rendering remains enabled by default.",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Optionally save one diagnostic-noise entity crop set. Disabled by default to reduce storage.",
    )
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-indices", type=int, nargs="*", default=None)
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=(0.1, 0.3, 0.6, 0.9),
        help="Gen4U-style noise sweep. The unchanged launch command uses these defaults.",
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=32,
        help="Backward-compatible no-op. Quantitative analysis always uses full hidden dimension.",
    )
    parser.add_argument(
        "--projection-seed",
        type=int,
        default=20260815,
        help="Backward-compatible no-op. No random projection is applied.",
    )
    parser.add_argument("--analysis-seed", type=int, default=3107)
    parser.add_argument("--max-entities", type=int, default=32)
    parser.add_argument("--minimum-track-length", type=int, default=3)
    parser.add_argument("--entity-classes", nargs="+", default=("car", "vehicle", "truck", "bus"))
    parser.add_argument("--capture-pattern", action="append", default=None)
    parser.add_argument("--gate", action="append", default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-save-crops", action="store_true")
    return parser


def parse_gate_spec(values: Optional[list[str]]) -> dict[str, float]:
    gate_spec: dict[str, float] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"gate must use PATTERN=ALPHA, got {value}")
        pattern, alpha_text = value.rsplit("=", 1)
        alpha = float(alpha_text)
        if not torch.isfinite(torch.tensor(alpha)):
            raise ValueError(f"gate alpha is not finite for {value}")
        gate_spec[pattern.strip()] = alpha
    return gate_spec


def _checkpoint_sort_value(path: Path) -> tuple[int, str]:
    numeric_tokens = re.findall(r"\d+", path.stem)
    numeric_value = int(numeric_tokens[-1]) if numeric_tokens else -1
    resolved_text = str(path.resolve())
    return numeric_value, resolved_text


def checkpoint_output_name(checkpoint_path: Path) -> str:
    checkpoint_path = Path(checkpoint_path)
    candidates = [checkpoint_path.stem]
    candidates.extend(parent.name for parent in checkpoint_path.parents[:3])
    patterns = (
        re.compile(r"^(?:checkpoint|ckpt)[-_]?(\d+)$", re.IGNORECASE),
        re.compile(r"^(?:step)[-_]?(\d+)$", re.IGNORECASE),
    )
    for candidate in candidates:
        for pattern in patterns:
            match = pattern.match(candidate)
            if match is not None:
                return f"checkpoint-{int(match.group(1))}"
    numeric_tokens = re.findall(r"\d+", checkpoint_path.stem)
    if numeric_tokens:
        return f"checkpoint-{int(numeric_tokens[-1])}"
    return checkpoint_path.stem


def resolve_checkpoint_paths(config: dict, requested: Optional[str], analyze_all: bool) -> list[Path]:
    source = requested
    if source is None:
        source = config.get("pipeline", {}).get("model_checkpoint_path", None)
    if source is None or str(source).strip() == "":
        raise ValueError("no checkpoint was provided and the config checkpoint path is empty")

    source_text = os.path.expanduser(os.path.expandvars(str(source)))
    wildcard = any(mark in source_text for mark in ("*", "?", "["))
    candidates: list[Path] = []
    if wildcard:
        candidates = [Path(item) for item in glob.glob(source_text)]
    else:
        source_path = Path(source_text)
        if source_path.is_file():
            candidates = [source_path]
        elif source_path.is_dir():
            search_dir = source_path / "checkpoints" if (source_path / "checkpoints").is_dir() else source_path
            candidates = list(search_dir.glob("*.pth")) + list(search_dir.glob("*.safetensors"))
        else:
            raise FileNotFoundError(f"checkpoint source does not exist {source_path}")

    candidates = [path.resolve() for path in candidates if path.is_file()]
    candidates = sorted(set(candidates), key=_checkpoint_sort_value)
    if not candidates:
        raise FileNotFoundError(f"no checkpoint file was found from {source_text}")
    return candidates if analyze_all else [candidates[-1]]


def detect_model_kind(config: dict) -> str:
    pipeline_class = str(config["pipeline"].get("_class_name", "")).lower()
    model_class = str(config["pipeline"]["model"].get("_class_name", "")).lower()
    is_bev = "bevpipeline" in pipeline_class or "bevconditioned" in model_class
    if not model_class:
        raise KeyError("config pipeline.model._class_name is missing")
    return "bev" if is_bev else "pv"


def setup_distributed(config: dict) -> torch.device:
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(config["device"], local_rank)
        if config["device"] == "cuda":
            torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend=config["ddp_backend"])
        return device
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(0)
    return device


def initialize_global_state(config: dict) -> None:
    global_state_config = config.get("global_state", {})
    for key, value in global_state_config.items():
        instance = dwm.common.create_instance_from_config(value)
        dwm.common.global_state[key] = instance
    if global_state_config and len(dwm.common.global_state) == 0:
        raise RuntimeError("global_state configuration was not initialized")


def instantiate_pipeline(
    base_config: dict,
    checkpoint_path: Path,
    output_path: Path,
    device: torch.device,
):
    config = copy.deepcopy(base_config)
    config["pipeline"]["model_checkpoint_path"] = str(checkpoint_path)
    if "metrics" in config["pipeline"]:
        config["pipeline"]["metrics"] = {}
    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"],
        output_path=str(output_path),
        config=config,
        device=device,
    )
    pipeline.model.eval()
    if pipeline.model.training:
        raise RuntimeError("analysis pipeline model must be in eval mode")
    return pipeline, config


def _search_config_values(config: object, interesting_keys: set[str], output: dict) -> None:
    pending = [config]
    visited = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        if isinstance(current, dict):
            for key, value in current.items():
                if key in interesting_keys and key not in output:
                    if isinstance(value, (str, int, float, bool, list, tuple)):
                        output[key] = value
                if isinstance(value, (dict, list, tuple)):
                    pending.append(value)
        elif isinstance(current, (list, tuple)):
            for value in current:
                if isinstance(value, (dict, list, tuple)):
                    pending.append(value)


def summarize_design_config(config: dict, setting_name: str) -> dict:
    interesting_keys = {
        "perspective_modeling_type",
        "enable_temporal",
        "enable_crossview",
        "crossview_block_layers",
        "temporal_block_layers",
        "block_layers",
        "crossview_attention_type",
        "condition_type",
        "condition_image_adapter",
    }
    summary: dict = {"setting_name": str(setting_name)}
    _search_config_values(config.get("pipeline", {}).get("model", {}), interesting_keys, summary)
    summary["model_class"] = str(config["pipeline"]["model"].get("_class_name", "unknown"))
    summary["pipeline_class"] = str(config["pipeline"].get("_class_name", "unknown"))
    return summary


def _resolve_sample_indices(args, dataset_length: int) -> list[int]:
    if args.sample_indices:
        sample_indices = [int(index) for index in args.sample_indices]
    else:
        if args.sample_count <= 0 or args.sample_stride <= 0:
            raise ValueError("sample count and sample stride must be positive")
        sample_indices = [
            int(args.start_index) + position * int(args.sample_stride)
            for position in range(int(args.sample_count))
        ]
    invalid = [index for index in sample_indices if index < 0 or index >= dataset_length]
    if invalid:
        raise IndexError(
            f"analysis sample indices exceed validation dataset, invalid={invalid[:10]}, length={dataset_length}"
        )
    return sample_indices


def _resolve_intervention_save_stages(
    stage_names: list[str],
    intervention: dict,
    num_layers: int,
) -> list[str]:
    stage_names = [str(name) for name in stage_names]
    if not stage_names:
        raise RuntimeError("cannot resolve causal stages from an empty baseline")
    descriptors = describe_stages(stage_names, int(num_layers))
    output_indices = layer_output_indices(descriptors)
    intervention_layer = int(intervention["layer"])
    downstream = [
        index
        for index in output_indices
        if descriptors[index].module == "final"
        or descriptors[index].layer > intervention_layer
    ]
    if not downstream:
        raise RuntimeError(
            f"cannot find a downstream output stage for {intervention['label']}"
        )
    next_stage_name = stage_names[int(downstream[0])]
    final_stage_name = stage_names[int(output_indices[-1])]
    save_stage_names = [next_stage_name]
    if final_stage_name != next_stage_name:
        save_stage_names.append(final_stage_name)
    return save_stage_names


def _validate_pv_conditions(latents: torch.Tensor, model_conditions: dict) -> None:
    expected_btv = tuple(int(value) for value in latents.shape[:3])
    pooled_shape = tuple(model_conditions["pooled_projections"].shape)
    text_shape = tuple(model_conditions["encoder_hidden_states"].shape)
    if pooled_shape[:3] != expected_btv:
        raise RuntimeError(
            f"PV pooled_projections must start with BTV={expected_btv}, got {pooled_shape}"
        )
    if text_shape[:3] != expected_btv:
        raise RuntimeError(
            f"PV encoder_hidden_states must start with BTV={expected_btv}, got {text_shape}"
        )


def _probe_result_to_resume_payload(result: CaptureProbeResult) -> dict:
    return {
        "sample_index": int(result.sample_index),
        "stage_names": [str(value) for value in result.stage_names],
        "metrics": {
            str(name): np.asarray(values, dtype=np.float64)
            for name, values in result.metrics.items()
        },
        "event_counts": {
            str(name): int(value)
            for name, value in result.event_counts.items()
        },
    }


def _probe_result_from_resume_payload(payload: dict) -> CaptureProbeResult:
    return CaptureProbeResult(
        capture_path=Path("<resume-statistics>"),
        sample_index=int(payload["sample_index"]),
        stage_names=[str(value) for value in payload["stage_names"]],
        metrics={
            str(name): np.asarray(values, dtype=np.float64)
            for name, values in payload["metrics"].items()
        },
        event_counts={
            str(name): int(value)
            for name, value in payload["event_counts"].items()
        },
    )


def _write_pickle_atomic(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)
    return path


def _read_pickle(path: Path) -> dict | None:
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    try:
        with path.open("rb") as file:
            payload = pickle.load(file)
    except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)
    return path


def _make_resume_signature(
    checkpoint_path: Path,
    dataset_length: int,
    sigmas: list[float],
    target_valid_count: int,
    args,
    intervention_plan: list[dict],
) -> dict:
    return {
        "resume_version": ENTITY_REACTOR_RESUME_VERSION,
        "analysis_version": ENTITY_REACTOR_ANALYSIS_VERSION,
        "selection_version": ENTITY_REACTOR_SELECTION_VERSION,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "dataset_length": int(dataset_length),
        "sigmas": [float(value) for value in sigmas],
        "target_valid_count": int(target_valid_count),
        "start_index": int(args.start_index),
        "sample_stride": int(args.sample_stride),
        "explicit_sample_indices": (
            [int(value) for value in args.sample_indices]
            if args.sample_indices
            else None
        ),
        "intervention_labels": [str(value["label"]) for value in intervention_plan],
    }


def _write_runtime_resume_state(
    state_path: Path,
    signature: dict,
    candidate_indices: list[int],
    next_backfill_index: int,
    rejected_samples: dict[int, str],
    accepted_sample_indices: list[int],
    first_metadata: Optional[dict],
    runtime_baselines: dict[float, dict[int, CaptureProbeResult]],
    transport_reservoirs: dict[tuple[float, int], dict[str, np.ndarray]],
) -> Path:
    baseline_payload = {
        float(sigma): {
            int(sample_index): _probe_result_to_resume_payload(result)
            for sample_index, result in per_sample.items()
        }
        for sigma, per_sample in runtime_baselines.items()
    }
    transport_payload = {
        (float(key[0]), int(key[1])): {
            str(name): np.asarray(values)
            for name, values in payload.items()
        }
        for key, payload in transport_reservoirs.items()
    }
    payload = {
        "signature": dict(signature),
        "candidate_indices": [int(value) for value in candidate_indices],
        "next_backfill_index": int(next_backfill_index),
        "rejected_samples": {int(key): str(value) for key, value in rejected_samples.items()},
        "accepted_sample_indices": [int(value) for value in accepted_sample_indices],
        "first_metadata": None if first_metadata is None else dict(first_metadata),
        "runtime_baselines": baseline_payload,
        "transport_reservoirs": transport_payload,
    }
    return _write_pickle_atomic(state_path, payload)


def _load_runtime_resume_state(state_path: Path, signature: dict) -> dict | None:
    payload = _read_pickle(state_path)
    if payload is None:
        return None
    if payload.get("signature") != signature:
        raise RuntimeError(
            f"resume state {state_path} belongs to a different analysis configuration; "
            "move or delete the _resume directory before starting a new configuration"
        )
    runtime_baselines = {
        float(sigma): {
            int(sample_index): _probe_result_from_resume_payload(result_payload)
            for sample_index, result_payload in per_sample.items()
        }
        for sigma, per_sample in payload.get("runtime_baselines", {}).items()
    }
    transport_reservoirs = {
        (float(key[0]), int(key[1])): {
            str(name): np.asarray(values)
            for name, values in reservoir.items()
        }
        for key, reservoir in payload.get("transport_reservoirs", {}).items()
    }
    payload["runtime_baselines"] = runtime_baselines
    payload["transport_reservoirs"] = transport_reservoirs
    return payload


def _write_resume_progress(
    progress_path: Path,
    signature: dict,
    status: str,
    candidate_indices: list[int],
    next_backfill_index: int,
    rejected_samples: dict[int, str],
    accepted_sample_indices: list[int],
) -> Path:
    payload = {
        "signature": dict(signature),
        "status": str(status),
        "candidate_indices": [int(value) for value in candidate_indices],
        "next_backfill_index": int(next_backfill_index),
        "rejected_samples": {str(int(key)): str(value) for key, value in rejected_samples.items()},
        "accepted_sample_indices": [int(value) for value in accepted_sample_indices],
    }
    return _write_json_atomic(progress_path, payload)


def _load_resume_progress(progress_path: Path, signature: dict) -> dict | None:
    progress_path = Path(progress_path)
    if not progress_path.is_file():
        return None
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if payload.get("signature") != signature:
        raise RuntimeError(
            f"resume progress {progress_path} belongs to a different analysis configuration; "
            "move or delete the _resume directory before starting a new configuration"
        )
    payload["rejected_samples"] = {
        int(key): str(value)
        for key, value in payload.get("rejected_samples", {}).items()
    }
    return payload


def _intervention_resume_path(resume_root: Path, sample_index: int, label: str) -> Path:
    return (
        Path(resume_root)
        / "interventions"
        / f"sample_{int(sample_index):05d}"
        / f"{str(label)}.pkl"
    )


def _write_intervention_resume(
    path: Path,
    signature: dict,
    label: str,
    metadata: dict,
    sample_index: int,
    result: CaptureProbeResult,
) -> Path:
    payload = {
        "signature": dict(signature),
        "label": str(label),
        "metadata": dict(metadata),
        "sample_index": int(sample_index),
        "result": _probe_result_to_resume_payload(result),
    }
    return _write_pickle_atomic(path, payload)


def _load_runtime_interventions(
    resume_root: Path,
    signature: dict,
    intervention_plan: list[dict],
) -> dict[str, dict]:
    runtime_interventions: dict[str, dict] = {}
    valid_labels = {str(value["label"]) for value in intervention_plan}
    intervention_root = Path(resume_root) / "interventions"
    if not intervention_root.is_dir():
        return runtime_interventions
    for path in sorted(intervention_root.glob("sample_*/*.pkl")):
        payload = _read_pickle(path)
        if payload is None or payload.get("signature") != signature:
            continue
        label = str(payload.get("label", ""))
        if label not in valid_labels:
            continue
        sample_index = int(payload.get("sample_index", -1))
        if sample_index < 0:
            continue
        result_payload = payload.get("result")
        if not isinstance(result_payload, dict):
            continue
        entry = runtime_interventions.setdefault(
            label,
            {"metadata": dict(payload.get("metadata", {})), "results": {}},
        )
        entry["results"][sample_index] = _probe_result_from_resume_payload(result_payload)
    return runtime_interventions


def _intervention_sample_is_complete(
    runtime_interventions: dict[str, dict],
    intervention_plan: list[dict],
    sample_index: int,
) -> bool:
    for intervention in intervention_plan:
        label = str(intervention["label"])
        if int(sample_index) not in runtime_interventions.get(label, {}).get("results", {}):
            return False
    return True


def _final_plot_cache_is_current(run_root: Path) -> bool:
    plot_path = Path(run_root) / "relation_plot_data.npz"
    if not plot_path.is_file() or plot_path.stat().st_size <= 0:
        return False
    try:
        with np.load(plot_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            feature_policy = str(metadata.get("feature_dimension_policy", ""))
            analysis_version = str(metadata.get("entity_reactor_analysis_version", ""))
    except (OSError, ValueError, KeyError, json.JSONDecodeError, EOFError):
        return False
    if analysis_version != ENTITY_REACTOR_ANALYSIS_VERSION:
        return False
    return feature_policy in ("", "full_hidden_dimension")


def _append_backfill_sample(
    sample_indices: list[int],
    seen_indices: set[int],
    next_candidate: int,
    sample_stride: int,
    dataset_length: int,
) -> int:
    candidate = int(next_candidate)
    stride = int(sample_stride)
    while candidate < int(dataset_length) and candidate in seen_indices:
        candidate += stride
    if candidate < 0 or candidate >= int(dataset_length):
        raise RuntimeError(
            "validation dataset ended before enough strict same-class clips were collected"
        )
    sample_indices.append(candidate)
    seen_indices.add(candidate)
    return candidate + stride

def run_checkpoint(
    args,
    base_config: dict,
    validation_dataset,
    track_resolver,
    checkpoint_path: Path,
    device: torch.device,
) -> Path:
    pipeline, config = instantiate_pipeline(
        base_config,
        checkpoint_path,
        args.output_path,
        device,
    )
    model_kind = detect_model_kind(config)
    input_builder = ControlledModelInputBuilder(pipeline, model_kind, args.analysis_seed)
    baseline_gate_spec = parse_gate_spec(args.gate)
    controller = EntityReactorController(
        capture_patterns=args.capture_pattern,
        gate_spec=baseline_gate_spec,
        projection_dim=args.projection_dim,
        projection_seed=args.projection_seed,
        max_cross_camera_tracks=int(args.max_entities),
    )
    controller.attach(pipeline)

    initial_sample_indices = _resolve_sample_indices(args, len(validation_dataset))
    target_valid_count = len(initial_sample_indices)
    allow_backfill = args.sample_indices is None
    sigmas = sorted(set(float(value) for value in args.sigmas))
    if not sigmas or min(sigmas) < 0.0 or max(sigmas) > 1.0:
        raise ValueError("sigmas must be unique values inside [0,1]")
    causal_sigma = min(sigmas, key=lambda value: abs(value - AUTO_CAUSAL_SIGMA))

    intervention_plan = []
    seen_interventions = set()
    module_layers = (
        ("camera", controller.camera_layers, "cam"),
        ("condition", controller.condition_layers, "cond"),
        ("temporal", controller.temporal_layers, "temp"),
        ("crossview", controller.crossview_layers, "view"),
    )
    for module_name, layer_values, stage_suffix in module_layers:
        for layer_index in sorted(set(int(value) for value in layer_values)):
            if layer_index not in controller.key_layers:
                continue
            key = (module_name, layer_index)
            if key in seen_interventions:
                continue
            seen_interventions.add(key)
            intervention_plan.append(
                {
                    "module": module_name,
                    "layer": layer_index,
                    "pattern": f"L{layer_index:02d}.{stage_suffix}",
                    "label": f"{module_name}_L{layer_index:02d}",
                }
            )
    automatic_interventions = len(baseline_gate_spec) == 0 and len(intervention_plan) > 0

    model_name = config["pipeline"]["model"]["_class_name"].split(".")[-1]
    checkpoint_name = checkpoint_output_name(checkpoint_path)
    setting_name = args.output_path.name
    design_summary = summarize_design_config(config, setting_name)
    run_root = args.output_path / checkpoint_name
    run_root.mkdir(parents=True, exist_ok=True)
    resume_root = run_root / "_resume"
    runtime_state_path = resume_root / "runtime_state.pkl"
    progress_path = resume_root / "progress.json"

    resume_signature = _make_resume_signature(
        checkpoint_path,
        len(validation_dataset),
        sigmas,
        target_valid_count,
        args,
        intervention_plan if automatic_interventions else [],
    )
    completed_plot_cache = run_root / "relation_plot_data.npz"
    progress_payload = _load_resume_progress(progress_path, resume_signature)

    runtime_baselines: dict[float, dict[int, CaptureProbeResult]] = {sigma: {} for sigma in sigmas}
    transport_reservoirs: dict[tuple[float, int], dict[str, np.ndarray]] = {}
    accepted_sample_indices: list[int] = []
    rejected_samples: dict[int, str] = {}
    first_metadata: Optional[dict] = None
    candidate_indices = [int(value) for value in initial_sample_indices]
    next_backfill_index = (
        int(args.start_index) + int(args.sample_count) * int(args.sample_stride)
        if allow_backfill
        else len(validation_dataset)
    )

    state_payload = _load_runtime_resume_state(runtime_state_path, resume_signature)
    if state_payload is not None:
        runtime_baselines = {
            float(sigma): dict(state_payload["runtime_baselines"].get(float(sigma), {}))
            for sigma in sigmas
        }
        transport_reservoirs = dict(state_payload.get("transport_reservoirs", {}))
        accepted_sample_indices = [
            int(value) for value in state_payload.get("accepted_sample_indices", [])
        ]
        rejected_samples = {
            int(key): str(value)
            for key, value in state_payload.get("rejected_samples", {}).items()
        }
        first_metadata = state_payload.get("first_metadata")
        candidate_indices = [
            int(value) for value in state_payload.get("candidate_indices", candidate_indices)
        ]
        next_backfill_index = int(
            state_payload.get("next_backfill_index", next_backfill_index)
        )

    if progress_payload is not None:
        candidate_indices = [
            int(value) for value in progress_payload.get("candidate_indices", candidate_indices)
        ]
        next_backfill_index = int(
            progress_payload.get("next_backfill_index", next_backfill_index)
        )
        rejected_samples.update(progress_payload.get("rejected_samples", {}))

    runtime_interventions = (
        _load_runtime_interventions(resume_root, resume_signature, intervention_plan)
        if automatic_interventions
        else {}
    )
    accepted_set = set(accepted_sample_indices)
    seen_indices = set(candidate_indices)

    print(
        f"[EntityReactor] {ENTITY_REACTOR_ANALYSIS_VERSION} / "
        f"{ENTITY_REACTOR_CAMERA_CAUSAL_VERSION} / {ENTITY_REACTOR_VIS_VERSION}",
        flush=True,
    )
    print(
        "[EntityReactor] full-D analysis, strict same-class negatives, "
        f"target_valid_clips={target_valid_count}",
        flush=True,
    )
    if accepted_sample_indices or runtime_interventions:
        completed_interventions = sum(
            len(payload.get("results", {}))
            for payload in runtime_interventions.values()
        )
        print(
            f"[EntityReactor] resume loaded baseline_clips={len(accepted_sample_indices)} "
            f"causal_results={completed_interventions}",
            flush=True,
        )

    try:
        if _final_plot_cache_is_current(run_root):
            print(
                f"[EntityReactor] completed cache already exists -> {completed_plot_cache}",
                flush=True,
            )
            if not args.no_render:
                causal_path, overview_path = render_model_reactor(run_root)
                print(f"[EntityReactor] causal figure -> {causal_path}", flush=True)
                print(f"[EntityReactor] overview figure -> {overview_path}", flush=True)
            return run_root

        candidate_position = 0
        while candidate_position < len(candidate_indices):
            sample_index = int(candidate_indices[candidate_position])
            candidate_position += 1
            sample_is_accepted = sample_index in accepted_set
            if not sample_is_accepted and len(accepted_sample_indices) >= target_valid_count:
                break
            if not sample_is_accepted and sample_index in rejected_samples:
                continue
            causal_complete = (
                not automatic_interventions
                or _intervention_sample_is_complete(
                    runtime_interventions,
                    intervention_plan,
                    sample_index,
                )
            )
            if sample_is_accepted and causal_complete:
                continue

            print(
                f"[EntityReactor] candidate sample={sample_index} "
                f"accepted={len(accepted_sample_indices)}/{target_valid_count} "
                f"resume_baseline={int(sample_is_accepted)}",
                flush=True,
            )

            try:
                item = validation_dataset[sample_index]
                batch = torch.utils.data.default_collate([item])
                if "clip_text" in item:
                    batch["clip_text"] = [item["clip_text"]]
                track_data = track_resolver.resolve(sample_index)
                if "lidar_to_camera" not in batch:
                    batch["lidar_to_camera"] = track_data["lidar_to_camera"].unsqueeze(0)
                with torch.no_grad():
                    latents = input_builder.encode_real_video(batch)
                    fixed_noise = input_builder.make_noise(latents.shape, sample_index)
                    model_conditions = input_builder.prepare_conditions(batch, latents)
                if model_kind == "pv":
                    _validate_pv_conditions(latents, model_conditions)
            except RuntimeError as error:
                print(
                    f"[EntityReactor] candidate sample={sample_index} failed before eligibility: {error}",
                    flush=True,
                )
                raise

            patch_size = int(pipeline.model.config.patch_size)
            token_height = int(latents.shape[-2] // patch_size)
            token_width = int(latents.shape[-1] // patch_size)
            eligibility_metadata = {
                "model_name": model_name,
                "checkpoint_name": checkpoint_name,
                "checkpoint_path": str(checkpoint_path),
                "sample_index": sample_index,
                "sigma": float(causal_sigma),
                "sequence_length": int(latents.shape[1]),
                "view_count": int(latents.shape[2]),
                "scene": track_data.get("scene", ""),
                "track_tokens": track_data.get("track_tokens", []),
                "entity_reactor_analysis_version": ENTITY_REACTOR_ANALYSIS_VERSION,
                "capture_role": "eligibility",
            }
            eligibility_metadata.update(design_summary)

            if not sample_is_accepted:
                rejected_reason = None
                try:
                    controller.recorder.configure(
                        batch,
                        track_data,
                        token_height=token_height,
                        token_width=token_width,
                        metadata=eligibility_metadata,
                    )
                    eligibility_capture = controller.recorder.build_probe_capture()
                    eligibility_events = build_probe_events(eligibility_capture)
                    if len(eligibility_events["cross_view"]) == 0:
                        rejected_reason = (
                            "no cross-view event has a same-class negative under strict policy"
                        )
                except NoCrossCameraTransitionError as error:
                    rejected_reason = str(error)

                if rejected_reason is not None:
                    rejected_samples[sample_index] = str(rejected_reason)
                    print(
                        f"[EntityReactor] reject sample={sample_index} reason={rejected_reason}",
                        flush=True,
                    )
                    if allow_backfill:
                        next_backfill_index = _append_backfill_sample(
                            candidate_indices,
                            seen_indices,
                            next_backfill_index,
                            int(args.sample_stride),
                            len(validation_dataset),
                        )
                        appended_index = int(candidate_indices[-1])
                        print(
                            f"[EntityReactor] backfill append sample={appended_index}",
                            flush=True,
                        )
                    _write_resume_progress(
                        progress_path,
                        resume_signature,
                        "running",
                        candidate_indices,
                        next_backfill_index,
                        rejected_samples,
                        accepted_sample_indices,
                    )
                    del batch, latents, fixed_noise, model_conditions, track_data
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                sample_results: dict[float, CaptureProbeResult] = {}
                pending_transport: dict[tuple[float, int], dict[str, np.ndarray]] = {}
                baseline_stage_names: Optional[list[str]] = None
                baseline_causal_result: Optional[CaptureProbeResult] = None

                for sigma in sigmas:
                    logical_path = (
                        run_root
                        / f"sample_{sample_index:05d}"
                        / f"sigma_{int(round(sigma * 1000)):04d}"
                        / "memory_only"
                    )
                    metadata = dict(eligibility_metadata)
                    metadata.update(
                        {
                            "sigma": sigma,
                            "capture_role": "baseline",
                            "storage_policy": "runtime_fullD_resume_statistics_only",
                        }
                    )
                    controller.gate_spec = dict(baseline_gate_spec)
                    controller.configure_capture(
                        batch,
                        track_data,
                        token_height,
                        token_width,
                        logical_path,
                        metadata=metadata,
                    )
                    noisy_latents = sigma * fixed_noise + (1.0 - sigma) * latents
                    timestep_value = input_builder.timestep_for_sigma(sigma)
                    timesteps = timestep_value.expand(*noisy_latents.shape[:3]).clone()
                    with torch.no_grad():
                        model_result = entity_reactor_forward(
                            pipeline,
                            noisy_latents.to(pipeline.model_dtype),
                            timesteps,
                            model_conditions,
                        )
                    capture = controller.latest_capture
                    if capture is None:
                        raise RuntimeError(
                            "forward finished without an in-memory Entity Reactor snapshot"
                        )
                    descriptors = describe_stages(capture["stage_names"], controller.num_layers)
                    output_stage_indices = set(layer_output_indices(descriptors))
                    result, transport_payloads = evaluate_capture(
                        capture,
                        transport_stage_indices=output_stage_indices,
                    )
                    sample_results[float(sigma)] = result
                    baseline_stage_names = list(result.stage_names)
                    if abs(float(sigma) - float(causal_sigma)) < 1e-8:
                        baseline_causal_result = result
                    for stage_index in sorted(output_stage_indices):
                        payload = transport_payloads["cross_view"][stage_index]
                        if payload is None or payload["feature_delta"].shape[0] == 0:
                            continue
                        pending_transport[(float(sigma), int(stage_index))] = payload
                    del model_result, capture, transport_payloads, noisy_latents, timesteps
                    controller.latest_capture = None
                    controller.recorder.stage_features.clear()
                    print(
                        f"[EntityReactor] baseline sample={sample_index} sigma={sigma:.1f} "
                        "full-D -> memory metrics",
                        flush=True,
                    )

                if len(sample_results) != len(sigmas):
                    raise RuntimeError(
                        f"sample {sample_index} did not complete every requested sigma"
                    )
                for sigma, result in sample_results.items():
                    runtime_baselines[float(sigma)][sample_index] = result
                for reservoir_key, payload in pending_transport.items():
                    sigma_value = float(reservoir_key[0])
                    transport_reservoirs[reservoir_key] = merge_transport_reservoir(
                        transport_reservoirs.get(reservoir_key),
                        payload,
                        maximum_events=768,
                        seed=(
                            int(args.analysis_seed)
                            + int(round(sigma_value * 1000)) * 1009
                        ),
                    )

                if first_metadata is None:
                    first_metadata = dict(controller.recorder.metadata)
                accepted_sample_indices.append(sample_index)
                accepted_set.add(sample_index)
                _write_runtime_resume_state(
                    runtime_state_path,
                    resume_signature,
                    candidate_indices,
                    next_backfill_index,
                    rejected_samples,
                    accepted_sample_indices,
                    first_metadata,
                    runtime_baselines,
                    transport_reservoirs,
                )
                _write_resume_progress(
                    progress_path,
                    resume_signature,
                    "running",
                    candidate_indices,
                    next_backfill_index,
                    rejected_samples,
                    accepted_sample_indices,
                )
                print(
                    f"[EntityReactor] accept sample={sample_index} "
                    f"accepted={len(accepted_sample_indices)}/{target_valid_count} "
                    "resume committed",
                    flush=True,
                )
            else:
                baseline_stage_names = list(
                    runtime_baselines[float(causal_sigma)][sample_index].stage_names
                )
                baseline_causal_result = runtime_baselines[float(causal_sigma)][sample_index]

            if (
                automatic_interventions
                and baseline_stage_names is not None
                and baseline_causal_result is not None
            ):
                sigma = float(causal_sigma)
                noisy_latents = sigma * fixed_noise + (1.0 - sigma) * latents
                timestep_value = input_builder.timestep_for_sigma(sigma)
                timesteps = timestep_value.expand(*noisy_latents.shape[:3]).clone()
                for intervention in intervention_plan:
                    label = str(intervention["label"])
                    existing_results = runtime_interventions.get(label, {}).get("results", {})
                    if sample_index in existing_results:
                        continue
                    save_stage_names = _resolve_intervention_save_stages(
                        baseline_stage_names,
                        intervention,
                        controller.num_layers,
                    )
                    metadata = dict(eligibility_metadata)
                    metadata.update(
                        {
                            "sigma": sigma,
                            "capture_role": "intervention",
                            "storage_policy": "resume_probe_statistics_only",
                            "save_stage_names": save_stage_names,
                            "intervention_label": label,
                            "intervention_module": intervention["module"],
                            "intervention_layer": int(intervention["layer"]),
                            "intervention_pattern": intervention["pattern"],
                            "intervention_sigma": sigma,
                        }
                    )
                    logical_path = run_root / f"sample_{sample_index:05d}" / "intervention" / label
                    controller.gate_spec = dict(baseline_gate_spec)
                    controller.gate_spec[intervention["pattern"]] = 0.0
                    try:
                        controller.configure_capture(
                            batch,
                            track_data,
                            token_height,
                            token_width,
                            logical_path,
                            metadata=metadata,
                        )
                        with torch.no_grad():
                            intervention_output = entity_reactor_forward(
                                pipeline,
                                noisy_latents.to(pipeline.model_dtype),
                                timesteps,
                                model_conditions,
                            )
                        capture = controller.latest_capture
                        if capture is None:
                            raise RuntimeError(
                                "intervention forward produced no in-memory snapshot"
                            )
                        intervention_result, _ = evaluate_capture(
                            capture,
                            transport_stage_indices=set(),
                        )
                        entry = runtime_interventions.setdefault(
                            label,
                            {"metadata": dict(metadata), "results": {}},
                        )
                        entry["results"][sample_index] = intervention_result
                        _write_intervention_resume(
                            _intervention_resume_path(resume_root, sample_index, label),
                            resume_signature,
                            label,
                            metadata,
                            sample_index,
                            intervention_result,
                        )
                        print(
                            f"[EntityReactor] causal sample={sample_index} {label} -> resume committed",
                            flush=True,
                        )
                        del intervention_output, capture
                        controller.latest_capture = None
                        controller.recorder.stage_features.clear()
                    except RuntimeError as error:
                        print(
                            f"[EntityReactor] causal skip sample={sample_index} {label} because {error}",
                            flush=True,
                        )
                    finally:
                        controller.gate_spec = dict(baseline_gate_spec)
                del noisy_latents, timesteps

            del batch, latents, fixed_noise, model_conditions, track_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(accepted_sample_indices) != target_valid_count:
            raise RuntimeError(
                f"requested {target_valid_count} valid clips but collected "
                f"{len(accepted_sample_indices)}"
            )
        if first_metadata is None:
            raise RuntimeError("no valid runtime metadata was collected")

        atlas = build_runtime_atlas(
            runtime_baselines,
            transport_reservoirs,
            runtime_interventions,
            metadata=first_metadata,
            num_layers=controller.num_layers,
            seed=int(args.analysis_seed),
        )
        plot_cache_path, plot_manifest_path = write_plot_data_cache(atlas, run_root)
        print(
            f"[EntityReactor] final visualization cache -> {plot_cache_path}",
            flush=True,
        )
        print(
            f"[EntityReactor] accepted samples -> {accepted_sample_indices}",
            flush=True,
        )
        _write_resume_progress(
            progress_path,
            resume_signature,
            "complete",
            candidate_indices,
            next_backfill_index,
            rejected_samples,
            accepted_sample_indices,
        )
        if resume_root.is_dir():
            shutil.rmtree(resume_root)
        if not args.no_render:
            causal_path, overview_path = render_model_reactor(run_root)
            print(f"[EntityReactor] causal figure -> {causal_path}", flush=True)
            print(f"[EntityReactor] overview figure -> {overview_path}", flush=True)
        return run_root
    finally:
        controller.gate_spec = dict(baseline_gate_spec)
        controller.detach()
        del controller, input_builder, pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main() -> None:
    args = create_parser().parse_args()
    base_config = json.loads(args.config_path.read_text(encoding="utf-8"))
    checkpoint_paths = resolve_checkpoint_paths(
        base_config,
        args.checkpoint,
        args.all_checkpoints,
    )
    device = setup_distributed(base_config)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != 0:
        raise RuntimeError("Entity Reactor analysis must run with one process")

    args.output_path.mkdir(parents=True, exist_ok=True)
    initialize_global_state(base_config)
    validation_dataset = dwm.common.create_instance_from_config(base_config["validation_dataset"])
    track_resolver = NuPlanTrackResolver(
        validation_dataset,
        max_entities=None,
        allowed_classes=args.entity_classes,
        minimum_track_length=args.minimum_track_length,
    )
    print("[EntityReactor] checkpoints", flush=True)
    for checkpoint_path in checkpoint_paths:
        print(f"  {checkpoint_path}", flush=True)

    completed_roots = []
    for checkpoint_path in checkpoint_paths:
        run_root = run_checkpoint(
            args,
            base_config,
            validation_dataset,
            track_resolver,
            checkpoint_path,
            device,
        )
        completed_roots.append(run_root)

    if not args.no_render:
        all_roots = discover_run_roots(args.output_path)
        comparison_path = render_design_comparison(
            all_roots,
            args.output_path / "entity_relation_design_comparison.png",
        )
        if comparison_path is not None:
            print(f"[EntityReactor] design comparison -> {comparison_path}", flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
