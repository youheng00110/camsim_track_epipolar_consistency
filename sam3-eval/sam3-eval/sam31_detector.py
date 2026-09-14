from __future__ import annotations

import importlib
import os
import sys
from contextlib import nullcontext
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torchvision.ops import nms


class MockDetector:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device
        self.load_info = {"backend": "mock", "parameter_coverage": 1.0}

    @torch.inference_mode()
    def predict_batch(
        self,
        images: list[torch.Tensor],
        metadata: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        outputs: list[list[dict[str, Any]]] = []
        for image, item in zip(images, metadata):
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError("Mock detector expects CHW RGB images")
            detections: list[dict[str, Any]] = []
            for projection in item["gt_projections"]:
                detections.append(
                    {
                        "prompt": "mock vehicle",
                        "score": 0.99,
                        "bbox_xyxy": list(projection["bbox_xyxy"]),
                        "bottom_center_xy": list(projection["bottom_center_xy"]),
                        "mask": None,
                    }
                )
            outputs.append(detections)
        return outputs


class Sam31Detector:
    def __init__(self, config: dict[str, Any], device: torch.device, sam3_repo: str, checkpoint: str) -> None:
        self.config = config
        self.device = device
        self.sam3_repo = Path(sam3_repo).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.sam3_repo.is_dir():
            raise FileNotFoundError(f"SAM 3.1 repository does not exist: {self.sam3_repo}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"SAM checkpoint does not exist: {self.checkpoint}")
        sys.path.insert(0, str(self.sam3_repo))
        sam3_module = importlib.import_module("sam3")
        print(f"Imported sam3 from {Path(sam3_module.__file__).resolve()}")

        self.resolution = int(config["input_resolution"])
        self.mask_resolution = int(config["mask_resolution"])
        self.save_masks = bool(config["save_masks"])
        self.prompts = [str(value) for value in config["prompts"]]
        self.confidence_threshold = float(config["confidence_threshold"])
        self.max_per_prompt = int(config["max_detections_per_prompt"])
        self.max_per_image = int(config["max_detections_per_image"])
        self.nms_iou_threshold = float(config["nms_iou_threshold"])
        self.mask_threshold = float(config["mask_threshold"])
        precision_name = str(config.get("precision", "bfloat16")).lower()
        self.autocast_dtype = torch.bfloat16 if precision_name == "bfloat16" else torch.float16
        self.model, self.find_stage_type, self.load_info = self._build_model()
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            self.text_outputs = self.model.backbone.forward_text(self.prompts, device=self.device)

    def _build_model(self) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
        from sam3.model.data_misc import FindStage
        from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
        from sam3.model.vl_combiner import SAM3VLBackboneTri
        from sam3.model_builder import (
            _create_dot_product_scoring,
            _create_geometry_encoder,
            _create_multiplex_tri_backbone,
            _create_sam3_transformer,
            _create_segmentation_head,
            _create_text_encoder,
        )

        bpe_path = self.sam3_repo / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe_path.is_file():
            installed_bpe = Path(str(files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz")))
            bpe_path = installed_bpe
        if not bpe_path.is_file():
            raise FileNotFoundError(f"SAM tokenizer vocabulary was not found under {self.sam3_repo}")

        tri_neck = _create_multiplex_tri_backbone(
            compile_mode=None,
            use_fa3=False,
            use_rope_real=False,
        )
        text_encoder = _create_text_encoder(str(bpe_path))
        backbone = SAM3VLBackboneTri(scalp=0, visual=tri_neck, text=text_encoder)
        transformer = _create_sam3_transformer(use_fa3=False)
        segmentation_head = None
        if self.save_masks:
            segmentation_head = _create_segmentation_head(compile_mode=None, use_fa3=False)
        model = Sam3MultiplexDetector(
            num_feature_levels=1,
            backbone=backbone,
            transformer=transformer,
            segmentation_head=segmentation_head,
            semantic_segmentation_head=None,
            input_geometry_encoder=_create_geometry_encoder(),
            use_early_fusion=True,
            use_dot_prod_scoring=True,
            dot_prod_scoring=_create_dot_product_scoring(),
            supervise_joint_box_scores=True,
            gather_backbone_out=False,
            is_multiplex=True,
        )
        load_info = load_detector_checkpoint(
            model=model,
            checkpoint_path=self.checkpoint,
            minimum_coverage=float(self.config["checkpoint_minimum_coverage"]),
            use_mmap=bool(self.config["checkpoint_mmap"]),
        )
        model = model.to(self.device)
        model.eval()
        print(
            "Loaded checkpoint "
            f"coverage={load_info['parameter_coverage']:.2%} "
            f"tensors={load_info['loaded_tensors']}/{load_info['model_tensors']}"
        )
        return model, FindStage, load_info

    def preprocess_images(self, images: list[torch.Tensor]) -> torch.Tensor:
        resized_images: list[torch.Tensor] = []
        for image in images:
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(f"Expected CHW RGB image, got {tuple(image.shape)}")
            device_image = image.to(self.device, non_blocking=True).float().div_(255.0)
            device_image = functional.interpolate(
                device_image.unsqueeze(0),
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
            resized_images.append(device_image.sub_(0.5).div_(0.5))
        return torch.stack(resized_images, dim=0)

    def run_prompt(
        self,
        visual_outputs: dict[str, Any],
        metadata: list[dict[str, Any]],
        prompt_index: int,
        per_image_outputs: list[list[dict[str, Any]]],
    ) -> None:
        image_count = len(metadata)
        image_ids = torch.arange(image_count, device=self.device, dtype=torch.long)
        text_ids = torch.full((image_count,), prompt_index, device=self.device, dtype=torch.long)
        find_stage = self.find_stage_type(
            img_ids=image_ids,
            text_ids=text_ids,
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        backbone_outputs = dict(visual_outputs)
        backbone_outputs.update(self.text_outputs)
        geometric_prompt = self.model._get_dummy_prompt(num_prompts=image_count)
        outputs = self.model.forward_grounding(
            backbone_out=backbone_outputs,
            find_input=find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )
        scores = outputs["pred_logits"].sigmoid().squeeze(-1)
        boxes = outputs["pred_boxes"]
        masks = outputs.get("pred_masks")
        for image_index in range(image_count):
            query_scores = scores[image_index]
            kept = torch.nonzero(query_scores >= self.confidence_threshold, as_tuple=False).flatten()
            if kept.numel() == 0:
                continue
            if kept.numel() > self.max_per_prompt:
                local_scores = query_scores[kept]
                kept = kept[torch.topk(local_scores, self.max_per_prompt, sorted=True).indices]
            image_height = int(metadata[image_index]["height"])
            image_width = int(metadata[image_index]["width"])
            selected_boxes = boxes[image_index, kept]
            center_x, center_y, box_width, box_height = selected_boxes.unbind(dim=-1)
            pixel_boxes = torch.stack(
                [
                    (center_x - box_width * 0.5) * image_width,
                    (center_y - box_height * 0.5) * image_height,
                    (center_x + box_width * 0.5) * image_width,
                    (center_y + box_height * 0.5) * image_height,
                ],
                dim=-1,
            )
            selected_masks = None
            if self.save_masks and masks is not None:
                selected_masks = functional.interpolate(
                    masks[image_index, kept].unsqueeze(1),
                    size=(self.mask_resolution, self.mask_resolution),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid().squeeze(1)
            boxes_cpu = pixel_boxes.float().cpu().numpy()
            scores_cpu = query_scores[kept].float().cpu().numpy()
            masks_cpu = None
            if selected_masks is not None:
                masks_cpu = (selected_masks > self.mask_threshold).cpu().numpy()
            for detection_index in range(boxes_cpu.shape[0]):
                box = boxes_cpu[detection_index]
                box[0::2] = np.clip(box[0::2], 0.0, float(image_width - 1))
                box[1::2] = np.clip(box[1::2], 0.0, float(image_height - 1))
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                bottom_center = [float((box[0] + box[2]) * 0.5), float(box[3])]
                binary_mask = None
                if masks_cpu is not None:
                    binary_mask = masks_cpu[detection_index]
                    rows, columns = np.nonzero(binary_mask)
                    if rows.size > 0:
                        bottom_row = int(rows.max())
                        band = rows >= max(0, bottom_row - 1)
                        bottom_column = float(np.median(columns[band]))
                        bottom_center = [
                            (bottom_column + 0.5) * image_width / self.mask_resolution,
                            (bottom_row + 1.0) * image_height / self.mask_resolution,
                        ]
                per_image_outputs[image_index].append(
                    {
                        "prompt": self.prompts[prompt_index],
                        "score": float(scores_cpu[detection_index]),
                        "bbox_xyxy": box.tolist(),
                        "bottom_center_xy": bottom_center,
                        "mask": binary_mask,
                    }
                )
        del outputs

    @torch.inference_mode()
    def predict_batch(
        self,
        images: list[torch.Tensor],
        metadata: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        per_image_outputs: list[list[dict[str, Any]]] = [[] for _ in images]
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            input_tensor = self.preprocess_images(images)
            visual_outputs = self.model.backbone.forward_image(
                input_tensor,
                need_sam3_out=True,
                need_interactive_out=False,
                need_propagation_out=False,
            )
            for prompt_index in range(len(self.prompts)):
                self.run_prompt(visual_outputs, metadata, prompt_index, per_image_outputs)
        filtered_outputs: list[list[dict[str, Any]]] = []
        for detections in per_image_outputs:
            if not detections:
                filtered_outputs.append([])
                continue
            box_tensor = torch.as_tensor(
                [value["bbox_xyxy"] for value in detections],
                dtype=torch.float32,
                device=self.device,
            )
            score_tensor = torch.as_tensor(
                [value["score"] for value in detections],
                dtype=torch.float32,
                device=self.device,
            )
            kept_indices = nms(box_tensor, score_tensor, self.nms_iou_threshold)
            kept_indices = kept_indices[: self.max_per_image].cpu().tolist()
            filtered_outputs.append([detections[index] for index in kept_indices])
        del visual_outputs, input_tensor
        return filtered_outputs


def discover_checkpoint(sam3_repo: str, configured_checkpoint: str) -> str:
    if configured_checkpoint:
        checkpoint = Path(configured_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Configured checkpoint does not exist: {checkpoint}")
        return str(checkpoint)
    environment_checkpoint = os.environ.get("SAM31_CHECKPOINT", "").strip()
    if environment_checkpoint:
        checkpoint = Path(environment_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM31_CHECKPOINT does not exist: {checkpoint}")
        return str(checkpoint)
    repository = Path(sam3_repo).expanduser().resolve()
    preferred_names = (
        "sam3.1_multiplex.pt",
        "sam3.1.pt",
        "sam3_multiplex.pt",
        "sam3.pt",
    )
    search_roots = [repository, repository / "checkpoints", repository / "weights"]
    for name in preferred_names:
        for root in search_roots:
            candidate = root / name
            if candidate.is_file():
                return str(candidate.resolve())
    found: list[Path] = []
    for pattern in ("*sam3*.pt", "*sam3*.pth"):
        found.extend(repository.glob(pattern))
        found.extend((repository / "checkpoints").glob(pattern) if (repository / "checkpoints").is_dir() else [])
        found.extend((repository / "weights").glob(pattern) if (repository / "weights").is_dir() else [])
    unique = sorted({value.resolve() for value in found if value.is_file()})
    if len(unique) == 1:
        return str(unique[0])
    if len(unique) > 1:
        listing = "\n".join(str(value) for value in unique)
        raise RuntimeError(f"Multiple SAM checkpoints were found. Set paths.checkpoint explicitly.\n{listing}")
    raise FileNotFoundError(
        "No SAM checkpoint was found. Set paths.checkpoint in config.yaml or export SAM31_CHECKPOINT."
    )


def load_detector_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    minimum_coverage: float,
    use_mmap: bool,
) -> dict[str, Any]:
    load_kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": True}
    if use_mmap:
        load_kwargs["mmap"] = True
    try:
        payload = torch.load(checkpoint_path, **load_kwargs)
    except (RuntimeError, TypeError, ValueError):
        load_kwargs.pop("mmap", None)
        payload = torch.load(checkpoint_path, **load_kwargs)
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload in {checkpoint_path}")

    raw_state = {str(key): value for key, value in payload.items() if isinstance(value, torch.Tensor)}
    normalized_state: dict[str, torch.Tensor] = {}
    for raw_key, value in raw_state.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if key.startswith("detector."):
            key = key[len("detector.") :]
        elif key.startswith("sam3_model."):
            key = key[len("sam3_model.") :]
        elif key.startswith("tracker."):
            continue
        normalized_state[key] = value

    model_state = model.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    shape_mismatches: list[str] = []
    for key, value in normalized_state.items():
        expected = model_state.get(key)
        if expected is None:
            continue
        if tuple(expected.shape) != tuple(value.shape):
            shape_mismatches.append(key)
            continue
        compatible_state[key] = value
    total_numel = sum(int(value.numel()) for value in model_state.values())
    loaded_numel = sum(int(model_state[key].numel()) for key in compatible_state)
    coverage = loaded_numel / max(total_numel, 1)
    if coverage < minimum_coverage:
        raise RuntimeError(
            f"Checkpoint coverage is only {coverage:.2%}, below {minimum_coverage:.2%}. "
            "The local SAM source and checkpoint are probably incompatible."
        )
    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "parameter_coverage": float(coverage),
        "loaded_tensors": len(compatible_state),
        "model_tensors": len(model_state),
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "shape_mismatches": shape_mismatches,
    }


def predict_with_oom_backoff(
    detector: Any,
    images: list[torch.Tensor],
    metadata: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    pending: list[tuple[list[torch.Tensor], list[dict[str, Any]]]] = [(images, metadata)]
    outputs: list[list[dict[str, Any]]] = []
    while pending:
        image_chunk, metadata_chunk = pending.pop(0)
        try:
            outputs.extend(detector.predict_batch(image_chunk, metadata_chunk))
        except torch.OutOfMemoryError:
            if len(image_chunk) <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            midpoint = len(image_chunk) // 2
            pending.insert(0, (image_chunk[midpoint:], metadata_chunk[midpoint:]))
            pending.insert(0, (image_chunk[:midpoint], metadata_chunk[:midpoint]))
    return outputs
