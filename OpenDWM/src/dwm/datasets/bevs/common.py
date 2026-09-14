import re
from typing import Dict, Iterable, Optional, Set


CLASS_NAMES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)


def require_track_id(track_id, dataset_name: str) -> str:
    if track_id is None:
        raise KeyError(f"{dataset_name} annotation is missing its official track ID.")
    if isinstance(track_id, bytes):
        value = track_id.decode("utf-8").strip()
    else:
        value = str(track_id).strip()
    if not value:
        raise ValueError(f"{dataset_name} annotation has an empty official track ID.")
    return value


def assign_stable_slot(
    track_id: str,
    class_id: int,
    slot_by_track: Dict[str, int],
    class_by_track: Dict[str, int],
    max_boxes: int,
) -> Optional[int]:
    if class_id < 0:
        return None
    if track_id in class_by_track and class_by_track[track_id] != class_id:
        raise ValueError(
            f"Track {track_id!r} changes class from "
            f"{class_by_track[track_id]} to {class_id}."
        )
    class_by_track[track_id] = class_id
    if track_id not in slot_by_track:
        if len(slot_by_track) >= max_boxes:
            return None
        slot_by_track[track_id] = len(slot_by_track)
    return slot_by_track[track_id]


def assert_unique_tracks(track_ids: Iterable[str], dataset_name: str) -> None:
    seen: Set[str] = set()
    for track_id in track_ids:
        if track_id in seen:
            raise ValueError(
                f"{dataset_name} frame contains duplicate track ID {track_id!r}."
            )
        seen.add(track_id)


def nuscenes_class_id(category_name: str) -> int:
    name = str(category_name)
    mapping = (
        ("vehicle.car", 0),
        ("vehicle.truck", 1),
        ("vehicle.construction", 2),
        ("vehicle.bus", 3),
        ("vehicle.trailer", 4),
        ("movable_object.barrier", 5),
        ("vehicle.motorcycle", 6),
        ("vehicle.bicycle", 7),
        ("human.pedestrian", 8),
        ("movable_object.trafficcone", 9),
    )
    for prefix, class_id in mapping:
        if name.startswith(prefix):
            return class_id
    return -1


def waymo_class_id(label_type: int) -> int:
    mapping = {
        1: 0,
        2: 8,
        4: 7,
    }
    return mapping.get(int(label_type), -1)


def argoverse_class_id(category_name: str) -> int:
    name = str(category_name).upper()
    mapping = {
        "REGULAR_VEHICLE": 0,
        "TRUCK": 1,
        "BOX_TRUCK": 1,
        "TRUCK_CAB": 1,
        "LARGE_VEHICLE": 1,
        "BUS": 3,
        "SCHOOL_BUS": 3,
        "ARTICULATED_BUS": 3,
        "VEHICULAR_TRAILER": 4,
        "MESSAGE_BOARD_TRAILER": 4,
        "TRAFFIC_LIGHT_TRAILER": 4,
        "BOLLARD": 5,
        "CONSTRUCTION_BARREL": 5,
        "MOTORCYCLE": 6,
        "MOTORCYCLIST": 6,
        "BICYCLE": 7,
        "BICYCLIST": 7,
        "PEDESTRIAN": 8,
        "OFFICIAL_SIGNALER": 8,
        "STROLLER": 8,
        "WHEELCHAIR": 8,
        "CONSTRUCTION_CONE": 9,
    }
    return mapping.get(name, -1)


def nuplan_class_id(category_name: str) -> int:
    name = re.sub(r"[\s/\-.]+", "_", str(category_name).strip().lower())
    mapping = {
        "car": 0,
        "vehicle": 0,
        "regular_vehicle": 0,
        "truck": 1,
        "box_truck": 1,
        "truck_cab": 1,
        "large_vehicle": 1,
        "construction_vehicle": 2,
        "bus": 3,
        "school_bus": 3,
        "articulated_bus": 3,
        "trailer": 4,
        "vehicular_trailer": 4,
        "barrier": 5,
        "bollard": 5,
        "construction_barrel": 5,
        "motorcycle": 6,
        "motorcyclist": 6,
        "bicycle": 7,
        "bicyclist": 7,
        "cyclist": 7,
        "bike": 7,
        "pedestrian": 8,
        "ped": 8,
        "official_signaler": 8,
        "traffic_cone": 9,
        "construction_cone": 9,
        "cone": 9,
    }
    return mapping.get(name, -1)


def keep_training_keys(result: dict, stub_key_data_dict: Optional[dict]) -> dict:
    keep = {
        "fps",
        "images",
        "camera_intrinsics",
        "image_size",
        "camera_transforms",
        "ego_transforms",
        "reference_ego_transforms",
        "3dbox_images",
        "hdmap_bev_images",
        "bbox_token_corners",
        "bbox_token_classes",
        "bbox_token_masks",
    }
    if stub_key_data_dict is not None:
        keep.update(stub_key_data_dict.keys())
    return {key: value for key, value in result.items() if key in keep}
