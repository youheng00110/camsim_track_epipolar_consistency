#!/usr/bin/env python3
"""
Create a NEW BEV + PV-condition + Pluecker + normal-CA experiment.

No original source file is modified.

Generated files:
  src/dwm/models/lyh/bev_pv_plucker.py
  src/dwm/pipelines/lyh/bev_pv.py
  src/dwm/datasets/lyh/bev_pv.py
  configs/lyh/BEV_PV_plucker_train.json

The BEV base remains authoritative:
  - TemporalBBoxConditionEncoder / EgoTrajectoryConditionEncoder
  - TemporalBEVResidualAdapter
  - ConditionCrossAttention
  - normal temporal attention
  - normal rowwise cross-view attention
  - Pluecker camera geometry

The only model-side addition is:
  PV 3dbox + PV hdmap + PV instance-flow
    -> 9-channel condition_image_tensor
    -> ImageAdapter
    -> residuals injected before the main SD3 blocks

Dataset strategy:
  pair each specialized dwm.datasets.bevs.* sample with the matching
  dwm.datasets.track_pv.* sample at the same index. The BEV sample supplies
  target images, stable bbox slots, BEV map, geometry, and 3dbox_images.
  The PV sample supplies hdmap_images and instance_flow_images.

Usage:
  python patch_bev_pv_plucker.py /path/to/OpenDWM
  python patch_bev_pv_plucker.py /path/to/OpenDWM --force

Optional:
  --bev-checkpoint /path/to/bev_base.pth
  --output-path /path/to/output
"""

import argparse
import ast
import copy
import json
import py_compile
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


EMBEDDED_BEV_CONFIG = r"""{
    "device": "cuda",
    "ddp_backend": "nccl",
    "train_epochs": 20,
    "generator_seed": 0,
    "same_dataset_per_global_batch": true,
    "data_shuffle": true,
    "fix_training_data_order": true,
    "global_state": {
        "nuscenes_fs": {
            "_class_name": "fsspec.implementations.dirfs.DirFileSystem",
            "path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh",
            "fs": {
                "_class_name": "dwm.fs.dirfs.DirFileSystem"
            }
        },
        "device_mesh": {
            "_class_name": "torch.distributed.device_mesh.init_device_mesh",
            "device_type": "cuda",
            "mesh_shape": [
                1,
                8
            ]
        }
    },
    "optimizer": {
        "_class_name": "torch.optim.AdamW",
        "lr": 6e-05
    },
    "pipeline": {
        "_class_name": "dwm.pipelines.bev.BEVPipeline",
        "common_config": {
            "distribution_framework": "fsdp",
            "print_load_state_info": true,
            "memory_efficient_batch": 12,
            "fsdp_settings": {
                "sharding_strategy": {
                    "_class_name": "torch.distributed.fsdp.ShardingStrategy",
                    "value": 4
                },
                "device_mesh": {
                    "_class_name": "dwm.common.get_state",
                    "key": "device_mesh"
                },
                "mixed_precision": {
                    "_class_name": "torch.distributed.fsdp.MixedPrecision",
                    "param_dtype": {
                        "_class_name": "get_class",
                        "class_name": "torch.float16"
                    }
                }
            }
        },
        "training_config": {
            "reference_frame_count": 3,
            "generation_task_ratio": 0.2,
            "image_generation_ratio": 0.5,
            "all_reference_visible_ratio": 0.5,
            "reference_visible_rate": 0.5,
            "condition_dropout_ratio": 0.15,
            "disable_reference_frame_loss": true
        },
        "inference_config": {
            "scheduler": "dwm.schedulers.temporal_independent.FlowMatchEulerDiscreteScheduler",
            "guidance_scale": 2.0,
            "inference_steps": 38,
            "preview_image_size": [
                512,
                288
            ],
            "generate_frames_for_reference": false,
            "sequence_length_per_iteration": 19,
            "reference_frame_count": 3,
            "autoregressive_stride": 16,
            "autoregression_data_exception_for_take_sequence": [
                "crossview_mask"
            ],
            "evaluation_item_count": 480
        },
        "model": {
            "_class_name": "dwm.models.bev_models.dit.BEVConditionedSD3TransformerModel",
            "attention_head_dim": 64,
            "caption_projection_dim": 1536,
            "in_channels": 16,
            "joint_attention_dim": 4096,
            "num_attention_heads": 24,
            "num_layers": 24,
            "out_channels": 16,
            "patch_size": 2,
            "pooled_projection_dim": 2048,
            "pos_embed_max_size": 192,
            "sample_size": 128,
            "block_layers": [
                1,
                5,
                9,
                13,
                17,
                21
            ],
            "merge_factor": 2.0,
            "bev_in_channels": 13,
            "bev_hidden_channels": 256,
            "trajectory_translation_scale": 10.0,
            "bbox_config": {
                "num_classes": 10,
                "points_per_box": 8,
                "num_freqs": 4,
                "hidden_dim": 768,
                "class_dim": 768,
                "temporal_depth": 1,
                "temporal_heads": 8,
                "position_min": [
                    -80.0,
                    -80.0,
                    -5.0
                ],
                "position_range": [
                    160.0,
                    160.0,
                    10.0
                ]
            }
        },
        "pretrained_model_name_or_path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/lyhmigration/stable-diffusion-3-medium",
        "model_checkpoint_path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/train_nuplantokensinglefixed3/checkpoints/50000_map13_from_map3.pth",
        "model_load_state_args": {
            "strict": false
        },
        "metrics": {
            "fid": {
                "_class_name": "torchmetrics.image.fid.FrechetInceptionDistance",
                "normalize": true
            },
            "fvd": {
                "_class_name": "dwm.metrics.fvd.FrechetVideoDistance",
                "inception_3d_checkpoint_path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/ckpt/i3d_pretrained_400.pt",
                "sequence_count": 19
            }
        }
    },
    "training_dataset": {
        "_class_name": "dwm.datasets.common.DatasetAdapter",
        "default_height": 288,
        "default_width": 512,
        "base_dataset": {
            "_class_name": "torch.utils.data.ConcatDataset",
            "datasets": [
                {
                    "_class_name": "dwm.datasets.bevs.waymo.MotionDataset",
                    "fs": {
                        "_class_name": "dwm.fs.dirfs.DirFileSystem",
                        "path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/waymo_link"
                    },
                    "split": "training",
                    "dataset_root": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/waymo_link",
                    "balanced_json_path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/waymo_balanced/balanced_windows_metadata.json",
                    "info_dict_path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/waymo_balanced/training.info.json",
                    "sequence_length": 20,
                    "fps_stride_tuples": [
                        [
                            6,
                            2.5,
                            0.95
                        ],
                        [
                            2,
                            3,
                            0.39
                        ]
                    ],
                    "sensor_channels": [
                        "LIDAR_TOP",
                        "CAM_FRONT_LEFT",
                        "CAM_FRONT_LEFT",
                        "CAM_FRONT_RIGHT",
                        "CAM_FRONT_RIGHT",
                        "CAM_SIDE_LEFT",
                        "CAM_FRONT",
                        "CAM_FRONT",
                        "CAM_SIDE_RIGHT"
                    ],
                    "hdmap_bev_settings": {
                        "bev_size": [
                            256,
                            256
                        ],
                        "bev_from_ego_transform": [
                            [
                                1.6,
                                0.0,
                                0.0,
                                128.0
                            ],
                            [
                                0.0,
                                -1.6,
                                0.0,
                                128.0
                            ],
                            [
                                0.0,
                                0.0,
                                -1.6,
                                0.0
                            ],
                            [
                                0.0,
                                0.0,
                                0.0,
                                1.0
                            ]
                        ]
                    },
                    "stub_key_data_dict": {
                        "crossview_mask": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": {
                                    "_class_name": "json.loads",
                                    "s": "[[1,0,0,0,1,0,0,0],[0,1,0,0,0,0,0,0],[1,0,0,0,1,1,0,0],[0,0,0,1,0,0,0,0],[1,0,1,0,0,1,0,0],[0,0,1,0,0,1,0,1],[0,0,0,0,0,0,1,0],[0,0,1,0,0,0,0,1]]"
                                },
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.bool"
                                }
                            }
                        ],
                        "dataset_tag": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": 2,
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.int64"
                                }
                            }
                        ]
                    },
                    "max_boxes": 64
                },
                {
                    "_class_name": "dwm.datasets.bevs.nuscenes.MotionDataset",
                    "fs": {
                        "_class_name": "dwm.common.get_state",
                        "key": "nuscenes_fs"
                    },
                    "dataset_name": "interp_12Hz_trainval",
                    "split": "train",
                    "sequence_length": 20,
                    "fps_stride_tuples": [
                        [
                            6,
                            2.5,
                            0.95
                        ],
                        [
                            2,
                            3,
                            0.39
                        ]
                    ],
                    "sensor_channels": [
                        "LIDAR_TOP",
                        "CAM_FRONT_LEFT",
                        "CAM_FRONT_LEFT",
                        "CAM_FRONT",
                        "CAM_FRONT",
                        "CAM_FRONT_RIGHT",
                        "CAM_BACK_RIGHT",
                        "CAM_BACK",
                        "CAM_BACK_LEFT"
                    ],
                    "hdmap_bev_settings": {
                        "bev_size": [
                            256,
                            256
                        ],
                        "bev_from_ego_transform": [
                            [
                                1.6,
                                0.0,
                                0.0,
                                128.0
                            ],
                            [
                                0.0,
                                -1.6,
                                0.0,
                                128.0
                            ],
                            [
                                0.0,
                                0.0,
                                -1.6,
                                0.0
                            ],
                            [
                                0.0,
                                0.0,
                                0.0,
                                1.0
                            ]
                        ]
                    },
                    "stub_key_data_dict": {
                        "crossview_mask": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": {
                                    "_class_name": "json.loads",
                                    "s": "[[1,0,1,0,0,0,0,1],[0,1,0,0,0,0,0,0],[1,0,1,0,1,0,0,0],[0,0,0,1,0,0,0,0],[0,0,1,0,1,1,0,0],[0,0,0,0,1,1,1,0],[0,0,0,0,0,1,1,1],[1,0,0,0,0,0,1,1]]"
                                },
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.bool"
                                }
                            }
                        ],
                        "dataset_tag": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": 1,
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.int64"
                                }
                            }
                        ]
                    },
                    "max_boxes": 64
                },
                {
                    "_class_name": "dwm.datasets.bevs.argoverse.MotionDataset",
                    "fs": {
                        "_class_name": "dwm.fs.dirfs.DirFileSystem",
                        "fs": {
                            "_class_name": "dwm.fs.dirfs.DirFileSystem",
                            "path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/avrgo2_link"
                        },
                        "enable_cached_info": true
                    },
                    "split": "train",
                    "dataset_root": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/avrgo2_link",
                    "index_json_path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/avrgo2_json",
                    "balanced_json_path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/avrgo2_balanced/balanced_windows.json",
                    "sequence_length": 20,
                    "fps_stride_tuples": [
                        [
                            6,
                            2.5,
                            0.95
                        ],
                        [
                            2,
                            3,
                            0.39
                        ]
                    ],
                    "sensor_channels": [
                        "lidar",
                        "cameras/ring_front_center",
                        "cameras/ring_front_center",
                        "cameras/ring_front_left",
                        "cameras/ring_front_right",
                        "cameras/ring_side_right",
                        "cameras/ring_rear_right",
                        "cameras/ring_rear_left",
                        "cameras/ring_side_left"
                    ],
                    "hdmap_bev_settings": {
                        "bev_size": [
                            256,
                            256
                        ],
                        "bev_from_ego_transform": [
                            [
                                1.6,
                                0.0,
                                0.0,
                                128.0
                            ],
                            [
                                0.0,
                                -1.6,
                                0.0,
                                128.0
                            ],
                            [
                                0.0,
                                0.0,
                                -1.6,
                                0.0
                            ],
                            [
                                0.0,
                                0.0,
                                0.0,
                                1.0
                            ]
                        ]
                    },
                    "stub_key_data_dict": {
                        "crossview_mask": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": {
                                    "_class_name": "json.loads",
                                    "s": "[[1,0,1,1,0,0,0,0],[0,1,0,0,0,0,0,0],[1,0,1,0,0,0,0,1],[1,0,0,1,1,0,0,0],[0,0,0,1,1,1,0,0],[0,0,0,0,1,1,1,0],[0,0,0,0,0,1,1,1],[0,0,1,0,0,0,1,1]]"
                                },
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.bool"
                                }
                            }
                        ],
                        "dataset_tag": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": 3,
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.int64"
                                }
                            }
                        ]
                    },
                    "max_boxes": 64
                },
                {
                    "_class_name": "dwm.datasets.bevs.argoverse.MotionDataset",
                    "fs": {
                        "_class_name": "dwm.fs.dirfs.DirFileSystem",
                        "fs": {
                            "_class_name": "dwm.fs.dirfs.DirFileSystem",
                            "path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/avrgo2_link"
                        },
                        "enable_cached_info": true
                    },
                    "split": "train",
                    "dataset_root": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/avrgo2_link",
                    "index_json_path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/avrgo2_json",
                    "balanced_json_path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/avrgo2_balanced/balanced_windows.json",
                    "sequence_length": 20,
                    "fps_stride_tuples": [
                        [
                            6,
                            2.5,
                            0.95
                        ],
                        [
                            2,
                            3,
                            0.39
                        ]
                    ],
                    "sensor_channels": [
                        "lidar",
                        "cameras/ring_front_left",
                        "cameras/ring_front_left",
                        "cameras/ring_front_center",
                        "cameras/ring_front_center",
                        "cameras/ring_front_right",
                        "cameras/ring_front_right",
                        "cameras/ring_rear_right",
                        "cameras/ring_rear_left"
                    ],
                    "hdmap_bev_settings": {
                        "bev_size": [
                            256,
                            256
                        ],
                        "bev_from_ego_transform": [
                            [
                                1.6,
                                0.0,
                                0.0,
                                128.0
                            ],
                            [
                                0.0,
                                -1.6,
                                0.0,
                                128.0
                            ],
                            [
                                0.0,
                                0.0,
                                -1.6,
                                0.0
                            ],
                            [
                                0.0,
                                0.0,
                                0.0,
                                1.0
                            ]
                        ]
                    },
                    "stub_key_data_dict": {
                        "crossview_mask": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": {
                                    "_class_name": "json.loads",
                                    "s": "[[1,0,1,0,0,0,0,0],[0,1,0,0,0,0,0,0],[1,0,1,0,1,0,0,0],[0,0,0,1,0,0,0,0],[0,0,1,0,1,0,1,0],[0,0,0,0,0,1,0,0],[0,0,0,0,1,0,1,1],[0,0,0,0,0,0,1,1]]"
                                },
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.bool"
                                }
                            }
                        ],
                        "dataset_tag": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": 3,
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.int64"
                                }
                            }
                        ]
                    },
                    "max_boxes": 64
                },
                {
                    "_class_name": "dwm.datasets.bevs.argoverse.MotionDataset",
                    "fs": {
                        "_class_name": "dwm.fs.dirfs.DirFileSystem",
                        "fs": {
                            "_class_name": "dwm.fs.dirfs.DirFileSystem",
                            "path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/avrgo2_link"
                        },
                        "enable_cached_info": true
                    },
                    "split": "train",
                    "dataset_root": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/avrgo2_link",
                    "index_json_path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/avrgo2_json",
                    "balanced_json_path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/avrgo2_balanced/balanced_windows.json",
                    "sequence_length": 20,
                    "fps_stride_tuples": [
                        [
                            6,
                            2.5,
                            0.95
                        ],
                        [
                            2,
                            3,
                            0.39
                        ]
                    ],
                    "sensor_channels": [
                        "lidar",
                        "cameras/ring_side_left",
                        "cameras/ring_side_left",
                        "cameras/ring_front_center",
                        "cameras/ring_front_center",
                        "cameras/ring_side_right",
                        "cameras/ring_side_right",
                        "cameras/ring_rear_right",
                        "cameras/ring_rear_left"
                    ],
                    "hdmap_bev_settings": {
                        "bev_size": [
                            256,
                            256
                        ],
                        "bev_from_ego_transform": [
                            [
                                1.6,
                                0.0,
                                0.0,
                                128.0
                            ],
                            [
                                0.0,
                                -1.6,
                                0.0,
                                128.0
                            ],
                            [
                                0.0,
                                0.0,
                                -1.6,
                                0.0
                            ],
                            [
                                0.0,
                                0.0,
                                0.0,
                                1.0
                            ]
                        ]
                    },
                    "stub_key_data_dict": {
                        "crossview_mask": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": {
                                    "_class_name": "json.loads",
                                    "s": "[[1,0,1,0,0,0,0,0],[0,1,0,0,0,0,0,0],[1,0,1,0,1,0,0,0],[0,0,0,1,0,0,0,0],[0,0,1,0,1,0,1,0],[0,0,0,0,0,1,0,0],[0,0,0,0,1,0,1,1],[0,0,0,0,0,0,1,1]]"
                                },
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.bool"
                                }
                            }
                        ],
                        "dataset_tag": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": 3,
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.int64"
                                }
                            }
                        ]
                    },
                    "max_boxes": 64
                }
            ]
        },
        "transform_list": [
            {
                "old_key": "images",
                "new_key": "vae_images",
                "transform": {
                    "_class_name": "torchvision.transforms.Compose",
                    "transforms": [
                        {
                            "_class_name": "torchvision.transforms.Resize",
                            "size": [
                                288,
                                512
                            ]
                        },
                        {
                            "_class_name": "torchvision.transforms.ToTensor"
                        }
                    ]
                }
            },
            {
                "old_key": "hdmap_bev_images",
                "new_key": "hdmap_bev_images",
                "transform": {
                    "_class_name": "torchvision.transforms.Compose",
                    "transforms": [
                        {
                            "_class_name": "torchvision.transforms.ToTensor"
                        }
                    ]
                }
            },
            {
                "old_key": "3dbox_images",
                "new_key": "3dbox_images",
                "transform": {
                    "_class_name": "torchvision.transforms.Compose",
                    "transforms": [
                        {
                            "_class_name": "torchvision.transforms.Resize",
                            "size": [
                                288,
                                512
                            ]
                        },
                        {
                            "_class_name": "torchvision.transforms.ToTensor"
                        }
                    ]
                }
            }
        ],
        "pop_list": [
            "images"
        ]
    },
    "validation_dataset": {
        "_class_name": "dwm.datasets.common.DatasetAdapter",
        "default_height": 288,
        "default_width": 512,
        "base_dataset": {
            "_class_name": "torch.utils.data.ConcatDataset",
            "datasets": [
                {
                    "_class_name": "dwm.datasets.bevs.nuplan.MotionDataset",
                    "sensor_root": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/nuplan_prepo/mini_sensors",
                    "pkl_path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/nuplan_prepo/mini_infos_val.pkl",
                    "cache_root": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/nuplan_cache",
                    "dataset_root": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/nuplan_link/plan_data/mini",
                    "map_root": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/nuplan_link/maps",
                    "balanced_json_path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/nuplan_balanced/val_nonidle_windows_metadata.json",
                    "sequence_length": 19,
                    "fps_stride_tuples": [
                        [
                            6,
                            1.2,
                            0.1
                        ]
                    ],
                    "sensor_channels": [
                        "CAM_L2",
                        "CAM_L1",
                        "CAM_L0",
                        "CAM_F0",
                        "CAM_R0",
                        "CAM_R1",
                        "CAM_R2",
                        "CAM_B0"
                    ],
                    "hdmap_bev_settings": {
                        "bev_size": [
                            256,
                            256
                        ],
                        "bev_from_ego_transform": [
                            [
                                1.6,
                                0.0,
                                0.0,
                                128.0
                            ],
                            [
                                0.0,
                                -1.6,
                                0.0,
                                128.0
                            ],
                            [
                                0.0,
                                0.0,
                                -1.6,
                                0.0
                            ],
                            [
                                0.0,
                                0.0,
                                0.0,
                                1.0
                            ]
                        ]
                    },
                    "stub_key_data_dict": {
                        "crossview_mask": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": {
                                    "_class_name": "json.loads",
                                    "s": "[[1,1,0,0,0,0,0,1],[1,1,1,0,0,0,0,0],[0,1,1,1,0,0,0,0],[0,0,1,1,1,0,0,0],[0,0,0,1,1,1,0,0],[0,0,0,0,1,1,1,0],[0,0,0,0,0,1,1,1],[1,0,0,0,0,0,1,1]]"
                                },
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.bool"
                                }
                            }
                        ],
                        "dataset_tag": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": 0,
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.int64"
                                }
                            }
                        ]
                    },
                    "max_boxes": 64
                }
            ]
        },
        "transform_list": [
            {
                "old_key": "images",
                "new_key": "vae_images",
                "transform": {
                    "_class_name": "torchvision.transforms.Compose",
                    "transforms": [
                        {
                            "_class_name": "torchvision.transforms.Resize",
                            "size": [
                                288,
                                512
                            ]
                        },
                        {
                            "_class_name": "torchvision.transforms.ToTensor"
                        }
                    ]
                }
            },
            {
                "old_key": "3dbox_images",
                "new_key": "3dbox_images",
                "transform": {
                    "_class_name": "torchvision.transforms.Compose",
                    "transforms": [
                        {
                            "_class_name": "torchvision.transforms.Resize",
                            "size": [
                                288,
                                512
                            ]
                        },
                        {
                            "_class_name": "torchvision.transforms.ToTensor"
                        }
                    ]
                }
            },
            {
                "old_key": "hdmap_bev_images",
                "new_key": "hdmap_bev_images",
                "transform": {
                    "_class_name": "torchvision.transforms.Compose",
                    "transforms": [
                        {
                            "_class_name": "torchvision.transforms.ToTensor"
                        }
                    ]
                }
            }
        ],
        "pop_list": [
            "images"
        ]
    },
    "training_dataloader": {
        "batch_size": 1,
        "num_workers": 4,
        "prefetch_factor": 1,
        "persistent_workers": true
    },
    "validation_dataloader": {
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 3,
        "persistent_workers": true
    },
    "preview_dataloader": {
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 1,
        "shuffle": false,
        "drop_last": true,
        "persistent_workers": true
    },
    "output_path": "/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/camsim_lyh/output/train_nuplan_bev_condcross"
}
"""
EMBEDDED_PV_CONFIG = r"""{
    "device": "cuda",
    "ddp_backend": "nccl",
    "train_epochs": 4,
    "generator_seed": 0,
    "same_dataset_per_global_batch": true,
    "data_shuffle": true,
    "fix_training_data_order": true,
    "global_state": {
        "nuscenes_fs": {
            "_class_name": "fsspec.implementations.dirfs.DirFileSystem",
            "path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/nus_local",
            "fs": {
                "_class_name": "dwm.fs.dirfs.DirFileSystem"
            }
        },
        "device_mesh": {
            "_class_name": "torch.distributed.device_mesh.init_device_mesh",
            "device_type": "cuda",
            "mesh_shape": [
                1,
                1
            ]
        }
    },
    "optimizer": {
        "_class_name": "torch.optim.AdamW",
        "lr": 6e-05
    },
    "pipeline": {
        "_class_name": "dwm.pipelines.camsim_track.CrossviewTemporalSD",
        "common_config": {
            "frame_prediction_style": "ctsd",
            "explicit_view_modeling": true,
            "cat_condition": true,
            "cond_with_action": false,
            "print_load_state_info": true,
            "condition_on_all_frames": true,
            "uncondition_image_color": 0.1255,
            "added_time_ids": "fps_camera_transforms",
            "explicit_geometry_clip": {
                "enabled": true,
                "translation_clip": 100.0,
                "hard_clip": 4096.0,
                "intrinsics_clip": 8.0,
                "log": true
            },
            "camera_intrinsic_embedding_indices": [
                0,
                4,
                2,
                5
            ],
            "camera_intrinsic_denom_embedding_indices": [
                1,
                1,
                0,
                1
            ],
            "camera_transform_embedding_indices": [
                2,
                6,
                10,
                3,
                7,
                11
            ],
            "camera_ego_sensor_indices": [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7
            ],
            "distribution_framework": "fsdp",
            "ddp_wrapper_settings": {
                "sharding_strategy": {
                    "_class_name": "torch.distributed.fsdp.ShardingStrategy",
                    "value": 4
                },
                "device_mesh": {
                    "_class_name": "dwm.common.get_state",
                    "key": "device_mesh"
                },
                "auto_wrap_policy": {
                    "_class_name": "torch.distributed.fsdp.wrap.ModuleWrapPolicy",
                    "module_classes": [
                        {
                            "_class_name": "get_class",
                            "class_name": "diffusers.models.attention.JointTransformerBlock"
                        },
                        {
                            "_class_name": "get_class",
                            "class_name": "dwm.models.crossview_temporal.VTSelfAttentionBlock"
                        }
                    ]
                },
                "mixed_precision": {
                    "_class_name": "torch.distributed.fsdp.MixedPrecision",
                    "param_dtype": {
                        "_class_name": "get_class",
                        "class_name": "torch.float16"
                    }
                }
            },
            "t5_fsdp_wrapper_settings": {
                "sharding_strategy": {
                    "_class_name": "torch.distributed.fsdp.ShardingStrategy",
                    "value": 4
                },
                "device_mesh": {
                    "_class_name": "dwm.common.get_state",
                    "key": "device_mesh"
                },
                "auto_wrap_policy": {
                    "_class_name": "torch.distributed.fsdp.wrap.ModuleWrapPolicy",
                    "module_classes": [
                        {
                            "_class_name": "get_class",
                            "class_name": "transformers.models.t5.modeling_t5.T5Block"
                        }
                    ]
                }
            },
            "text_encoder_load_args": {
                "variant": "fp16",
                "torch_dtype": {
                    "_class_name": "get_class",
                    "class_name": "torch.float16"
                }
            },
            "memory_efficient_batch": 12
        },
        "training_config": {
            "text_prompt_condition_ratio": 0.8,
            "3dbox_condition_ratio": 0.8,
            "explicit_view_modeling_ratio": 0.8,
            "reference_frame_count": 3,
            "generation_task_ratio": 0.2,
            "image_generation_ratio": 0.5,
            "all_reference_visible_ratio": 0.5,
            "reference_visible_rate": 0.5,
            "disable_reference_frame_loss": true,
            "enable_grad_scaler": true,
            "hdmap_condition_ratio": 0.8,
            "action_condition_ratio": 0.0
        },
        "inference_config": {
            "scheduler": "dwm.schedulers.temporal_independent.FlowMatchEulerDiscreteScheduler",
            "guidance_scale": 2,
            "inference_steps": 38,
            "preview_image_size": [
                512,
                288
            ],
            "force_preview_resize": false,
            "generate_frames_for_reference": false,
            "sequence_length_per_iteration": 19,
            "clear_reference_frame_count": 0,
            "reference_frame_count": 3,
            "autoregressive_stride": 16,
            "autoregression_data_exception_for_take_sequence": [
                "crossview_mask",
                "camera_names",
                "distortion",
                "angle",
                "dist"
            ],
            "evaluation_item_count": 480
        },
        "model": {
            "_class_name": "dwm.models.crossview_temporal_dit_PLUCKER_track.DiTCrossviewTemporalConditionModel",
            "attention_head_dim": 64,
            "caption_projection_dim": 1536,
            "in_channels": 16,
            "joint_attention_dim": 4096,
            "num_attention_heads": 24,
            "num_layers": 24,
            "out_channels": 16,
            "patch_size": 2,
            "pooled_projection_dim": 2048,
            "pos_embed_max_size": 192,
            "sample_size": 128,
            "perspective_modeling_type": "explicit",
            "enable_crossview": true,
            "crossview_attention_type": "rowwise",
            "crossview_block_layers": [
                1,
                5,
                9,
                13,
                17,
                21
            ],
            "crossview_gradient_checkpointing": true,
            "enable_temporal": true,
            "temporal_attention_type": "rowwise",
            "temporal_block_layers": [
                1,
                5,
                9,
                13,
                17,
                21
            ],
            "temporal_gradient_checkpointing": true,
            "mixer_type": "AlphaBlender",
            "merge_factor": 2,
            "condition_image_adapter_config": {
                "in_channels": 9,
                "channels": [
                    1536,
                    1536,
                    1536,
                    1536,
                    1536,
                    1536
                ],
                "is_downblocks": [
                    true,
                    false,
                    false,
                    false,
                    false,
                    false
                ],
                "num_res_blocks": 2,
                "downscale_factor": 8,
                "use_zero_convs": true
            }
        },
        "pretrained_model_name_or_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/lyhmigration/stable-diffusion-3-medium",
        "model_checkpoint_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/ckpt/28000.pth",
        "model_load_state_args": {
            "strict": false
        },
        "metrics": {
            "fid": {
                "_class_name": "torchmetrics.image.fid.FrechetInceptionDistance",
                "normalize": true
            },
            "fvd": {
                "_class_name": "dwm.metrics.fvd.FrechetVideoDistance",
                "inception_3d_checkpoint_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/pretrain/lyhmigration/i3d_pretrained_400.pt",
                "sequence_count": 16
            }
        }
    },
    "training_dataset": {
        "_class_name": "dwm.datasets.common.DatasetAdapter",
        "default_height": 288,
        "default_width": 512,
        "base_dataset": {
            "_class_name": "torch.utils.data.ConcatDataset",
            "datasets": [
                {
                    "_class_name": "dwm.datasets.track_pv.waymo.MotionDataset",
                    "fs": {
                        "_class_name": "dwm.fs.dirfs.DirFileSystem",
                        "path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/waymo_open_dataset_v_1_4_3"
                    },
                    "split": "training",
                    "dataset_root": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/waymo_open_dataset_v_1_4_3",
                    "balanced_json_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/waymo_balanced/balanced_windows_metadata.json",
                    "info_dict_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/waymo_balanced/training.info.json",
                    "sequence_length": 20,
                    "fps_stride_tuples": [
                        [
                            6,
                            2.5,
                            0.95
                        ],
                        [
                            2,
                            3,
                            0.39
                        ]
                    ],
                    "sensor_channels": [
                        "LIDAR_TOP",
                        "CAM_FRONT_LEFT",
                        "CAM_FRONT_LEFT",
                        "CAM_FRONT_RIGHT",
                        "CAM_FRONT_RIGHT",
                        "CAM_SIDE_LEFT",
                        "CAM_FRONT",
                        "CAM_FRONT",
                        "CAM_SIDE_RIGHT"
                    ],
                    "enable_camera_transforms": true,
                    "enable_ego_transforms": true,
                    "_3dbox_image_settings": {},
                    "hdmap_image_settings": {},
                    "image_description_settings": {
                        "path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/waymo_caption_v2/waymo_caption_v2_train.json",
                        "time_list_dict_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/waymo_caption_v2/waymo_caption_v2_times_train.json",
                        "align_keys": [
                            "time",
                            "weather"
                        ],
                        "reorder_keys": true,
                        "drop_rates": {
                            "environment": 0.04,
                            "objects": 0.08,
                            "image_description": 0.16
                        }
                    },
                    "stub_key_data_dict": {
                        "crossview_mask": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": {
                                    "_class_name": "json.loads",
                                    "s": "[[1,0,0,0,1,0,0,0],[0,1,0,0,0,0,0,0],[1,0,0,0,1,1,0,0],[0,0,0,1,0,0,0,0],[1,0,1,0,0,1,0,0],[0,0,1,0,0,1,0,1],[0,0,0,0,0,0,1,0],[0,0,1,0,0,0,0,1]]"
                                },
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.bool"
                                }
                            }
                        ],
                        "dataset_tag": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": 2,
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.int64"
                                }
                            }
                        ]
                    },
                    "instance_flow_image_settings": {
                        "offset_scale": [
                            5.0,
                            5.0,
                            2.0
                        ],
                        "encoding": "tanh",
                        "near_plane": 0.1,
                        "render_size": [
                            288,
                            512
                        ]
                    }
                },
                {
                    "_class_name": "dwm.datasets.track_pv.nuscenes.MotionDataset",
                    "fs": {
                        "_class_name": "dwm.common.get_state",
                        "key": "nuscenes_fs"
                    },
                    "dataset_name": "interp_12Hz_trainval",
                    "split": "train",
                    "sequence_length": 20,
                    "fps_stride_tuples": [
                        [
                            6,
                            2.5,
                            0.95
                        ],
                        [
                            2,
                            3,
                            0.39
                        ]
                    ],
                    "sensor_channels": [
                        "LIDAR_TOP",
                        "CAM_FRONT_LEFT",
                        "CAM_FRONT_LEFT",
                        "CAM_FRONT",
                        "CAM_FRONT",
                        "CAM_FRONT_RIGHT",
                        "CAM_BACK_RIGHT",
                        "CAM_BACK",
                        "CAM_BACK_LEFT"
                    ],
                    "keyframe_only": true,
                    "enable_synchronization_check": false,
                    "enable_camera_transforms": true,
                    "enable_ego_transforms": true,
                    "_3dbox_image_settings": {},
                    "hdmap_image_settings": {},
                    "image_description_settings": {
                        "path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/nuscenes_caption/nuscenes_v1.0-trainval_caption_v2_train.json",
                        "time_list_dict_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/nuscenes_caption/nuscenes_v1.0-trainval_caption_v2_times_train.json",
                        "align_keys": [
                            "time",
                            "weather"
                        ]
                    },
                    "stub_key_data_dict": {
                        "crossview_mask": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": {
                                    "_class_name": "json.loads",
                                    "s": "[[1,0,1,0,0,0,0,1],[0,1,0,0,0,0,0,0],[1,0,1,0,1,0,0,0],[0,0,0,1,0,0,0,0],[0,0,1,0,1,1,0,0],[0,0,0,0,1,1,1,0],[0,0,0,0,0,1,1,1],[1,0,0,0,0,0,1,1]]"
                                },
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.bool"
                                }
                            }
                        ],
                        "dataset_tag": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": 1,
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.int64"
                                }
                            }
                        ]
                    },
                    "instance_flow_image_settings": {
                        "offset_scale": [
                            5.0,
                            5.0,
                            2.0
                        ],
                        "encoding": "tanh",
                        "near_plane": 0.1,
                        "render_size": [
                            288,
                            512
                        ]
                    }
                },
                {
                    "_class_name": "dwm.datasets.track_pv.argoverse.MotionDataset",
                    "fs": {
                        "_class_name": "dwm.fs.dirfs.DirFileSystem",
                        "fs": {
                            "_class_name": "dwm.fs.dirfs.DirFileSystem",
                            "path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/argoverse2"
                        },
                        "enable_cached_info": true
                    },
                    "split": "train",
                    "dataset_root": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/argoverse2",
                    "index_json_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/avrgo2_json",
                    "balanced_json_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/avrgo2_balanced/balanced_windows.json",
                    "sequence_length": 20,
                    "fps_stride_tuples": [
                        [
                            6,
                            2.5,
                            0.95
                        ],
                        [
                            2,
                            3,
                            0.39
                        ]
                    ],
                    "sensor_channels": [
                        "lidar",
                        "cameras/ring_front_center",
                        "cameras/ring_front_center",
                        "cameras/ring_front_left",
                        "cameras/ring_front_right",
                        "cameras/ring_side_right",
                        "cameras/ring_rear_right",
                        "cameras/ring_rear_left",
                        "cameras/ring_side_left"
                    ],
                    "enable_camera_transforms": true,
                    "enable_ego_transforms": true,
                    "_3dbox_image_settings": {},
                    "hdmap_image_settings": {},
                    "image_description_settings": {
                        "path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/av2_sensor_caption_v2/av2_sensor_caption_v2_train.json",
                        "time_list_dict_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/av2_sensor_caption_v2/av2_sensor_caption_v2_times_train.json",
                        "align_keys": [
                            "time",
                            "weather"
                        ],
                        "reorder_keys": true,
                        "drop_rates": {
                            "environment": 0.04,
                            "objects": 0.08,
                            "image_description": 0.16
                        }
                    },
                    "stub_key_data_dict": {
                        "crossview_mask": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": {
                                    "_class_name": "json.loads",
                                    "s": "[[1,0,1,1,0,0,0,0],[0,1,0,0,0,0,0,0],[1,0,1,0,0,0,0,1],[1,0,0,1,1,0,0,0],[0,0,0,1,1,1,0,0],[0,0,0,0,1,1,1,0],[0,0,0,0,0,1,1,1],[0,0,1,0,0,0,1,1]]"
                                },
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.bool"
                                }
                            }
                        ],
                        "dataset_tag": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": 3,
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.int64"
                                }
                            }
                        ]
                    },
                    "instance_flow_image_settings": {
                        "offset_scale": [
                            5.0,
                            5.0,
                            2.0
                        ],
                        "encoding": "tanh",
                        "near_plane": 0.1,
                        "render_size": [
                            288,
                            512
                        ]
                    },
                    "hide_lidar": true
                },
                {
                    "_class_name": "dwm.datasets.track_pv.argoverse.MotionDataset",
                    "fs": {
                        "_class_name": "dwm.fs.dirfs.DirFileSystem",
                        "fs": {
                            "_class_name": "dwm.fs.dirfs.DirFileSystem",
                            "path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/argoverse2"
                        },
                        "enable_cached_info": true
                    },
                    "split": "train",
                    "dataset_root": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/argoverse2",
                    "index_json_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/avrgo2_json",
                    "balanced_json_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/avrgo2_balanced/balanced_windows.json",
                    "sequence_length": 20,
                    "fps_stride_tuples": [
                        [
                            6,
                            2.5,
                            0.95
                        ],
                        [
                            2,
                            3,
                            0.39
                        ]
                    ],
                    "sensor_channels": [
                        "lidar",
                        "cameras/ring_front_left",
                        "cameras/ring_front_left",
                        "cameras/ring_front_center",
                        "cameras/ring_front_center",
                        "cameras/ring_front_right",
                        "cameras/ring_front_right",
                        "cameras/ring_rear_right",
                        "cameras/ring_rear_left"
                    ],
                    "enable_camera_transforms": true,
                    "enable_ego_transforms": true,
                    "_3dbox_image_settings": {},
                    "hdmap_image_settings": {},
                    "image_description_settings": {
                        "path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/av2_sensor_caption_v2/av2_sensor_caption_v2_train.json",
                        "time_list_dict_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/av2_sensor_caption_v2/av2_sensor_caption_v2_times_train.json",
                        "align_keys": [
                            "time",
                            "weather"
                        ],
                        "reorder_keys": true,
                        "drop_rates": {
                            "environment": 0.04,
                            "objects": 0.08,
                            "image_description": 0.16
                        }
                    },
                    "stub_key_data_dict": {
                        "crossview_mask": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": {
                                    "_class_name": "json.loads",
                                    "s": "[[1,0,1,0,0,0,0,0],[0,1,0,0,0,0,0,0],[1,0,1,0,1,0,0,0],[0,0,0,1,0,0,0,0],[0,0,1,0,1,0,1,0],[0,0,0,0,0,1,0,0],[0,0,0,0,1,0,1,1],[0,0,0,0,0,0,1,1]]"
                                },
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.bool"
                                }
                            }
                        ],
                        "dataset_tag": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": 3,
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.int64"
                                }
                            }
                        ]
                    },
                    "instance_flow_image_settings": {
                        "offset_scale": [
                            5.0,
                            5.0,
                            2.0
                        ],
                        "encoding": "tanh",
                        "near_plane": 0.1,
                        "render_size": [
                            288,
                            512
                        ]
                    },
                    "hide_lidar": true
                },
                {
                    "_class_name": "dwm.datasets.track_pv.argoverse.MotionDataset",
                    "fs": {
                        "_class_name": "dwm.fs.dirfs.DirFileSystem",
                        "fs": {
                            "_class_name": "dwm.fs.dirfs.DirFileSystem",
                            "path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/argoverse2"
                        },
                        "enable_cached_info": true
                    },
                    "split": "train",
                    "dataset_root": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/argoverse2",
                    "index_json_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/avrgo2_json",
                    "balanced_json_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/avrgo2_balanced/balanced_windows.json",
                    "sequence_length": 20,
                    "fps_stride_tuples": [
                        [
                            6,
                            2.5,
                            0.95
                        ],
                        [
                            2,
                            3,
                            0.39
                        ]
                    ],
                    "sensor_channels": [
                        "lidar",
                        "cameras/ring_side_left",
                        "cameras/ring_side_left",
                        "cameras/ring_front_center",
                        "cameras/ring_front_center",
                        "cameras/ring_side_right",
                        "cameras/ring_side_right",
                        "cameras/ring_rear_right",
                        "cameras/ring_rear_left"
                    ],
                    "enable_camera_transforms": true,
                    "enable_ego_transforms": true,
                    "_3dbox_image_settings": {},
                    "hdmap_image_settings": {},
                    "image_description_settings": {
                        "path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/av2_sensor_caption_v2/av2_sensor_caption_v2_train.json",
                        "time_list_dict_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/av2_sensor_caption_v2/av2_sensor_caption_v2_times_train.json",
                        "align_keys": [
                            "time",
                            "weather"
                        ],
                        "reorder_keys": true,
                        "drop_rates": {
                            "environment": 0.04,
                            "objects": 0.08,
                            "image_description": 0.16
                        }
                    },
                    "stub_key_data_dict": {
                        "crossview_mask": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": {
                                    "_class_name": "json.loads",
                                    "s": "[[1,0,1,0,0,0,0,0],[0,1,0,0,0,0,0,0],[1,0,1,0,1,0,0,0],[0,0,0,1,0,0,0,0],[0,0,1,0,1,0,1,0],[0,0,0,0,0,1,0,0],[0,0,0,0,1,0,1,1],[0,0,0,0,0,0,1,1]]"
                                },
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.bool"
                                }
                            }
                        ],
                        "dataset_tag": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": 3,
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.int64"
                                }
                            }
                        ]
                    },
                    "instance_flow_image_settings": {
                        "offset_scale": [
                            5.0,
                            5.0,
                            2.0
                        ],
                        "encoding": "tanh",
                        "near_plane": 0.1,
                        "render_size": [
                            288,
                            512
                        ]
                    },
                    "hide_lidar": true
                }
            ]
        },
        "transform_list": [
            {
                "old_key": "images",
                "new_key": "vae_images",
                "transform": {
                    "_class_name": "torchvision.transforms.Compose",
                    "transforms": [
                        {
                            "_class_name": "torchvision.transforms.Resize",
                            "size": [
                                288,
                                512
                            ]
                        },
                        {
                            "_class_name": "torchvision.transforms.ToTensor"
                        }
                    ]
                }
            },
            {
                "old_key": "3dbox_images",
                "new_key": "3dbox_images",
                "transform": {
                    "_class_name": "torchvision.transforms.Compose",
                    "transforms": [
                        {
                            "_class_name": "torchvision.transforms.Resize",
                            "size": [
                                288,
                                512
                            ]
                        },
                        {
                            "_class_name": "torchvision.transforms.ToTensor"
                        }
                    ]
                }
            },
            {
                "old_key": "hdmap_images",
                "new_key": "hdmap_images",
                "transform": {
                    "_class_name": "torchvision.transforms.Compose",
                    "transforms": [
                        {
                            "_class_name": "torchvision.transforms.Resize",
                            "size": [
                                288,
                                512
                            ]
                        },
                        {
                            "_class_name": "torchvision.transforms.ToTensor"
                        }
                    ]
                }
            },
            {
                "old_key": "instance_flow_images",
                "new_key": "instance_flow_images",
                "transform": {
                    "_class_name": "dwm.datasets.track_pv.pv_flow.ResizeInstanceFlow",
                    "size": [
                        288,
                        512
                    ]
                }
            },
            {
                "old_key": "image_description",
                "new_key": "clip_text",
                "transform": {
                    "_class_name": "dwm.datasets.common.Copy"
                },
                "stack": false
            }
        ],
        "pop_list": [
            "images",
            "image_description"
        ]
    },
    "validation_dataset": {
        "_class_name": "dwm.datasets.common.DatasetAdapter",
        "default_height": 288,
        "default_width": 512,
        "base_dataset": {
            "_class_name": "torch.utils.data.ConcatDataset",
            "datasets": [
                {
                    "_class_name": "dwm.datasets.track_pv.nuplan.MotionDataset",
                    "sensor_root": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/nuplan_prepo/mini_sensors",
                    "pkl_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/nuplan_prepo/mini_infos_val.pkl",
                    "balanced_json_path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/nuplan_balanced/val_nonidle_windows_metadata.json",
                    "cache_root": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/cache",
                    "dataset_root": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/nuPlan/plan_data/mini",
                    "map_root": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/nuPlan/maps",
                    "sequence_length": 19,
                    "fps_stride_tuples": [
                        [
                            6,
                            1.2,
                            0.1
                        ]
                    ],
                    "sensor_channels": [
                        "CAM_L2",
                        "CAM_L1",
                        "CAM_L0",
                        "CAM_F0",
                        "CAM_R0",
                        "CAM_R1",
                        "CAM_R2",
                        "CAM_B0"
                    ],
                    "enable_camera_transforms": true,
                    "enable_ego_transforms": true,
                    "enable_synchronization_check": true,
                    "_3dbox_image_settings": {},
                    "hdmap_image_settings": {},
                    "image_description_settings": {
                        "path": "/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/dataset/nuplan_prepo/nuplan_scene.json",
                        "align_keys": [
                            "time",
                            "weather"
                        ],
                        "reorder_keys": true,
                        "drop_rates": {
                            "environment": 0.04,
                            "objects": 0.08,
                            "image_description": 0.16
                        }
                    },
                    "stub_key_data_dict": {
                        "crossview_mask": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": {
                                    "_class_name": "json.loads",
                                    "s": "[[1,1,0,0,0,0,0,1],[1,1,1,0,0,0,0,0],[0,1,1,1,0,0,0,0],[0,0,1,1,1,0,0,0],[0,0,0,1,1,1,0,0],[0,0,0,0,1,1,1,0],[0,0,0,0,0,1,1,1],[1,0,0,0,0,0,1,1]]"
                                },
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.bool"
                                }
                            }
                        ],
                        "dataset_tag": [
                            "content",
                            {
                                "_class_name": "torch.tensor",
                                "data": 0,
                                "dtype": {
                                    "_class_name": "get_class",
                                    "class_name": "torch.int64"
                                }
                            }
                        ]
                    },
                    "instance_flow_image_settings": {
                        "offset_scale": [
                            5.0,
                            5.0,
                            2.0
                        ],
                        "encoding": "tanh",
                        "near_plane": 0.1,
                        "render_size": [
                            288,
                            512
                        ]
                    }
                }
            ]
        },
        "transform_list": [
            {
                "old_key": "images",
                "new_key": "vae_images",
                "transform": {
                    "_class_name": "torchvision.transforms.Compose",
                    "transforms": [
                        {
                            "_class_name": "torchvision.transforms.Resize",
                            "size": [
                                288,
                                512
                            ]
                        },
                        {
                            "_class_name": "torchvision.transforms.ToTensor"
                        }
                    ]
                }
            },
            {
                "old_key": "3dbox_images",
                "new_key": "3dbox_images",
                "transform": {
                    "_class_name": "torchvision.transforms.Compose",
                    "transforms": [
                        {
                            "_class_name": "torchvision.transforms.Resize",
                            "size": [
                                288,
                                512
                            ]
                        },
                        {
                            "_class_name": "torchvision.transforms.ToTensor"
                        }
                    ]
                }
            },
            {
                "old_key": "hdmap_images",
                "new_key": "hdmap_images",
                "transform": {
                    "_class_name": "torchvision.transforms.Compose",
                    "transforms": [
                        {
                            "_class_name": "torchvision.transforms.Resize",
                            "size": [
                                288,
                                512
                            ]
                        },
                        {
                            "_class_name": "torchvision.transforms.ToTensor"
                        }
                    ]
                }
            },
            {
                "old_key": "instance_flow_images",
                "new_key": "instance_flow_images",
                "transform": {
                    "_class_name": "dwm.datasets.track_pv.pv_flow.ResizeInstanceFlow",
                    "size": [
                        288,
                        512
                    ]
                }
            },
            {
                "old_key": "image_description",
                "new_key": "clip_text",
                "transform": {
                    "_class_name": "dwm.datasets.common.Copy"
                },
                "stack": false
            }
        ],
        "pop_list": [
            "images",
            "image_description"
        ]
    },
    "training_dataloader": {
        "batch_size": 1,
        "num_workers": 6,
        "prefetch_factor": 1,
        "collate_fn": {
            "_class_name": "dwm.datasets.common.CollateFnIgnoring",
            "keys": [
                "clip_text"
            ]
        },
        "persistent_workers": true
    },
    "validation_dataloader": {
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 3,
        "collate_fn": {
            "_class_name": "dwm.datasets.common.CollateFnIgnoring",
            "keys": [
                "clip_text"
            ]
        },
        "persistent_workers": true
    },
    "preview_dataloader": {
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 1,
        "shuffle": true,
        "drop_last": true,
        "collate_fn": {
            "_class_name": "dwm.datasets.common.CollateFnIgnoring",
            "keys": [
                "clip_text"
            ]
        },
        "persistent_workers": true
    },
    "informations": {
        "fid": 13.13,
        "fvd": 98.39,
        "fvd_on_nusc_by_1_ref_frames": 42.32,
        "fvd_on_nusc_without_ref_frame": 89.37,
        "average_total_batch_sizes": 48,
        "steps": 40000
    }
}"""


DATASET_WRAPPER = '"""Aligned BEV + PV dataset composition for the joint conditioning experiment."""\n\nfrom typing import Any\n\nimport torch\n\n\nclass AlignedBEVPVDataset(torch.utils.data.Dataset):\n    """\n    Keep the BEV sample as the authoritative target/geometry sample and attach\n    the missing PV image-condition fields from the matching track_pv sample.\n\n    This preserves the BEV stable-slot annotations and BEV map implementation,\n    while reusing track_pv for PV HD-map rendering and instance flow.\n    """\n\n    PV_REQUIRED_KEYS = (\n        "3dbox_images",\n        "hdmap_images",\n        "instance_flow_images",\n    )\n    PV_COPY_KEYS = (\n        "hdmap_images",\n        "instance_flow_images",\n    )\n\n    def __init__(\n        self,\n        bev_dataset,\n        pv_dataset,\n        verify_alignment: bool = True,\n    ):\n        self.bev_dataset = bev_dataset\n        self.pv_dataset = pv_dataset\n        self.verify_alignment = bool(verify_alignment)\n\n        bev_length = len(self.bev_dataset)\n        pv_length = len(self.pv_dataset)\n        if bev_length != pv_length:\n            raise ValueError(\n                "BEV/PV dataset lengths differ: "\n                f"bev={bev_length}, pv={pv_length}. "\n                "They must use the same split and balanced windows."\n            )\n\n    def __len__(self):\n        return len(self.bev_dataset)\n\n    @staticmethod\n    def _shape(value: Any):\n        return tuple(value.shape) if torch.is_tensor(value) else None\n\n    def _verify(self, index: int, bev: dict, pv: dict):\n        for key in ("camera_intrinsics", "camera_transforms"):\n            if key not in bev or key not in pv:\n                raise KeyError(\n                    f"Aligned BEV/PV dataset requires {key!r} on both sides."\n                )\n            if self._shape(bev[key]) != self._shape(pv[key]):\n                raise ValueError(\n                    f"BEV/PV {key} shape mismatch at index {index}: "\n                    f"bev={self._shape(bev[key])}, pv={self._shape(pv[key])}."\n                )\n\n        if "crossview_mask" in bev and "crossview_mask" in pv:\n            bev_mask = bev["crossview_mask"]\n            pv_mask = pv["crossview_mask"]\n            if torch.is_tensor(bev_mask) and torch.is_tensor(pv_mask):\n                if bev_mask.shape != pv_mask.shape or not torch.equal(\n                    bev_mask.bool(),\n                    pv_mask.bool(),\n                ):\n                    raise ValueError(\n                        f"BEV/PV crossview_mask mismatch at index {index}."\n                    )\n\n        if "3dbox_images" not in bev:\n            raise KeyError(\n                f"BEV dataset did not produce \'3dbox_images\' at index {index}."\n            )\n\n        for key in self.PV_REQUIRED_KEYS:\n            if key not in pv:\n                raise KeyError(\n                    f"PV dataset did not produce {key!r} at index {index}."\n                )\n\n    def __getitem__(self, index):\n        bev = self.bev_dataset[index]\n        pv = self.pv_dataset[index]\n\n        if self.verify_alignment:\n            self._verify(index, bev, pv)\n\n        result = dict(bev)\n\n        # Keep BEV\'s own 3dbox_images so the original BEV box-weighted loss\n        # remains exactly on its original data path. Only attach PV fields\n        # that BEV base does not produce.\n        for key in self.PV_COPY_KEYS:\n            result[key] = pv[key]\n\n        return result\n'


def git_status(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except OSError:
        return ""


def backup(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak.{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def ensure_target(path: Path, force: bool) -> None:
    if not path.exists():
        return
    if not force:
        raise SystemExit(
            "Generated target already exists; originals were not touched:\n"
            f"  {path}\n"
            "Use --force to replace only generated targets."
        )
    print("Backup:", backup(path))


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"Expected exactly one {label} anchor, found {count}. "
            "The local source may differ from the reviewed version."
        )
    return text.replace(old, new, 1)


def build_model(source_text: str) -> str:
    text = source_text

    text = replace_once(
        text,
        "import torch\n\nfrom dwm.models.crossview_temporal",
        "import torch\n\nimport dwm.models.adapters\n"
        "from dwm.models.crossview_temporal",
        "model import",
    )
    text = replace_once(
        text,
        "from .condition import (",
        "from dwm.models.bev_models.condition import (",
        "condition import",
    )
    text = replace_once(
        text,
        """        bev_hidden_channels: int = 256,
        trajectory_translation_scale: float = 10.0,
        **kwargs,
""",
        """        bev_hidden_channels: int = 256,
        trajectory_translation_scale: float = 10.0,
        condition_image_adapter_config: Optional[dict] = None,
        **kwargs,
""",
        "model constructor signature",
    )
    text = replace_once(
        text,
        """        inner_dim = attention_head_dim * num_attention_heads
        self.index_proj = diffusers.models.embeddings.Timesteps(
""",
        """        inner_dim = attention_head_dim * num_attention_heads

        # PV image-condition branch. This is intentionally independent from
        # BEV token/residual conditioning and from Pluecker/normal CA.
        if condition_image_adapter_config is not None:
            self.condition_image_in_channels = int(
                condition_image_adapter_config["in_channels"]
            )
            self.condition_image_adapter = dwm.models.adapters.ImageAdapter(
                **condition_image_adapter_config
            )
        else:
            self.condition_image_in_channels = None
            self.condition_image_adapter = None

        self.index_proj = diffusers.models.embeddings.Timesteps(
""",
        "ImageAdapter construction",
    )
    text = replace_once(
        text,
        """        bev_map: torch.Tensor,
        crossview_attention_mask: torch.Tensor,
        condition_keep: torch.Tensor,
        disable_temporal: torch.BoolTensor,
        return_dict: bool = False,
""",
        """        bev_map: torch.Tensor,
        crossview_attention_mask: torch.Tensor,
        condition_keep: torch.Tensor,
        disable_temporal: torch.BoolTensor,
        condition_image_tensor: torch.Tensor = None,
        return_dict: bool = False,
""",
        "model forward signature",
    )
    text = replace_once(
        text,
        """        bev_residuals = self.bev_control(
            bev_map.to(hidden_states.dtype),
            height,
            width,
            condition_keep,
        )
        disable_crossview = torch.zeros_like(disable_temporal, dtype=torch.bool)

        for layer_index, block in enumerate(self.transformer_blocks):
""",
        """        bev_residuals = self.bev_control(
            bev_map.to(hidden_states.dtype),
            height,
            width,
            condition_keep,
        )

        if (
            condition_image_tensor is not None
            and self.condition_image_in_channels is not None
            and condition_image_tensor.shape[-3] != self.condition_image_in_channels
        ):
            raise ValueError(
                "condition_image_tensor has {} channels, model expects {}. "
                "PV box/map/instance-flow must be concatenated consistently.".format(
                    condition_image_tensor.shape[-3],
                    self.condition_image_in_channels,
                )
            )

        condition_residuals = (
            None
            if self.condition_image_adapter is None
            or condition_image_tensor is None
            else self.condition_image_adapter(
                condition_image_tensor.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            )
        )

        disable_crossview = torch.zeros_like(disable_temporal, dtype=torch.bool)

        for layer_index, block in enumerate(self.transformer_blocks):
            # Preserve the original PVTrack adapter semantics: one adapter
            # residual is injected before each successive main SD3 block.
            if condition_residuals is not None and len(condition_residuals) > 0:
                hidden_states = hidden_states + (
                    condition_residuals.pop(0)
                    .flatten(0, 2)
                    .flatten(2)
                    .permute(0, 2, 1)
                )

""",
        "PV residual injection",
    )

    ast.parse(text)
    return text


def build_pipeline(source_text: str) -> str:
    text = source_text

    # Preserve condition_image_adapter.* if a future hybrid checkpoint already
    # contains it. Old keys are removed only when the target model lacks them.
    text = replace_once(
        text,
        """        if old_key.startswith(REMOVED_LEGACY_PREFIXES):
            remapped.pop(old_key, None)
            removed_keys.append(old_key)
            continue
""",
        """        if (
            old_key.startswith(REMOVED_LEGACY_PREFIXES)
            and old_key not in target_state_dict
        ):
            remapped.pop(old_key, None)
            removed_keys.append(old_key)
            continue
""",
        "legacy checkpoint filter",
    )

    text = replace_once(
        text,
        """            "hdmap_bev_images",
            "crossview_mask",
""",
        """            "hdmap_bev_images",
            "3dbox_images",
            "hdmap_images",
            "instance_flow_images",
            "crossview_mask",
""",
        "required PV keys",
    )

    text = replace_once(
        text,
        """        bev_map = batch["hdmap_bev_images"].float()
        crossview_mask = batch["crossview_mask"].bool()

        if condition_keep is None:
            condition_keep = torch.ones(batch_size, dtype=torch.bool)
        condition_keep = condition_keep.bool()

        if classifier_free_guidance:
""",
        """        bev_map = batch["hdmap_bev_images"].float()
        crossview_mask = batch["crossview_mask"].bool()

        # PV branch: exact 3-image-stream concatenation used by PVTrack.
        pv_condition_parts = [
            batch["3dbox_images"].float(),
            batch["hdmap_images"].float(),
            batch["instance_flow_images"].float(),
        ]
        expected_pv_prefix = (batch_size, sequence_length, view_count)
        for name, value in zip(
            ("3dbox_images", "hdmap_images", "instance_flow_images"),
            pv_condition_parts,
        ):
            if value.ndim != 6 or tuple(value.shape[:3]) != expected_pv_prefix:
                raise ValueError(
                    f"{name} must be [B,T,V,C,H,W] with prefix "
                    f"{expected_pv_prefix}, got {tuple(value.shape)}."
                )
        pv_hw = pv_condition_parts[0].shape[-2:]
        if any(value.shape[-2:] != pv_hw for value in pv_condition_parts[1:]):
            raise ValueError(
                "PV condition images must share the same spatial size: "
                + ", ".join(
                    f"{name}={tuple(value.shape[-2:])}"
                    for name, value in zip(
                        ("3dbox_images", "hdmap_images", "instance_flow_images"),
                        pv_condition_parts,
                    )
                )
            )
        condition_image_tensor = torch.cat(pv_condition_parts, dim=-3)

        if condition_keep is None:
            condition_keep = torch.ones(batch_size, dtype=torch.bool)
        condition_keep = condition_keep.bool()

        # Use one keep/drop decision for the complete BEV+PV condition bundle.
        uncondition_image_color = float(
            self.common_config.get("uncondition_image_color", 0.1255)
        )
        pv_keep = condition_keep[:, None, None, None, None, None]
        condition_image_tensor = torch.where(
            pv_keep,
            condition_image_tensor,
            torch.full_like(condition_image_tensor, uncondition_image_color),
        )

        if classifier_free_guidance:
""",
        "PV condition construction",
    )

    text = replace_once(
        text,
        """            crossview_mask = torch.cat(
                [crossview_mask, crossview_mask],
                dim=0,
            )
            condition_keep = torch.cat(
""",
        """            crossview_mask = torch.cat(
                [crossview_mask, crossview_mask],
                dim=0,
            )
            condition_image_tensor = torch.cat(
                [
                    torch.full_like(
                        condition_image_tensor,
                        uncondition_image_color,
                    ),
                    condition_image_tensor,
                ],
                dim=0,
            )
            condition_keep = torch.cat(
""",
        "PV classifier-free guidance",
    )

    text = replace_once(
        text,
        """            "bev_map": bev_map.to(self.device),
            "crossview_attention_mask": crossview_mask.to(self.device),
            "condition_keep": condition_keep.to(self.device),
""",
        """            "bev_map": bev_map.to(self.device),
            "condition_image_tensor": condition_image_tensor.to(
                device=self.device,
                dtype=self.model_dtype,
            ),
            "crossview_attention_mask": crossview_mask.to(self.device),
            "condition_keep": condition_keep.to(self.device),
""",
        "PV model condition return",
    )

    ast.parse(text)
    return text


def sync_bev_and_pv_dataset(bev_source: dict, pv_source: dict) -> dict:
    bev = copy.deepcopy(bev_source)
    pv = copy.deepcopy(pv_source)

    bev_name = bev.get("_class_name", "")
    pv_name = pv.get("_class_name", "")
    bev_family = bev_name.split(".")[-2] if "." in bev_name else bev_name
    pv_family = pv_name.split(".")[-2] if "." in pv_name else pv_name
    if bev_family != pv_family:
        raise ValueError(
            "BEV/PV dataset families do not align: "
            f"{bev_name!r} vs {pv_name!r}."
        )

    # Keep BEV's sampling/camera layout authoritative so its stable slots and
    # loss remain unchanged. Force PV companion to request the same clip.
    for key in (
        "sensor_channels",
        "sequence_length",
        "fps_stride_tuples",
        "split",
        "dataset_name",
    ):
        if key in bev:
            pv[key] = copy.deepcopy(bev[key])

    # Use the current PV config's dataset roots for both halves when the
    # equivalent field exists on each config. This lets the reviewed BEV base
    # config be transplanted into the user's current dataset root without
    # changing BEV-only parameters.
    for key in (
        "fs",
        "dataset_root",
        "balanced_json_path",
        "info_dict_path",
        "index_json_path",
        "sensor_root",
        "pkl_path",
        "cache_root",
        "map_root",
    ):
        if key in bev and key in pv:
            bev[key] = copy.deepcopy(pv[key])

    # Both sides must expose the same CA topology.
    if "stub_key_data_dict" in bev and "stub_key_data_dict" in pv:
        bev_stub = bev["stub_key_data_dict"]
        pv_stub = pv["stub_key_data_dict"]
        if "crossview_mask" in bev_stub:
            pv_stub["crossview_mask"] = copy.deepcopy(bev_stub["crossview_mask"])

    # BEV pipeline uses an empty SD3 prompt, so avoid unnecessary caption IO
    # in the companion PV dataset.
    if "image_description_settings" in pv:
        pv["image_description_settings"] = None

    return {
        "_class_name": "dwm.datasets.lyh.bev_pv.AlignedBEVPVDataset",
        "bev_dataset": bev,
        "pv_dataset": pv,
        "verify_alignment": True,
    }


def add_missing_pv_transforms(target_adapter: dict, pv_adapter: dict) -> None:
    transforms = target_adapter["transform_list"]
    existing = {item["new_key"] for item in transforms}
    for item in pv_adapter["transform_list"]:
        new_key = item.get("new_key")
        if new_key in ("hdmap_images", "instance_flow_images"):
            if new_key not in existing:
                transforms.append(copy.deepcopy(item))
                existing.add(new_key)


def build_config(
    root: Path,
    bev_config: dict,
    pv_config: dict,
    bev_checkpoint: str | None,
    output_path: str | None,
) -> dict:
    config = copy.deepcopy(bev_config)

    if (
        config["pipeline"]["model"]["_class_name"]
        != "dwm.models.bev_models.dit.BEVConditionedSD3TransformerModel"
    ):
        raise ValueError("Embedded/source BEV config is not the reviewed BEV base.")
    if (
        pv_config["pipeline"]["model"]["_class_name"]
        != "dwm.models.crossview_temporal_dit_PLUCKER_track."
           "DiTCrossviewTemporalConditionModel"
    ):
        raise ValueError("Embedded/source PV config is not the reviewed PVTrack config.")

    config["pipeline"]["_class_name"] = "dwm.pipelines.lyh.bev_pv.BEVPipeline"
    config["pipeline"]["common_config"]["uncondition_image_color"] = 0.1255

    model = config["pipeline"]["model"]
    model["_class_name"] = (
        "dwm.models.lyh.bev_pv_plucker.BEVConditionedSD3TransformerModel"
    )
    model["condition_image_adapter_config"] = copy.deepcopy(
        pv_config["pipeline"]["model"]["condition_image_adapter_config"]
    )
    if int(model["condition_image_adapter_config"]["in_channels"]) != 9:
        raise ValueError("Expected the reviewed PV adapter to use 9 input channels.")

    # Keep the BEV base checkpoint by default; it initializes all pre-existing
    # BEV/Pluecker/temporal/normal-CA weights. The PV ImageAdapter is new.
    if bev_checkpoint is not None:
        config["pipeline"]["model_checkpoint_path"] = bev_checkpoint

    # Prefer the current user's SD3/I3D locations from the PV config.
    config["pipeline"]["pretrained_model_name_or_path"] = copy.deepcopy(
        pv_config["pipeline"]["pretrained_model_name_or_path"]
    )
    if (
        "fvd" in config["pipeline"].get("metrics", {})
        and "fvd" in pv_config["pipeline"].get("metrics", {})
    ):
        config["pipeline"]["metrics"]["fvd"][
            "inception_3d_checkpoint_path"
        ] = copy.deepcopy(
            pv_config["pipeline"]["metrics"]["fvd"][
                "inception_3d_checkpoint_path"
            ]
        )

    # Current nuscenes root, while retaining the BEV base device mesh.
    if "nuscenes_fs" in pv_config.get("global_state", {}):
        config["global_state"]["nuscenes_fs"] = copy.deepcopy(
            pv_config["global_state"]["nuscenes_fs"]
        )

    for split_key in ("training_dataset", "validation_dataset"):
        bev_adapter = config[split_key]
        pv_adapter = pv_config[split_key]

        bev_datasets = bev_adapter["base_dataset"]["datasets"]
        pv_datasets = pv_adapter["base_dataset"]["datasets"]
        if len(bev_datasets) != len(pv_datasets):
            raise ValueError(
                f"{split_key} BEV/PV dataset counts differ: "
                f"{len(bev_datasets)} vs {len(pv_datasets)}."
            )

        bev_adapter["base_dataset"]["datasets"] = [
            sync_bev_and_pv_dataset(bev_ds, pv_ds)
            for bev_ds, pv_ds in zip(bev_datasets, pv_datasets)
        ]
        add_missing_pv_transforms(bev_adapter, pv_adapter)

    if output_path is None:
        output_path = str(
            root.parent / "output" / "train_bev_pv_plucker"
        )
    config["output_path"] = output_path

    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="OpenDWM repository root")
    parser.add_argument(
        "--bev-config",
        type=Path,
        default=None,
        help="Optional local BEV-base JSON. Defaults to reviewed uploaded BEV config.",
    )
    parser.add_argument(
        "--pv-config",
        type=Path,
        default=None,
        help="Optional local PVTrack JSON. Defaults to reviewed uploaded PV config.",
    )
    parser.add_argument(
        "--bev-checkpoint",
        default=None,
        help="Override BEV-base initialization checkpoint path.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Override training output path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only generated targets, making timestamped backups.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()

    model_source = root / "src/dwm/models/bev_models/dit.py"
    pipeline_source = root / "src/dwm/pipelines/bev.py"
    condition_source = root / "src/dwm/models/bev_models/condition.py"
    adapter_source = root / "src/dwm/models/adapters.py"
    pv_flow_source = root / "src/dwm/datasets/track_pv/pv_flow.py"

    model_target = root / "src/dwm/models/lyh/bev_pv_plucker.py"
    pipeline_target = root / "src/dwm/pipelines/lyh/bev_pv.py"
    dataset_target = root / "src/dwm/datasets/lyh/bev_pv.py"
    config_target = root / "configs/lyh/BEV_PV_plucker_train.json"

    required_sources = [
        model_source,
        pipeline_source,
        condition_source,
        adapter_source,
        pv_flow_source,
    ]
    for family in ("waymo", "nuscenes", "argoverse", "nuplan"):
        required_sources.extend(
            [
                root / f"src/dwm/datasets/bevs/{family}.py",
                root / f"src/dwm/datasets/track_pv/{family}.py",
            ]
        )
    missing_sources = [path for path in required_sources if not path.is_file()]
    if missing_sources:
        raise SystemExit(
            "Required source file(s) not found:\n"
            + "\n".join(f"  {path}" for path in missing_sources)
        )

    model_source_text = model_source.read_text()
    pipeline_source_text = pipeline_source.read_text()

    for marker in (
        "class BEVConditionedSD3TransformerModel",
        "self.bev_control = TemporalBEVResidualAdapter",
        "self.cond_cross_blocks = torch.nn.ModuleList",
        "self.crossview_transformer_blocks = torch.nn.ModuleList",
        "self.rayencoder = PluckerEncoder",
    ):
        if marker not in model_source_text:
            raise SystemExit(f"BEV model is missing expected marker: {marker}")

    for marker in (
        "class BEVPipeline",
        "def prepare_model_conditions",
        '"hdmap_bev_images"',
        "bbox_token_corners",
        "REMOVED_LEGACY_PREFIXES",
    ):
        if marker not in pipeline_source_text:
            raise SystemExit(f"BEV pipeline is missing expected marker: {marker}")

    bev_config = (
        json.loads(args.bev_config.read_text())
        if args.bev_config is not None
        else json.loads(EMBEDDED_BEV_CONFIG)
    )
    pv_config = (
        json.loads(args.pv_config.read_text())
        if args.pv_config is not None
        else json.loads(EMBEDDED_PV_CONFIG)
    )

    print("=== BEFORE git status ===")
    print(git_status(root) or "(clean or git unavailable)")
    print()

    for target in (
        model_target,
        pipeline_target,
        dataset_target,
        config_target,
    ):
        ensure_target(target, args.force)

    for directory in (
        model_target.parent,
        pipeline_target.parent,
        dataset_target.parent,
        config_target.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    for init_path in (
        model_target.parent / "__init__.py",
        pipeline_target.parent / "__init__.py",
        dataset_target.parent / "__init__.py",
    ):
        if not init_path.exists():
            init_path.write_text("")

    new_model = build_model(model_source_text)
    new_pipeline = build_pipeline(pipeline_source_text)
    ast.parse(DATASET_WRAPPER)

    model_target.write_text(new_model)
    pipeline_target.write_text(new_pipeline)
    dataset_target.write_text(DATASET_WRAPPER)

    new_config = build_config(
        root,
        bev_config,
        pv_config,
        args.bev_checkpoint,
        args.output_path,
    )
    config_target.write_text(
        json.dumps(new_config, indent=4, ensure_ascii=False) + "\n"
    )

    py_compile.compile(str(model_target), doraise=True)
    py_compile.compile(str(pipeline_target), doraise=True)
    py_compile.compile(str(dataset_target), doraise=True)
    json.loads(config_target.read_text())

    print("=== CREATED ===")
    print(model_target)
    print(pipeline_target)
    print(dataset_target)
    print(config_target)
    print()

    print("=== ARCHITECTURE ===")
    print("BEV base: stable bbox + trajectory + BEV residual + cond cross-attn")
    print("PV: 3dbox + hdmap + instance-flow -> 9ch -> ImageAdapter residual")
    print("Geometry: Pluecker")
    print("Cross-view: normal rowwise CA")
    print("Temporal: original BEV temporal attention")
    print("URoPE: disabled/not present")
    print("TV: disabled/not present")
    print()

    model_cfg = new_config["pipeline"]["model"]
    print("=== CONFIG CHECK ===")
    print("pipeline =", new_config["pipeline"]["_class_name"])
    print("model =", model_cfg["_class_name"])
    print("block_layers =", model_cfg["block_layers"])
    print(
        "PV channels =",
        model_cfg["condition_image_adapter_config"]["in_channels"],
    )
    print("bev_in_channels =", model_cfg["bev_in_channels"])
    print(
        "BEV checkpoint =",
        new_config["pipeline"].get("model_checkpoint_path"),
    )
    print("output_path =", new_config["output_path"])
    print()

    for split_key in ("training_dataset", "validation_dataset"):
        transforms = [
            item["new_key"]
            for item in new_config[split_key]["transform_list"]
        ]
        print(f"{split_key} transforms =", transforms)

    print()
    print("=== IMPORTANT ===")
    print(
        "AlignedBEVPVDataset reads BEV and PV companion datasets at the same "
        "index and verifies geometry/mask alignment."
    )
    print(
        "This first implementation favors correctness and reuse over I/O "
        "efficiency; it may read the same underlying clip twice."
    )
    print(
        "device_mesh remains the BEV-base value; make it match torchrun "
        "world size before multi-GPU training."
    )
    print()

    print("=== AFTER git status ===")
    print(git_status(root) or "(clean or git unavailable)")


if __name__ == "__main__":
    main()
