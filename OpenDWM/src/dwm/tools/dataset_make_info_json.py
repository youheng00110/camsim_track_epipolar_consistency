import argparse
import dwm.tools.fs_make_info_json
import fsspec.implementations.local
import json
import os
import struct
import re


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to make information JSON(s) for dataset to "
        "accelerate initialization.")
    parser.add_argument(
        "-dt", "--dataset-type", type=str,
        choices=["nuscenes", "waymo", "argoverse"], required=True,
        help="The dataset type.")
    parser.add_argument(
        "-s", "--split", default=None, type=str,
        help="The split, optional depending on the dataset type.")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True,
        help="The path of the dataset root.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path to save the information JSON file(s) on the local file "
        "system.")
    parser.add_argument(
        "-fs", "--fs-config-path", default=None, type=str,
        help="The path of file system JSON config to open the dataset.")
    return parser


if __name__ == "__main__":
    import tqdm

    parser = create_parser()
    args = parser.parse_args()

    if args.fs_config_path is None:
        fs = fsspec.implementations.local.LocalFileSystem()
    else:
        import dwm.common
        with open(args.fs_config_path, "r", encoding="utf-8") as f:
            fs = dwm.common.create_instance_from_config(json.load(f))

    if args.dataset_type == "nuscenes":
        files = [
            os.path.relpath(i, args.input_path)
            for i in fs.ls(args.input_path, detail=False)
        ]
        filtered_files = [
            i for i in files
            if (
                (args.split is None or i.startswith(args.split)) and
                i.endswith(".zip")
            )
        ]
        assert len(filtered_files) > 0, (
            "No files detected, please check the split (one of \"v1.0-mini\", "
            "\"v1.0-trainval\", \"v1.0-test\") is correct, and ensure the "
            "blob files are already converted to the ZIP format."
        )

        os.makedirs(args.output_path, exist_ok=True)
        for i in tqdm.tqdm(filtered_files):
            with fs.open("{}/{}".format(args.input_path, i)) as f:
                items = dwm.tools.fs_make_info_json.make_info_dict(
                    os.path.splitext(i)[-1], f)

            with open(
                os.path.join(
                    args.output_path, i.replace(".zip", ".info.json")),
                "w", encoding="utf-8"
            ) as f:
                json.dump(items, f)

    elif args.dataset_type == "waymo":
        import waymo_open_dataset.dataset_pb2 as waymo_pb

        files = [
            os.path.relpath(i, args.input_path)
            for i in fs.ls(args.input_path, detail=False)
            if i.endswith(".tfrecord")
        ]
        assert len(files) > 0, "No files detected."

        pattern = re.compile(
            "^segment-(?P<scene>.*)_with_camera_labels.tfrecord$")
        info_dict = {}
        for i in tqdm.tqdm(files):
            match = re.match(pattern, i)
            scene = match.group("scene")
            pt = 0
            info_list = []
            with fs.open("{}/{}".format(args.input_path, i)) as f:
                while True:
                    start = f.read(8)
                    if len(start) == 0:
                        break

                    size, = struct.unpack("<Q", start)
                    f.seek(pt + 12)
                    frame = waymo_pb.Frame()
                    frame.ParseFromString(f.read(size))
                    info_list.append([frame.timestamp_micros, size, pt + 12])

                    pt += size + 16
                    f.seek(pt)

            info_dict[scene] = info_list

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(info_dict, f)

    elif args.dataset_type == "argoverse":
        # 适配已解压的Argoverse数据集（不再筛选.tar文件）
        import glob


        # 遍历已解压的train/val/test目录下的场景文件夹
        # 先确定目标拆分目录（train/val/test），默认用train
        split_dir = args.split if args.split else "val"
        split_path = os.path.join(args.input_path, split_dir)
        
        # 检查拆分目录是否存在
        assert os.path.exists(split_path), f"拆分目录不存在: {split_path}"
        
        # 获取所有场景文件夹（Argoverse解压后每个场景是一个独立文件夹）
        scene_folders = [
            os.path.relpath(i, args.input_path)
            for i in glob.glob(os.path.join(split_path, "*"))
            if os.path.isdir(i)
        ]
        
        # 检查是否有场景文件夹
        assert len(scene_folders) > 0, (
            "No scene folders detected, please check the input path or split (one of \"train\", "
            "\"val\", \"test\") is correct."
        )

        # 创建输出目录
        os.makedirs(args.output_path, exist_ok=True)
        
        # 为每个场景文件夹生成info.json（模拟原tar包的info逻辑）
        for scene_folder in tqdm.tqdm(scene_folders):
            scene_path = os.path.join(args.input_path, scene_folder)
            # 构建该场景的信息字典（核心：适配Argoverse解压后的数据结构）
            items = {
                "scene_name": os.path.basename(scene_folder),
                "split": split_dir,
                "path": scene_path,
                "files": [
                    os.path.join(scene_folder, f)
                    for f in glob.glob(os.path.join(scene_path, "**/*"), recursive=True)
                    if os.path.isfile(f)
                ]
            }
            
            # 保存info.json文件（命名规则：场景名.info.json）
            info_filename = f"{os.path.basename(scene_folder)}.info.json"
            with open(
                os.path.join(args.output_path, info_filename),
                "w", encoding="utf-8"
            ) as f:
                json.dump(items, f, indent=2)  # indent=2让JSON更易读
    else:
        raise Exception("Unknown dataset type {}.".format(args.dataset_type))
