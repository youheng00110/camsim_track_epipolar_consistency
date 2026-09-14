import argparse
import json
import os
import shutil
from pathlib import Path


def create_parser():
    parser = argparse.ArgumentParser(
        description="Merge distributed rank preview outputs into one paired manifest."
    )
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--manifest-name", type=str, default="stflow_manifest.jsonl")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_manifest(path):
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items.append(json.loads(line))

    return items


def rank_sort_key(path):
    name = path.name

    if name.startswith("rank_"):
        suffix = name.split("rank_", 1)[1]
        if suffix.isdigit():
            return int(suffix)

    return 10**9


def copy_file_if_needed(src, dst, use_copy):
    if not os.path.exists(src):
        raise FileNotFoundError(src)

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if use_copy:
        shutil.copy2(src, dst)
    else:
        os.link(src, dst)


def remap_path_and_copy(input_rank_root, output_root, old_rel_path, old_video_id, new_video_id, use_copy):
    if old_rel_path is None:
        return None

    new_rel_path = old_rel_path.replace(old_video_id, new_video_id)

    src = os.path.join(input_rank_root, old_rel_path)
    dst = os.path.join(output_root, new_rel_path)

    copy_file_if_needed(src, dst, use_copy)

    return new_rel_path


def infer_dataset_name(items, fallback):
    if fallback is not None:
        return fallback

    if len(items) == 0:
        return "video"

    video_id = str(items[0].get("video_id", "video_000000"))
    if "_video_" in video_id:
        return video_id.split("_video_")[0]

    return "video"


def update_manifest_item(item, input_rank_root, output_root, new_video_id, use_copy):
    old_video_id = item["video_id"]
    item["video_id"] = new_video_id

    for frame in item["frames"]:
        for view in frame["views"]:
            view["image_path"] = remap_path_and_copy(
                input_rank_root,
                output_root,
                view["image_path"],
                old_video_id,
                new_video_id,
                use_copy,
            )

            if view.get("real_image_path", None) is not None:
                view["real_image_path"] = remap_path_and_copy(
                    input_rank_root,
                    output_root,
                    view["real_image_path"],
                    old_video_id,
                    new_video_id,
                    use_copy,
                )

            if view.get("valid_mask_path", None) is not None:
                view["valid_mask_path"] = remap_path_and_copy(
                    input_rank_root,
                    output_root,
                    view["valid_mask_path"],
                    old_video_id,
                    new_video_id,
                    use_copy,
                )

    return item


def main():
    args = create_parser().parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if output_root.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"output-root already exists: {output_root}. "
                "Use --overwrite or choose another output-root."
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    rank_dirs = [
        path for path in input_root.iterdir()
        if path.is_dir() and path.name.startswith("rank_")
    ]
    rank_dirs = sorted(rank_dirs, key=rank_sort_key)

    if len(rank_dirs) == 0:
        raise RuntimeError(f"No rank_* directories found in {input_root}")

    merged_items = []
    global_index = 0
    dataset_name = args.dataset_name

    for rank_dir in rank_dirs:
        manifest_path = rank_dir / args.manifest_name

        if not manifest_path.exists():
            print(f"[skip] missing manifest: {manifest_path}", flush=True)
            continue

        items = load_manifest(manifest_path)

        if dataset_name is None:
            dataset_name = infer_dataset_name(items, args.dataset_name)

        print(
            f"[merge] {rank_dir.name}: {len(items)} videos",
            flush=True,
        )

        for item in items:
            if args.max_videos is not None and global_index >= args.max_videos:
                break

            new_video_id = f"{dataset_name}_video_{global_index:06d}"
            updated_item = update_manifest_item(
                item,
                str(rank_dir),
                str(output_root),
                new_video_id,
                args.copy,
            )
            merged_items.append(updated_item)
            global_index += 1

        if args.max_videos is not None and global_index >= args.max_videos:
            break

    manifest_out = output_root / args.manifest_name
    with open(manifest_out, "w", encoding="utf-8") as f:
        for item in merged_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[done] merged videos: {len(merged_items)}")
    print(f"[done] output root: {output_root}")
    print(f"[done] manifest: {manifest_out}")


if __name__ == "__main__":
    main()
