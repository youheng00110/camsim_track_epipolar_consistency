import argparse
import json
import os
import shutil
from pathlib import Path


def create_parser():
    parser = argparse.ArgumentParser(
        description="Interleave rank_* preview manifests into one paired manifest."
    )
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--manifest-name", type=str, default="stflow_manifest.jsonl")
    parser.add_argument("--dataset-name", type=str, default="nuplan")
    parser.add_argument("--max-videos", type=int, default=200)
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_manifest(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def rank_sort_key(path):
    name = path.name
    if name.startswith("rank_"):
        suffix = name.split("rank_", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return 10**9


def copy_or_link(src, dst, use_copy):
    if not os.path.exists(src):
        raise FileNotFoundError(src)

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if os.path.exists(dst):
        os.remove(dst)

    if use_copy:
        shutil.copy2(src, dst)
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def remap_one_path(input_rank_root, output_root, old_path, old_video_id, new_video_id, use_copy):
    if old_path is None:
        return None

    new_path = old_path.replace(old_video_id, new_video_id)

    if os.path.isabs(old_path):
        src = old_path
    else:
        src = os.path.join(input_rank_root, old_path)

    if os.path.isabs(new_path):
        dst = new_path
    else:
        dst = os.path.join(output_root, new_path)

    copy_or_link(src, dst, use_copy)

    return new_path


def update_item_paths(item, input_rank_root, output_root, new_video_id, use_copy):
    old_video_id = item["video_id"]
    item["video_id"] = new_video_id

    for frame in item["frames"]:
        for view in frame["views"]:
            for key in ["image_path", "real_image_path", "valid_mask_path"]:
                if key in view and view[key] is not None:
                    view[key] = remap_one_path(
                        input_rank_root,
                        output_root,
                        view[key],
                        old_video_id,
                        new_video_id,
                        use_copy,
                    )

    return item


def get_rank_manifests(input_root, manifest_name):
    rank_dirs = [
        path for path in Path(input_root).iterdir()
        if path.is_dir() and path.name.startswith("rank_")
    ]
    rank_dirs = sorted(rank_dirs, key=rank_sort_key)

    rank_manifests = []
    for rank_dir in rank_dirs:
        manifest_path = rank_dir / manifest_name
        if not manifest_path.exists():
            print(f"[skip] missing manifest: {manifest_path}", flush=True)
            continue

        items = load_manifest(manifest_path)
        print(f"[load] {rank_dir.name}: {len(items)} videos", flush=True)
        rank_manifests.append((rank_dir, items))

    return rank_manifests


def main():
    args = create_parser().parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if output_root.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"output-root already exists: {output_root}. Use --overwrite."
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    rank_manifests = get_rank_manifests(input_root, args.manifest_name)
    if len(rank_manifests) == 0:
        raise RuntimeError(f"No valid rank manifests found in {input_root}")

    max_len = max(len(items) for _, items in rank_manifests)

    merged_items = []
    global_index = 0

    for local_index in range(max_len):
        for rank_dir, items in rank_manifests:
            if local_index >= len(items):
                continue
            if args.max_videos is not None and global_index >= args.max_videos:
                break

            item = items[local_index]
            new_video_id = f"{args.dataset_name}_video_{global_index:06d}"

            updated = update_item_paths(
                item,
                str(rank_dir),
                str(output_root),
                new_video_id,
                args.copy,
            )
            merged_items.append(updated)
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
