#!/usr/bin/env python3
import argparse
import shutil
from datetime import datetime
from pathlib import Path


ANCHOR = """        load_args = {"strict": False}
        if model_load_state_args is not None:
            load_args.update(model_load_state_args)
        missing_keys, unexpected_keys = self.model.load_state_dict(
            state_dict,
            **load_args,
        )
"""

REPLACEMENT = """        # Expand legacy PV ImageAdapter input weights when the checkpoint
        # was trained with fewer image-condition channels (e.g. 6ch box+map)
        # than the current model (9ch box+map+instance-flow).
        #
        # Keep all existing channel weights exactly and zero-initialize only
        # the newly added input channels. This matches the existing PVTrack
        # checkpoint-loading behavior.
        target_state_dict = self.model.state_dict()
        expanded_condition_keys = []
        for key, value in list(state_dict.items()):
            if not key.startswith("condition_image_adapter."):
                continue
            if key not in target_state_dict:
                continue

            target_value = target_state_dict[key]
            if tuple(value.shape) == tuple(target_value.shape):
                continue

            if value.ndim < 2 or value.ndim != target_value.ndim:
                continue

            same_non_channel_shape = (
                value.shape[0] == target_value.shape[0]
                and tuple(value.shape[2:]) == tuple(target_value.shape[2:])
            )
            if (
                same_non_channel_shape
                and value.shape[1] < target_value.shape[1]
            ):
                expanded_value = torch.zeros_like(
                    target_value,
                    device="cpu",
                )
                expanded_value[:, :value.shape[1]] = value.to(
                    dtype=expanded_value.dtype,
                    device="cpu",
                )
                state_dict[key] = expanded_value
                expanded_condition_keys.append(
                    (
                        key,
                        tuple(value.shape),
                        tuple(target_value.shape),
                    )
                )

        if self.should_save and expanded_condition_keys:
            print(
                "expanded condition adapter input weights:",
                expanded_condition_keys,
                flush=True,
            )

        load_args = {"strict": False}
        if model_load_state_args is not None:
            load_args.update(model_load_state_args)
        missing_keys, unexpected_keys = self.model.load_state_dict(
            state_dict,
            **load_args,
        )
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    path = root / "src/dwm/pipelines/lyh/bev_pv.py"

    if not path.is_file():
        raise SystemExit(f"not found: {path}")

    text = path.read_text()

    if "expanded condition adapter input weights:" in text:
        print("Already patched:", path)
        return

    count = text.count(ANCHOR)
    if count != 1:
        raise SystemExit(
            f"Expected exactly one load_state_dict anchor, found {count}. "
            "Refusing to modify the file."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.name + f".bak.{stamp}")
    shutil.copy2(path, backup)

    path.write_text(text.replace(ANCHOR, REPLACEMENT, 1))

    compile(path.read_text(), str(path), "exec")

    print("PATCHED:", path)
    print("BACKUP :", backup)
    print()
    print("Expected next startup log:")
    print(
        "expanded condition adapter input weights: "
        "[('condition_image_adapter.body.0.in_conv.weight', "
        "(1536, 384, 1, 1), (1536, 576, 1, 1))]"
    )


if __name__ == "__main__":
    main()
