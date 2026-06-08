"""
Exports a finetune.py checkpoint into a demucs-loadable model file.

Takes the trained weights from a checkpoint and writes a serialized demucs
package into models/sonaveil_v1/, which app.py loads in-process for inference.

Note: we load/serialize with the demucs package format (klass + args + state),
but inference must load it with weights_only=False (torch >= 2.6 default flip),
because the demucs CLI's local --repo loader is broken under new torch. app.py
does this in-process rather than shelling out to the demucs CLI.

Usage:
    python training/export.py --checkpoint ckpt/best.pt --name sonaveil_v1
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model
from demucs.states import save_with_checksum, serialize_model

BASE_MODEL = "htdemucs_6s"


def build_skeleton(sources: list[str]) -> HTDemucs:
    """Rebuild the 2-source architecture (config only; weights come from ckpt)."""
    pre = get_model(BASE_MODEL).models[0]
    args, kwargs = pre._init_args_kwargs
    kwargs = dict(kwargs)
    kwargs["sources"] = sources
    return HTDemucs(*args, **kwargs)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--name", default="sonaveil_v1")
    p.add_argument("--out-dir", type=Path, default=Path("models"))
    args = p.parse_args()

    if not args.checkpoint.exists():
        print(f"error: checkpoint not found: {args.checkpoint}")
        return 1

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sources = ck.get("sources", ["piano", "orchestra"])
    model = build_skeleton(sources)
    model.load_state_dict(ck["model"])
    model.eval()

    out_dir = args.out_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear any previous export so a stale checksum file can't linger.
    for old in out_dir.glob(f"{args.name}*.th"):
        old.unlink()

    package = serialize_model(model, training_args=OmegaConf.create({}), half=False)
    save_with_checksum(package, out_dir / f"{args.name}.th")

    written = next(out_dir.glob(f"{args.name}*.th"))
    print(f"exported {sources} model -> {written}")
    print(f"app.py should load this file in-process with weights_only=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
