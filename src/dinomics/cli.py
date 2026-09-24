"""Command-line entry point for dinomics feature extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from dinomics.extraction import extract_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract DINOv2/DINOv3 CLS and patch features from a 2D or 3D medical image."
    )
    parser.add_argument("--image", required=True, help="Path to a 2D or 3D medical image")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML or TOML config (defaults to DINOv2-base settings)",
    )
    parser.add_argument("--mask", default=None, help="Optional ROI mask (same space or resampled)")
    parser.add_argument("--label-mask", default=None, help="Optional label map for per-patch labels")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz path. Defaults to <image_stem>_dino_features.npz next to the image.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_path = Path(args.image)
    output = Path(args.output) if args.output else image_path.with_name(f"{image_path.stem}_dino_features.npz")

    result = extract_features(
        image=image_path,
        config=args.config,
        mask=args.mask,
        label_mask=args.label_mask,
        output_path=output,
    )
    print(f"Saved features to {output}")
    print(f"  CLS embeddings:   {result.cls_embeddings.shape}")
    print(f"  Patch embeddings: {result.patch_embeddings.shape}")
    print(f"  Slices:           {result.slice_indices.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
