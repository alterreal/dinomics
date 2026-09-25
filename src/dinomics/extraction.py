"""CLS and patch feature extraction for 2D images and 3D volumes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from dinomics.config import DinomicsConfig, load_config
from dinomics.io import ImageLike, Volume, align_mask, load_volume
from dinomics.model import load_dino_model
from dinomics.preprocess import (
    apply_intensity,
    apply_mask_to_patch_embeddings,
    content_mask_from_pad,
    create_rgb_from_grayscale,
    crop_centered_on_mask,
    pad_to_square,
    process_mask,
)


@dataclass
class FeatureResult:
    """Embeddings extracted from one image or volume.

    Shapes use ``N`` processed slices, ``G`` the patch grid size, and ``D`` the
    embedding dimension.

    Attributes
    ----------
    cls_embeddings
        CLS (slice) embeddings, shape ``(N, D)``.
    patch_embeddings
        Spatial patch embeddings, shape ``(N, G, G, D)``.
    patch_embeddings_maxpooled
        Max-pooled patch embeddings per slice, shape ``(N, D)``.
    patch_mask
        Patch occupancy from the ROI mask, shape ``(N, G, G)``, or ``None``.
    patch_labels
        Optional per-patch labels from ``label_mask``, shape ``(N, G, G)``.
    slice_indices
        Original slice indices along the configured axis.
    patch_embeddings_flat
        Optional flat patches, shape ``(N, G*G, D)``.
        metadata
        Image path, model name, crop window, letterbox pad, and related bookkeeping.
    """

    cls_embeddings: np.ndarray
    patch_embeddings: np.ndarray
    patch_embeddings_maxpooled: np.ndarray
    slice_indices: np.ndarray
    patch_mask: np.ndarray | None = None
    patch_labels: np.ndarray | None = None
    patch_embeddings_flat: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "cls_embeddings": self.cls_embeddings,
            "global_embeddings": self.cls_embeddings,
            "patch_embeddings": self.patch_embeddings,
            "patch_embeddings_maxpooled": self.patch_embeddings_maxpooled,
            "slice_indices": self.slice_indices,
        }
        if self.patch_mask is not None:
            payload["patch_mask"] = self.patch_mask
        if self.patch_labels is not None:
            payload["patch_labels"] = self.patch_labels
        if self.patch_embeddings_flat is not None:
            payload["patch_embeddings_flat"] = self.patch_embeddings_flat
        payload["metadata"] = np.array(self.metadata, dtype=object)
        return payload

    def aggregate_patches(self, reducers, mask: np.ndarray | None = None) -> np.ndarray:
        """Case-level vector from patch tokens. See :func:`dinomics.aggregate_patches`."""
        from dinomics.aggregate import aggregate_patches

        occ = self.patch_mask if mask is None else mask
        return aggregate_patches(self.patch_embeddings, reducers, mask=occ)

    def aggregate_slices(self, reducers) -> np.ndarray:
        """Case-level vector from CLS embeddings. See :func:`dinomics.aggregate_slices`."""
        from dinomics.aggregate import aggregate_slices

        return aggregate_slices(self.cls_embeddings, reducers)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.to_dict())
        return path

    @classmethod
    def load(cls, path: str | Path) -> FeatureResult:
        with np.load(path, allow_pickle=True) as data:
            metadata = data["metadata"].item() if "metadata" in data else {}
            cls_key = "cls_embeddings" if "cls_embeddings" in data else "global_embeddings"
            return cls(
                cls_embeddings=data[cls_key],
                patch_embeddings=data["patch_embeddings"],
                patch_embeddings_maxpooled=data["patch_embeddings_maxpooled"],
                slice_indices=data["slice_indices"],
                patch_mask=data["patch_mask"] if "patch_mask" in data else None,
                patch_labels=data["patch_labels"] if "patch_labels" in data else None,
                patch_embeddings_flat=(
                    data["patch_embeddings_flat"] if "patch_embeddings_flat" in data else None
                ),
                metadata=metadata,
            )


def extract_dino_embeddings(
    rgb_image: np.ndarray,
    processor,
    model,
    n_register_tokens: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract CLS and patch tokens from a single RGB numpy image."""
    inputs = processor(images=rgb_image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state
    cls_embedding = hidden[:, 0, :].squeeze(0).detach().cpu().numpy()
    patch_start = 1 + int(n_register_tokens)
    patch_embeddings = hidden[:, patch_start:, :].squeeze(0).detach().cpu().numpy()
    return cls_embedding, patch_embeddings


class DinoExtractor:
    """Reusable DINOv2/DINOv3 extractor. Load the model once, extract many images."""

    def __init__(self, config: DinomicsConfig | str | Path | dict[str, Any] | None = None):
        self.config = load_config(config)
        self.processor, self.model, self.device = load_dino_model(self.config.model)

    @classmethod
    def from_config(cls, config: DinomicsConfig | str | Path | dict[str, Any] | None = None) -> DinoExtractor:
        return cls(config)

    def extract(
        self,
        image: ImageLike,
        mask: ImageLike | None = None,
        label_mask: ImageLike | None = None,
        output_path: str | Path | None = None,
    ) -> FeatureResult:
        """Extract CLS and patch features from a 2D image or 3D volume (slice-wise)."""
        volume = load_volume(image)
        mask_vol = load_volume(mask) if mask is not None else None
        label_vol = load_volume(label_mask) if label_mask is not None else None

        if mask_vol is not None:
            mask_vol = align_mask(mask_vol, volume)
            mask_vol.array = (mask_vol.array > 0).astype(np.uint8)
        if label_vol is not None:
            label_vol = align_mask(label_vol, volume)

        array = volume.array
        mask_arr = mask_vol.array if mask_vol is not None else None
        label_arr = label_vol.array if label_vol is not None else None
        crop_region = None

        settings = self.config.extraction
        if settings.slice_axis != 0:
            array = np.moveaxis(array, settings.slice_axis, 0)
            if mask_arr is not None:
                mask_arr = np.moveaxis(mask_arr, settings.slice_axis, 0)
            if label_arr is not None:
                label_arr = np.moveaxis(label_arr, settings.slice_axis, 0)

        original_z: list[int] | None = None
        n_input_slices = int(array.shape[0])
        if mask_arr is not None and (settings.remove_small_holes or settings.mirror_mask_y):
            mask_arr = process_mask(
                mask_arr,
                remove_holes=settings.remove_small_holes,
                hole_area_threshold=settings.hole_area_threshold,
                mirror_y=settings.mirror_mask_y,
            )
        if settings.crop_to_mask:
            if mask_arr is None:
                raise ValueError("crop_to_mask=True requires a mask")
            extras = [label_arr] if label_arr is not None else None
            array, mask_arr, extras_out, crop_region = crop_centered_on_mask(
                array,
                mask_arr,
                crop_size=settings.crop_size,
                min_mask_pixels=settings.min_mask_pixels,
                extras=extras,
            )
            if extras_out:
                label_arr = extras_out[0]
            original_z = crop_region["valid_slice_indices"]

        if settings.apply_image_mask and mask_arr is not None:
            array = array * mask_arr

        select_settings = settings
        if original_z is not None and settings.slice_indices is not None:
            wanted = {int(i) for i in settings.slice_indices}
            select_settings = replace(
                settings,
                slice_indices=[i for i, z in enumerate(original_z) if z in wanted],
            )

        loop_indices = _select_slice_indices(
            num_slices=array.shape[0],
            array=array,
            mask_arr=mask_arr,
            settings=select_settings,
        )
        slice_indices = (
            [original_z[i] for i in loop_indices] if original_z is not None else loop_indices
        )

        cls_embs: list[np.ndarray] = []
        maxpool_embs: list[np.ndarray] = []
        patch_embs: list[np.ndarray] = []
        patch_masks: list[np.ndarray] = []
        patch_labels_all: list[np.ndarray | None] = []
        flat_patches: list[np.ndarray] = []

        n_register = int(self.config.model.n_register_tokens or 0)
        include_flat = self.config.output.include_flat_patches
        square_pad: dict | None = None

        for slice_idx in tqdm(loop_indices, desc="Extracting DINO features", leave=False):
            img_slice = array[slice_idx]
            mask_slice = mask_arr[slice_idx] if mask_arr is not None else None
            label_slice = label_arr[slice_idx] if label_arr is not None else None

            img_slice = apply_intensity(img_slice, self.config.intensity)
            rgb_slice = create_rgb_from_grayscale(img_slice)
            if mask_arr is None:
                rgb_slice, square_pad = pad_to_square(rgb_slice, fill=0)
                if label_slice is not None:
                    label_slice, _ = pad_to_square(label_slice, fill=0)
            cls_emb, patch_emb = extract_dino_embeddings(
                rgb_slice,
                self.processor,
                self.model,
                n_register_tokens=n_register,
                device=self.device,
            )

            patch_mask = None
            patch_labels = None
            if (
                mask_slice is None
                and settings.apply_patch_mask
                and square_pad is not None
                and square_pad["padded"]
            ):
                patch_emb, patch_labels, patch_mask = apply_mask_to_patch_embeddings(
                    patch_emb, content_mask_from_pad(square_pad), mask_label=label_slice
                )
            elif settings.apply_patch_mask and mask_slice is not None:
                patch_emb, patch_labels, patch_mask = apply_mask_to_patch_embeddings(
                    patch_emb, mask_slice, mask_label=label_slice
                )
            elif label_slice is not None:
                _, patch_labels, patch_mask = apply_mask_to_patch_embeddings(
                    patch_emb, label_slice if mask_slice is None else mask_slice, mask_label=label_slice
                )

            grid_size = int(patch_emb.shape[0] ** 0.5)
            if grid_size * grid_size != patch_emb.shape[0]:
                raise ValueError(
                    f"Patch count {patch_emb.shape[0]} is not a square grid. "
                    "Check model image_size / architecture."
                )

            patch_spatial = patch_emb.reshape(grid_size, grid_size, patch_emb.shape[1])
            cls_embs.append(cls_emb)
            maxpool_embs.append(patch_emb.max(axis=0))
            patch_embs.append(patch_spatial)
            if include_flat:
                flat_patches.append(patch_emb)
            if patch_mask is not None:
                patch_masks.append(patch_mask.reshape(grid_size, grid_size))
            if patch_labels is not None:
                patch_labels_all.append(patch_labels.reshape(grid_size, grid_size))

        if not cls_embs:
            raise RuntimeError("No slices were processed. Check the image, mask, and extraction settings.")

        result = FeatureResult(
            cls_embeddings=np.asarray(cls_embs),
            patch_embeddings=np.asarray(patch_embs),
            patch_embeddings_maxpooled=np.asarray(maxpool_embs),
            slice_indices=np.asarray(slice_indices, dtype=np.int32),
            patch_mask=np.asarray(patch_masks) if patch_masks else None,
            patch_labels=np.asarray(patch_labels_all) if patch_labels_all and patch_labels_all[0] is not None else None,
            patch_embeddings_flat=np.asarray(flat_patches) if flat_patches else None,
            metadata={
                "image_path": volume.path,
                "is_2d": volume.is_2d,
                "model_name": self.config.model.name,
                "feature_type": self.config.model.feature_type,
                "image_size": self.config.model.image_size,
                "n_register_tokens": n_register,
                "device": str(self.device),
                "crop_region": crop_region,
                "square_pad": square_pad,
                "mask_processing": {
                    "remove_small_holes": bool(settings.remove_small_holes),
                    "hole_area_threshold": int(settings.hole_area_threshold),
                    "mirror_mask_y": bool(settings.mirror_mask_y),
                },
                "spacing": volume.spacing,
                "num_input_slices": n_input_slices,
                "config": asdict(self.config),
            },
        )
        if output_path is not None:
            result.save(output_path)
        return result


def extract_features(
    image: ImageLike,
    config: DinomicsConfig | str | Path | dict[str, Any] | None = None,
    mask: ImageLike | None = None,
    label_mask: ImageLike | None = None,
    output_path: str | Path | None = None,
) -> FeatureResult:
    """Extract CLS and patch features using a config file, dict, or :class:`DinomicsConfig`.

    Parameters
    ----------
    image
        Path, SimpleITK image, or numpy array (2D or 3D).
    config
        YAML/TOML path, mapping, or :class:`DinomicsConfig`. Defaults are DINOv2-base.
    mask
        Optional ROI mask. Background voxels/patches can be zeroed per the config.
    label_mask
        Optional label map used to derive per-patch labels.
    output_path
        If set, save a compressed ``.npz`` next to returning the result.
    """
    extractor = DinoExtractor.from_config(config)
    return extractor.extract(image, mask=mask, label_mask=label_mask, output_path=output_path)


def _select_slice_indices(
    num_slices: int,
    array: np.ndarray,
    mask_arr: np.ndarray | None,
    settings,
) -> list[int]:
    if settings.slice_indices is not None:
        indices = [int(i) for i in settings.slice_indices]
    else:
        indices = list(range(0, num_slices, max(int(settings.slice_step), 1)))

    indices = [i for i in indices if 0 <= i < num_slices]

    if settings.skip_empty_slices:
        if mask_arr is not None:
            indices = [
                i for i in indices if int(np.sum(mask_arr[i] > 0)) >= settings.min_mask_pixels
            ]
        else:
            indices = [i for i in indices if np.any(array[i])]

    if settings.max_slices is not None:
        indices = indices[: int(settings.max_slices)]
    return indices
