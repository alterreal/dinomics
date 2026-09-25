"""Slice-level preprocessing shared by DINOv2 and DINOv3."""

from __future__ import annotations

import inspect

import numpy as np
import torch

from skimage.morphology import remove_small_holes

from dinomics.config import IntensityConfig


def pad_to_square(array: np.ndarray, fill: float | int = 0) -> tuple[np.ndarray, dict]:
    """Center-pad a 2D slice (or ``H×W×C``) so height equals width.

    Used when no ROI mask is given, so the HuggingFace square resize does not
    stretch the anatomy. Already-square inputs are returned unchanged.
    """
    arr = np.asarray(array)
    if arr.ndim < 2:
        raise ValueError(f"Expected at least a 2D array, got shape {arr.shape}")

    height, width = int(arr.shape[0]), int(arr.shape[1])
    size = max(height, width)
    top = (size - height) // 2
    left = (size - width) // 2
    bottom = size - height - top
    right = size - width - left
    info = {
        "padded": bool(top or left or bottom or right),
        "top": int(top),
        "bottom": int(bottom),
        "left": int(left),
        "right": int(right),
        "content_shape": (height, width),
        "square_size": int(size),
    }
    if not info["padded"]:
        return arr, info

    pad_width: list[tuple[int, int]] = [(top, bottom), (left, right)]
    pad_width.extend([(0, 0)] * (arr.ndim - 2))
    return np.pad(arr, pad_width, mode="constant", constant_values=fill), info


def content_mask_from_pad(info: dict) -> np.ndarray:
    """Binary ``(S, S)`` mask of the unpadded content inside a letterboxed square."""
    size = int(info["square_size"])
    height, width = info["content_shape"]
    mask = np.zeros((size, size), dtype=np.uint8)
    y0, x0 = int(info["top"]), int(info["left"])
    mask[y0 : y0 + int(height), x0 : x0 + int(width)] = 1
    return mask


def mirror_mask_y(mask: np.ndarray, cx: float) -> np.ndarray:
    """Reflect a 2D mask across the vertical line ``x = cx`` (left–right).

    Same construction as dinov2-radiomics: nonzero pixels are copied to
    ``2 * cx - x``, then clipped to the slice bounds.
    """
    mask_bin = np.asarray(mask) > 0
    coords = np.column_stack(np.nonzero(mask_bin))
    if coords.size == 0:
        return np.zeros_like(mask_bin, dtype=np.uint8)

    mirrored = coords.copy()
    mirrored[:, 1] = np.round(2.0 * float(cx) - coords[:, 1]).astype(int)
    valid = (
        (mirrored[:, 0] >= 0)
        & (mirrored[:, 0] < mask_bin.shape[0])
        & (mirrored[:, 1] >= 0)
        & (mirrored[:, 1] < mask_bin.shape[1])
    )
    out = np.zeros_like(mask_bin, dtype=np.uint8)
    out[mirrored[valid, 0], mirrored[valid, 1]] = 1
    return out


def process_mask(
    mask: np.ndarray,
    *,
    remove_holes: bool = False,
    hole_area_threshold: int = 512,
    mirror_y: bool = False,
) -> np.ndarray:
    """Fill small holes and/or symmetrize a ``(Z, Y, X)`` mask.

    Order matches dinov2-radiomics: per-slice ``remove_small_holes``, then
    union with the left–right reflection across the 3D mask centroid's x.
    A 2D mask is treated as a single-slice volume.
    """
    arr = np.asarray(mask)
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 2D or (Z, Y, X) mask, got shape {arr.shape}")

    processed = (arr > 0).astype(np.uint8)
    if remove_holes:
        if hole_area_threshold < 0:
            raise ValueError(f"hole_area_threshold must be >= 0, got {hole_area_threshold}")
        hole_kwargs: dict = {"connectivity": 2}
        hole_params = inspect.signature(remove_small_holes).parameters
        if "max_size" in hole_params:
            hole_kwargs["max_size"] = int(hole_area_threshold)
        else:
            hole_kwargs["area_threshold"] = int(hole_area_threshold)
        processed = np.stack(
            [
                remove_small_holes(slice_.astype(bool), **hole_kwargs).astype(np.uint8)
                for slice_ in processed
            ]
        )

    if mirror_y:
        coords = np.argwhere(processed > 0)
        if coords.size:
            cx = float(coords.mean(axis=0)[2])
            mirrored = np.stack([mirror_mask_y(slice_, cx) for slice_ in processed])
            processed = ((processed > 0) | (mirrored > 0)).astype(np.uint8)

    return processed[0] if squeeze else processed


def create_rgb_from_grayscale(image_array: np.ndarray) -> np.ndarray:
    """Normalize a grayscale slice to ``[0, 255]`` and stack it into RGB."""
    image_min = float(image_array.min())
    image_max = float(image_array.max())

    if image_max > image_min:
        normalized = ((image_array - image_min) / (image_max - image_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(image_array, dtype=np.uint8)

    return np.stack([normalized] * 3, axis=2)


def apply_intensity(image_array: np.ndarray, intensity: IntensityConfig | None) -> np.ndarray:
    """Optionally clip percentiles or apply a CT-style window before RGB conversion."""
    arr = np.asarray(image_array, dtype=np.float32)
    if intensity is None:
        return arr

    if intensity.window is not None:
        if len(intensity.window) != 2:
            raise ValueError("intensity.window must be [center, width]")
        center, width = map(float, intensity.window)
        low = center - width / 2.0
        high = center + width / 2.0
        return np.clip(arr, low, high)

    if intensity.clip_percentiles is not None:
        if len(intensity.clip_percentiles) != 2:
            raise ValueError("intensity.clip_percentiles must be [low, high]")
        low, high = np.percentile(arr, intensity.clip_percentiles)
        if high > low:
            return np.clip(arr, low, high)
    return arr


def _min_square_covering_mask(center_y: int, center_x: int, coords: np.ndarray) -> int:
    """Smallest square, centered on ``(center_y, center_x)``, that contains all mask coords."""
    ymin, xmin = int(coords[:, 1].min()), int(coords[:, 2].min())
    ymax, xmax = int(coords[:, 1].max()), int(coords[:, 2].max())
    half = max(center_y - ymin, ymax - center_y, center_x - xmin, xmax - center_x)
    return int(2 * half + 1)


def _crop_or_pad_xy(
    array: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    fill: float | int = 0,
) -> np.ndarray:
    """Take ``array[:, y0:y1, x0:x1]``, padding with ``fill`` where the window leaves the volume."""
    out_h, out_w = y1 - y0, x1 - x0
    out = np.full((array.shape[0], out_h, out_w), fill, dtype=array.dtype)

    src_y0 = max(y0, 0)
    src_y1 = min(y1, array.shape[1])
    src_x0 = max(x0, 0)
    src_x1 = min(x1, array.shape[2])
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out

    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    out[:, dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = array[
        :, src_y0:src_y1, src_x0:src_x1
    ]
    return out


def crop_centered_on_mask(
    array: np.ndarray,
    mask: np.ndarray,
    crop_size: int | None = None,
    min_mask_pixels: int = 1,
    extras: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], dict]:
    """Square in-plane crop centered on the mask centroid.

    Height and width are equal. The window stays centered on the centroid; voxels
    outside the image are zero-padded. If ``crop_size`` cannot hold the full
    mask, the square is expanded so every mask voxel is included.

    Depth keeps only slices whose cropped window contains at least
    ``min_mask_pixels`` mask voxels.

    Parameters
    ----------
    array, mask
        Volumes shaped ``(Z, Y, X)``. A 2D image is a single-slice volume.
    crop_size
        In-plane side length. If ``None``, uses the smallest centered square
        that covers the mask.
    extras
        Optional arrays cropped with the same window and slices (e.g. labels).

    Returns
    -------
    cropped, cropped_mask, cropped_extras, info
        ``info`` includes ``crop_region`` ``(y0, y1, x0, x1)`` in original
        coordinates (may be negative if padded), ``centroid``, ``crop_size``,
        ``padded``, and ``valid_slice_indices`` (original Z).
    """
    array = np.asarray(array)
    mask_arr = np.asarray(mask)
    mask_bin = mask_arr > 0
    if array.shape != mask_bin.shape:
        raise ValueError(f"Image and mask shapes differ: {array.shape} vs {mask_bin.shape}")
    if array.ndim != 3:
        raise ValueError(f"Expected a (Z, Y, X) volume, got shape {array.shape}")

    coords = np.argwhere(mask_bin)
    if coords.size == 0:
        raise ValueError("Cannot crop to mask: mask is empty")

    centroid = coords.mean(axis=0)
    cy = int(round(float(centroid[1])))
    cx = int(round(float(centroid[2])))
    _, height, width = array.shape

    needed = _min_square_covering_mask(cy, cx, coords)
    if crop_size is None:
        size = needed
    else:
        size = int(crop_size)
        if size < 1:
            raise ValueError(f"crop_size must be >= 1, got {size}")
        if needed > size:
            size = needed

    y0 = cy - size // 2
    x0 = cx - size // 2
    y1 = y0 + size
    x1 = x0 + size
    padded = y0 < 0 or x0 < 0 or y1 > height or x1 > width

    cropped_full = _crop_or_pad_xy(array, y0, y1, x0, x1, fill=0)
    cropped_mask_full = _crop_or_pad_xy(mask_arr, y0, y1, x0, x1, fill=0)
    in_plane = cropped_mask_full > 0
    valid = np.flatnonzero(in_plane.reshape(in_plane.shape[0], -1).sum(axis=1) >= min_mask_pixels)
    if valid.size == 0:
        raise ValueError("No slices found with sufficient mask pixels in the crop window")

    cropped = cropped_full[valid]
    cropped_mask = cropped_mask_full[valid]
    cropped_extras: list[np.ndarray] = []
    if extras:
        for extra in extras:
            extra_arr = np.asarray(extra)
            if extra_arr.shape != array.shape:
                raise ValueError(
                    f"Extra array shape {extra_arr.shape} does not match image {array.shape}"
                )
            cropped_extras.append(_crop_or_pad_xy(extra_arr, y0, y1, x0, x1, fill=0)[valid])

    info = {
        "crop_region": (int(y0), int(y1), int(x0), int(x1)),
        "centroid": (int(cy), int(cx)),
        "crop_size": int(size),
        "padded": bool(padded),
        "valid_slice_indices": [int(z) for z in valid.tolist()],
    }
    return cropped, cropped_mask, cropped_extras, info


def patch_grid_edges(length: int, grid_size: int) -> np.ndarray:
    """Pixel edges of a uniform ``grid_size``-cell partition of ``length``.

    This matches DINO's square resize: token ``i`` covers
    ``[i * length // G, (i + 1) * length // G)``. Remainder is spread across
    cells instead of dumped on the last row/column.
    """
    if grid_size < 1:
        raise ValueError(f"grid_size must be >= 1, got {grid_size}")
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}")
    return np.array([index * int(length) // grid_size for index in range(grid_size + 1)], dtype=int)


def apply_mask_to_patch_embeddings(
    patch_emb: np.ndarray | torch.Tensor,
    mask: np.ndarray,
    mask_label: np.ndarray | None = None,
    mask_channel: int = 0,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Zero patch embeddings that do not overlap the ROI mask.

    Occupancy is tested on the same uniform G×G tiling DINO uses after a
    square resize. A token is zeroed when its own cell has no mask voxels.

    Returns
    -------
    patch_emb
        Embeddings with background patches set to zero. Shape ``[num_patches, dim]``.
    patch_labels
        Binary labels from ``mask_label`` when provided, else ``None``.
    patch_mask
        Binary occupancy of each patch against ``mask``.
    """
    if hasattr(patch_emb, "detach"):
        patch_emb_np = patch_emb.detach().cpu().numpy()
    else:
        patch_emb_np = np.array(patch_emb, copy=True)

    num_patches = patch_emb_np.shape[0]
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Patch grid is not square (got {num_patches} patches)")

    if mask_label is not None and mask_label.shape != mask.shape:
        raise ValueError("Mask label shape does not match mask shape")

    if mask.ndim == 3:
        mask_bin = (mask[:, :, mask_channel] > 0).astype(np.uint8)
        mask_label_bin = (
            (mask_label[:, :, mask_channel] > 0).astype(np.uint8) if mask_label is not None else None
        )
    else:
        mask_bin = (mask > 0).astype(np.uint8)
        mask_label_bin = (mask_label > 0).astype(np.uint8) if mask_label is not None else None

    height, width = mask_bin.shape
    y_edges = patch_grid_edges(height, grid_size)
    x_edges = patch_grid_edges(width, grid_size)

    patch_labels = np.zeros(num_patches, dtype=np.uint8) if mask_label is not None else None
    patch_mask = np.zeros(num_patches, dtype=np.uint8)

    for i in range(grid_size):
        y0, y1 = int(y_edges[i]), int(y_edges[i + 1])
        for j in range(grid_size):
            patch_idx = i * grid_size + j
            x0, x1 = int(x_edges[j]), int(x_edges[j + 1])
            if y1 <= y0 or x1 <= x0 or np.sum(mask_bin[y0:y1, x0:x1]) < 1:
                patch_emb_np[patch_idx] = 0
            else:
                patch_mask[patch_idx] = 1

            if (
                mask_label_bin is not None
                and y1 > y0
                and x1 > x0
                and np.sum(mask_label_bin[y0:y1, x0:x1]) >= 1
            ):
                patch_labels[patch_idx] = 1

    return patch_emb_np, patch_labels, patch_mask


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> torch.Tensor:
    """Reverse ImageNet normalization for visualization."""
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    mean_t = torch.as_tensor(mean, device=tensor.device, dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.as_tensor(std, device=tensor.device, dtype=tensor.dtype).view(3, 1, 1)
    return torch.clamp(tensor * std_t + mean_t, 0, 1)
