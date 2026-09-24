"""Generic medical image I/O. Accepts paths, SimpleITK images, or numpy arrays."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

ImageLike = str | Path | sitk.Image | np.ndarray


@dataclass
class Volume:
    """In-memory volume used by the extractor.

    ``array`` is always ``(Z, Y, X)``. A 2D input is stored as a single-slice volume.
    """

    array: np.ndarray
    image: sitk.Image | None = None
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0)
    origin: tuple[float, ...] = (0.0, 0.0, 0.0)
    direction: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    path: str | None = None
    is_2d: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)


def _as_sitk(source: ImageLike) -> sitk.Image:
    if isinstance(source, sitk.Image):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return sitk.ReadImage(str(path))
    raise TypeError(f"Cannot build a SimpleITK image from {type(source)}")


def load_volume(source: ImageLike) -> Volume:
    """Load a 2D or 3D medical image as a :class:`Volume`."""
    if isinstance(source, np.ndarray):
        array = np.asarray(source)
        if array.ndim == 2:
            return Volume(array=array[np.newaxis, ...], is_2d=True)
        if array.ndim == 3:
            return Volume(array=array, is_2d=False)
        raise ValueError(f"Expected a 2D or 3D array, got shape {array.shape}")

    path = str(source) if isinstance(source, (str, Path)) else None
    image = _as_sitk(source)
    array = sitk.GetArrayFromImage(image)
    is_2d = image.GetDimension() == 2 or array.ndim == 2
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    elif array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image, got array shape {array.shape}")

    return Volume(
        array=array,
        image=image,
        spacing=image.GetSpacing(),
        origin=image.GetOrigin(),
        direction=image.GetDirection(),
        path=path,
        is_2d=is_2d,
    )


def resample_to_reference(moving: Volume, reference: Volume, interpolator: int | None = None) -> Volume:
    """Resample ``moving`` onto the grid of ``reference`` (nearest neighbor by default)."""
    if reference.image is None:
        if moving.array.shape != reference.array.shape:
            raise ValueError(
                "Cannot resample without SimpleITK metadata when array shapes differ: "
                f"{moving.array.shape} vs {reference.array.shape}"
            )
        return moving

    if moving.image is None:
        moving_img = sitk.GetImageFromArray(moving.array)
        moving_img.SetSpacing(moving.spacing)
        moving_img.SetOrigin(moving.origin)
        moving_img.SetDirection(moving.direction)
    else:
        moving_img = moving.image

    if interpolator is None:
        interpolator = sitk.sitkNearestNeighbor

    resampled = sitk.Resample(
        moving_img,
        reference.image,
        sitk.Transform(),
        interpolator,
        0,
        moving_img.GetPixelID(),
    )
    array = sitk.GetArrayFromImage(resampled)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    return Volume(
        array=array,
        image=resampled,
        spacing=resampled.GetSpacing(),
        origin=resampled.GetOrigin(),
        direction=resampled.GetDirection(),
        path=moving.path,
        is_2d=reference.is_2d,
    )


def _same_grid(mask: Volume, reference: Volume) -> bool:
    if mask.array.shape != reference.array.shape:
        return False
    if mask.image is None or reference.image is None:
        return True
    return (
        mask.image.GetSize() == reference.image.GetSize()
        and np.allclose(mask.image.GetSpacing(), reference.image.GetSpacing())
        and np.allclose(mask.image.GetOrigin(), reference.image.GetOrigin())
    )


def align_mask(mask: Volume, reference: Volume) -> Volume:
    """Ensure a mask lives on the same voxel grid as the image."""
    if _same_grid(mask, reference):
        return mask
    return resample_to_reference(mask, reference, interpolator=sitk.sitkNearestNeighbor)
