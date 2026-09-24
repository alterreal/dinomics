"""PCA visualization of DINO patch embeddings (as in dinov2_feature_extraction.ipynb)."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from dinomics.preprocess import patch_grid_edges

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _as_spatial_patches(patch_embeddings: np.ndarray) -> np.ndarray:
    arr = np.asarray(patch_embeddings)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:
        grid = int(np.sqrt(arr.shape[0]))
        if grid * grid != arr.shape[0]:
            raise ValueError(f"Cannot reshape {arr.shape[0]} patches into a square grid")
        return arr.reshape(grid, grid, arr.shape[1])
    raise ValueError(f"Expected (G, G, D) or (P, D), got shape {arr.shape}")


def non_zero_patch_mask(patch_embeddings: np.ndarray) -> np.ndarray:
    """Boolean ``(G, G)`` mask of patches that are not the zero vector."""
    spatial = _as_spatial_patches(patch_embeddings)
    return ~np.all(np.isclose(spatial, 0.0), axis=-1)


def pca_patch_map(
    patch_embeddings: np.ndarray,
    whiten: bool = True,
    sigmoid_scale: float = 2.0,
    hide_zero_patches: bool = True,
) -> np.ndarray:
    """Reduce patch embeddings to a 3-channel RGB map.

    This follows the notebook technique: PCA to 3 components (optionally whitened),
    reshape to the square patch grid, then ``sigmoid(scale * pca)``.

    Parameters
    ----------
    patch_embeddings
        ``(G, G, D)`` spatial patches or ``(P, D)`` flat patches.
    whiten
        Passed to ``sklearn.decomposition.PCA``.
    sigmoid_scale
        Multiplier applied before the sigmoid. ``2.0`` matches the original notebook.
    hide_zero_patches
        Fit PCA on non-zero patches only. Zero patches are left as NaN so they
        can be drawn as transparent.

    Returns
    -------
    np.ndarray
        RGB heatmap in ``[0, 1]`` with shape ``(G, G, 3)``. Zero patches are
        NaN when ``hide_zero_patches`` is True.
    """
    spatial = _as_spatial_patches(patch_embeddings)
    grid_h, grid_w, dim = spatial.shape
    flat = spatial.reshape(grid_h * grid_w, dim)
    valid = non_zero_patch_mask(spatial).reshape(-1)

    reduced = np.full((flat.shape[0], 3), np.nan, dtype=np.float64)
    if hide_zero_patches:
        if not np.any(valid):
            return reduced.reshape(grid_h, grid_w, 3)
        pca = PCA(n_components=3, whiten=whiten)
        reduced[valid] = pca.fit_transform(flat[valid])
    else:
        pca = PCA(n_components=3, whiten=whiten)
        reduced = pca.fit_transform(flat)

    heatmap = reduced.reshape(grid_h, grid_w, 3)
    finite = np.isfinite(heatmap).all(axis=-1, keepdims=True)
    if sigmoid_scale is not None:
        colored = 1.0 / (1.0 + np.exp(-sigmoid_scale * np.nan_to_num(heatmap, nan=0.0)))
    else:
        filled = np.nan_to_num(heatmap, nan=0.0)
        low = filled.min(axis=(0, 1), keepdims=True)
        high = filled.max(axis=(0, 1), keepdims=True)
        colored = (filled - low) / (high - low + 1e-8)
    colored = np.clip(colored, 0.0, 1.0)
    return np.where(finite, colored, np.nan)


def resize_feature_map(heatmap: np.ndarray, size: tuple[int, int], resample=None) -> np.ndarray:
    """Paint a ``(G, G, ·)`` map onto ``(H, W)`` with DINO's uniform patch tiling."""
    del resample
    heatmap = np.asarray(heatmap)
    out_h, out_w = int(size[0]), int(size[1])
    grid_h, grid_w = heatmap.shape[:2]
    y_edges = patch_grid_edges(out_h, grid_h)
    x_edges = patch_grid_edges(out_w, grid_w)
    if heatmap.ndim == 2:
        out = np.zeros((out_h, out_w), dtype=np.float32)
    else:
        out = np.zeros((out_h, out_w) + heatmap.shape[2:], dtype=np.float32)
    for i in range(grid_h):
        y0, y1 = int(y_edges[i]), int(y_edges[i + 1])
        if y1 <= y0:
            continue
        for j in range(grid_w):
            x0, x1 = int(x_edges[j]), int(x_edges[j + 1])
            if x1 <= x0:
                continue
            out[y0:y1, x0:x1] = heatmap[i, j]
    return out


def _gray_cmap():
    cmap = plt.cm.gray.copy()
    cmap.set_bad((0, 0, 0, 0))
    return cmap


def _rgba(rgb: np.ndarray, visible: np.ndarray) -> np.ndarray:
    rgb = np.nan_to_num(np.asarray(rgb, dtype=np.float32), nan=0.0)
    alpha = np.asarray(visible, dtype=np.float32)
    if alpha.ndim == 3:
        alpha = alpha[..., 0]
    rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.float32)
    rgba[..., :3] = np.clip(rgb, 0.0, 1.0)
    rgba[..., 3] = alpha
    return rgba


def plot_pca_features(
    image_slice: np.ndarray,
    patch_embeddings: np.ndarray,
    mask_slice: np.ndarray | None = None,
    label_slice: np.ndarray | None = None,
    original_slice: np.ndarray | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (14, 4),
    ax=None,
):
    """Four-panel PCA figure: original (pre-crop), preprocessed, PCA, overlay.

    Zero pixels and zero patch embeddings are drawn as transparent in the
    preprocessed, PCA, and overlay panels.

    The overlay upsamples the PCA map with the same uniform patch tiling used
    to zero background tokens. When ``label_slice`` is provided, red contours
    of that label are drawn.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plot_pca_features")

    cropped = np.asarray(image_slice, dtype=np.float32)
    original = cropped if original_slice is None else np.asarray(original_slice, dtype=np.float32)
    if mask_slice is not None:
        mask_bin = np.asarray(mask_slice) > 0
        preprocessed = cropped * mask_bin
    else:
        mask_bin = None
        preprocessed = cropped

    pixel_visible = preprocessed != 0
    if mask_bin is not None:
        pixel_visible &= mask_bin

    heatmap = pca_patch_map(patch_embeddings, hide_zero_patches=True)
    patch_visible = non_zero_patch_mask(patch_embeddings)
    heatmap_rgb = np.nan_to_num(heatmap, nan=0.0)
    heatmap_resized = resize_feature_map(heatmap_rgb, preprocessed.shape[:2])
    patch_visible_resized = resize_feature_map(patch_visible.astype(np.float32), preprocessed.shape[:2]) > 0.5

    created_fig = ax is None
    if created_fig:
        fig, axes = plt.subplots(1, 4, figsize=figsize)
    else:
        fig = None
        axes = ax

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    for axis in axes[1:]:
        axis.set_facecolor("white")

    axes[1].imshow(np.ma.masked_where(~pixel_visible, preprocessed), cmap=_gray_cmap())
    axes[1].set_title("Preprocessed")
    axes[1].axis("off")

    axes[2].imshow(_rgba(heatmap_rgb, patch_visible))
    axes[2].set_title("PCA visualization")
    axes[2].axis("off")

    axes[3].imshow(_rgba(heatmap_resized, patch_visible_resized))
    axes[3].imshow(np.ma.masked_where(~pixel_visible, preprocessed), cmap=_gray_cmap(), alpha=0.5)
    if label_slice is not None:
        from skimage import measure

        contours = measure.find_contours((np.asarray(label_slice) > 0).astype(np.uint8), 0.5)
        for contour in contours:
            axes[3].plot(contour[:, 1], contour[:, 0], color="red", linewidth=1)
    axes[3].set_title("Overlay: image + PCA")
    axes[3].axis("off")

    if title:
        if fig is not None:
            fig.suptitle(title)
        else:
            axes[0].set_title(title)

    if created_fig:
        fig.tight_layout()
    return fig, axes, heatmap
