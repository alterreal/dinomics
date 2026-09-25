"""Case-level pooling of slice (CLS) and patch embeddings."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

Reducer = Callable[..., object]
Reducers = Reducer | Sequence[Reducer]


def _as_reducer_list(reducers: Reducers) -> list[Reducer]:
    if callable(reducers) and not isinstance(reducers, type):
        return [reducers]
    if isinstance(reducers, Sequence) and not isinstance(reducers, (str, bytes)):
        fns = list(reducers)
        if not fns:
            raise ValueError("reducers must be a function or a non-empty list of functions")
        if not all(callable(fn) for fn in fns):
            raise TypeError("every item in reducers must be callable")
        return fns
    raise TypeError("reducers must be a NumPy-style function or a list of functions")


def _apply_reducer(fn: Reducer, tokens: np.ndarray) -> np.ndarray:
    """Reduce ``(P, D)`` tokens along the token axis to a 1-D vector."""
    try:
        out = fn(tokens, axis=0)
    except TypeError:
        out = fn(tokens)
    vec = np.asarray(out, dtype=np.float64)
    if vec.ndim == 0:
        return vec.reshape(1)
    return np.reshape(vec, -1)


def apply_reducers(tokens: np.ndarray, reducers: Reducers) -> np.ndarray:
    """Apply one reducer or concatenate several over the token/slice axis.

    Each function is called as ``fn(tokens, axis=0)`` (``tokens`` is ``(P, D)``).
    Typical choices are ``np.mean``, ``np.max``, ``np.std``, ``np.median``.
    """
    tokens = np.asarray(tokens)
    if tokens.ndim != 2 or tokens.shape[0] == 0:
        raise ValueError(f"Expected a non-empty (P, D) array, got shape {tokens.shape}")
    parts = [_apply_reducer(fn, tokens) for fn in _as_reducer_list(reducers)]
    return np.concatenate(parts, axis=0)


def _select_patch_tokens(
    patch_embeddings: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    arr = np.asarray(patch_embeddings)
    if arr.ndim == 2:
        tokens = arr
        if mask is not None:
            flag = np.asarray(mask).astype(bool).reshape(-1)
            if flag.size != tokens.shape[0]:
                raise ValueError(
                    f"mask size {flag.size} does not match {tokens.shape[0]} patches"
                )
            tokens = tokens[flag]
        return tokens

    if arr.ndim == 4:
        if mask is not None:
            flag = np.asarray(mask).astype(bool)
            if flag.shape != arr.shape[:3]:
                raise ValueError(
                    f"mask shape {flag.shape} does not match patches {arr.shape[:3]}"
                )
            return arr[flag]
        flat = arr.reshape(-1, arr.shape[-1])
        valid = ~np.all(np.isclose(flat, 0.0), axis=-1)
        return flat[valid] if np.any(valid) else flat

    if arr.ndim == 3:
        flat = arr.reshape(-1, arr.shape[-1])
        if mask is not None:
            flag = np.asarray(mask).astype(bool).reshape(-1)
            if flag.size != flat.shape[0]:
                raise ValueError(
                    f"mask size {flag.size} does not match {flat.shape[0]} patches"
                )
            return flat[flag]
        valid = ~np.all(np.isclose(flat, 0.0), axis=-1)
        return flat[valid] if np.any(valid) else flat

    raise ValueError(
        f"Expected patch embeddings (N, G, G, D), (G, G, D), or (P, D), got {arr.shape}"
    )


def aggregate_patches(
    patch_embeddings: np.ndarray,
    reducers: Reducers,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Pool patch tokens over the case (all slices, in-ROI only when ``mask`` is set).

    Parameters
    ----------
    patch_embeddings
        ``(N, G, G, D)`` spatial patches, ``(G, G, D)``, or flat ``(P, D)``.
    reducers
        ``np.mean`` or ``[np.mean, np.max, ...]``. Each call uses ``axis=0``.
        A list is concatenated into one vector.
    mask
        Optional occupancy ``(N, G, G)`` (or broadcastable). Background patches
        are dropped. If omitted, all-zero tokens are dropped when the array is
        spatial.
    """
    tokens = _select_patch_tokens(patch_embeddings, mask)
    if tokens.size == 0:
        raise ValueError("No patch tokens left to aggregate (mask may be empty)")
    return apply_reducers(tokens, reducers)


def aggregate_slices(cls_embeddings: np.ndarray, reducers: Reducers) -> np.ndarray:
    """Pool CLS / slice embeddings over the case.

    Parameters
    ----------
    cls_embeddings
        ``(N, D)`` one vector per processed slice.
    reducers
        ``np.mean`` or ``[np.mean, np.max, ...]``. A list is concatenated.
    """
    arr = np.asarray(cls_embeddings)
    if arr.ndim != 2:
        raise ValueError(f"Expected CLS embeddings (N, D), got shape {arr.shape}")
    return apply_reducers(arr, reducers)
