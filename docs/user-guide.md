# dinomics user guide

This guide is for a first run: install the package, extract features from a medical image, understand the output, and change the settings that matter.

dinomics wraps HuggingFace **DINOv2** and **DINOv3** so you can pull **slice-level** and **patch-level** embeddings from 2D images or 3D volumes. Volumes are processed **slice by slice**. Radiomics and anatomy-specific I/O are out of scope.

---

## What you get

For each processed slice the model returns three descriptors:

| Name | Shape | What it is |
|---|---|---|
| `cls_embeddings` | `(N, D)` | One vector per slice (the CLS token). Use this as a slice-level fingerprint. |
| `patch_embeddings` | `(N, G, G, D)` | A square grid of patch tokens. Use this when you need spatial features. |
| `patch_embeddings_maxpooled` | `(N, D)` | The max over all patches on that slice. A second slice-level vector. |

`N` is the number of slices that were actually encoded. `G` is the patch grid (for DINOv2-base, `G = image_size / 14`, so `224 → 16` and `448 → 32`). `D` is `768` for the default ViT-B checkpoints.

If you pass a mask, you also get `patch_mask` (`1` = that patch overlaps the ROI). Background patch tokens can be set to the **zero vector**.

---

## 1. Install

You need [uv](https://docs.astral.sh/uv/), Python **3.11–3.13** (the repo pins **3.12**), and an internet connection the first time (PyTorch wheels + the HuggingFace checkpoint).

Clone the repo and pick a **PyTorch extra** that matches your GPU driver. The wheel CUDA version must be **less than or equal to** the driver CUDA version shown by `nvidia-smi`. No NVIDIA GPU, or macOS: use `cpu`.

| Extra | PyTorch wheels | Use when |
|---|---|---|
| `cpu` | CPU-only | no NVIDIA GPU |
| `cu118` | CUDA 11.8 | driver 11.8+ |
| `cu121` | CUDA 12.1 | driver 12.1+ |
| `cu124` | CUDA 12.4 | driver 12.4+ |
| `cu126` | CUDA 12.6 | driver 12.6+ |

```bash
nvidia-smi          # look at "CUDA Version" in the header
cd dinomics
uv sync --extra cu121
```

Examples: driver **12.2** → `cu121`. Driver **12.8** → `cu126` (highest extra that still fits). **Do not** enable two CUDA extras at once.

For the example notebook, add Jupyter:

```bash
uv sync --extra cu121 --extra notebook
```

`uv sync` with **no** extra does not install PyTorch. Always pass one of `cpu` / `cu118` / `cu121` / `cu124` / `cu126`.

Activate the environment, or prefix commands with `uv run`:

```bash
source .venv/bin/activate
# or
uv run python your_script.py
```

The first extraction downloads the model weights from HuggingFace (a few hundred MB) and caches them under `~/.cache/huggingface`.

---

## 2. Run something immediately

Bundled examples (if present in your checkout):

- MRI T2: `example_data/MRI/10000_1000000_t2w.mha` + mask `example_data/MRI/10000_1000000.nii.gz`
- CT: `example_data/CT/volume-10.nii` + labels `example_data/CT/labels-10.nii`

### Python, one image

```python
from dinomics import extract_features

result = extract_features(
    "example_data/MRI/10000_1000000_t2w.mha",
    config="configs/dinov2.yaml",
    mask="example_data/MRI/10000_1000000.nii.gz",
)

print(result.cls_embeddings.shape)      # (N, 768)
print(result.patch_embeddings.shape)    # (N, G, G, 768)
print(result.slice_indices)             # original z of each encoded slice
```

`extract_features` loads the model, runs one image, and returns. Fine for a trial. For a cohort, load the model **once** (next section).

### Python, many images

```python
from dinomics import DinoExtractor, load_config

config = load_config("configs/dinov2.yaml")
extractor = DinoExtractor.from_config(config)

for path, mask in cases:
    result = extractor.extract(path, mask=mask)
    result.save(f"outputs/{Path(path).stem}.npz")
```

### Command line

```bash
uv run dinomics \
    --image example_data/MRI/10000_1000000_t2w.mha \
    --mask example_data/MRI/10000_1000000.nii.gz \
    --config configs/dinov2.yaml \
    --output outputs/mri_dino.npz
```

`--output` defaults to `<image_stem>_dino_features.npz` next to the image.

### Notebook

`notebooks/feature_extraction.ipynb` extracts MRI and CT and draws a four-panel PCA figure.

1. `uv sync --extra cu121 --extra notebook`
2. Select the **dinomics** kernel (the interpreter at `.venv/bin/python`).
3. Run all cells. The first DINO call is slow (weights + compile); later slices are faster.

---

## 3. What to pass in

### Images

Anything SimpleITK can read: `.mha`, `.nii`, `.nii.gz`, and similar. You can also pass a SimpleITK `Image` or a NumPy array.

- **2D** → treated as a volume with one slice, shape `(1, H, W)`.
- **3D** → array layout is always `(Z, Y, X)` after load (SimpleITK’s convention).
- Arrays must be 2D or 3D grayscale. There is no RGB medical-image path; the package builds RGB itself.

### Masks (optional)

A mask is a second image in the **same patient space**. Values `> 0` are the ROI. If the voxel grid does not match the image, dinomics resamples the mask onto the image with **nearest neighbor**.

Without a mask:

- The **full field of view** is encoded.
- Each slice is **letterboxed** to a square (centered, zero-filled) *after* intensity/RGB and *before* the model resize, so a non-square image is not stretched. Already-square slices are left as-is.
- `apply_image_mask` does nothing. If `apply_patch_mask` is on and padding was added, letterbox patches are zeroed and `patch_mask` marks the content cells; otherwise `patch_mask` is `None`.
- `skip_empty_slices` drops slices that are all zeros (not “empty mask”).
- `metadata["square_pad"]` stores the pad amounts (`top`, `bottom`, `left`, `right`).

**`crop_to_mask: true` requires a mask.** The starter file `configs/dinov2.yaml` has this on. Use `configs/default.yaml` or set `crop_to_mask: false` if you have no ROI.

### Label maps (optional)

`--label-mask` / `label_mask=` is a second mask used only to tag patches (`patch_labels`). It does not replace the ROI mask.

---

## 4. What happens to each slice

This is the full pipeline, in order. Every knob lives in the config file.

1. **Load** the image (and mask / label map). Align the mask to the image grid.
2. **Optional mask cleanup** (ROI mask only, not `label_mask`): `remove_small_holes` fills per-slice holes up to `hole_area_threshold` (default 512). `mirror_mask_y` unions the mask with its left–right reflection across the 3D centroid's x (same as dinov2-radiomics). Both run **before** the crop.
3. **Optional crop** (`crop_to_mask`): square window in-plane, centered on the mask **centroid**. Height equals width. `crop_size: null` uses the smallest square that covers the mask. If a requested size is too small, the square **grows**. Voxels outside the volume are **zero-padded**. Depth keeps only slices whose crop contains at least `min_mask_pixels` mask voxels.
4. **Optional image mask** (`apply_image_mask`): multiply the (cropped) volume by the mask so background voxels are zero **before** the network.
5. **Pick slices**: `slice_indices`, or every `slice_step`-th slice. `skip_empty_slices` drops empty ones. `max_slices` truncates the list.
6. **Intensity**: optional CT window `[center, width]` or percentile clip, then per-slice **min–max** to `0–255` and stack to RGB.
7. **Square pad (no mask only)**: if the slice is not square, center-pad with black so height equals width. This keeps aspect ratio when the processor later resizes to `image_size`.
8. **HuggingFace processor**: square resize to `image_size × image_size`, ImageNet normalize. `do_center_crop` is off in the shipped configs so the whole (cropped or letterboxed) slice is kept.
9. **Forward pass**: CLS token + patch tokens. DINOv3 also drops **register tokens** (4 by default) before the patch grid.
10. **Optional patch mask** (`apply_patch_mask`): each DINO token is tested on the **same uniform G×G tiling** the model uses after the square resize. If that cell has **no** mask voxels, the embedding is set to the zero vector.

Background that DINO still “sees” via attention (black padded pixels, nearby anatomy) can influence kept tokens. Zeroing is applied **after** the forward pass.

---

## 5. Configuration

Copy a file from `configs/` and pass its path into `extract_features`, `DinoExtractor`, or `--config`. YAML and TOML both work. You can also pass a dict or edit the dataclass after `load_config`.

| File | Intent |
|---|---|
| `configs/default.yaml` | DINOv2-base, no crop required |
| `configs/dinov2.yaml` | DINOv2-base, crop to mask (`crop_size: 128`) |
| `configs/dinov3.yaml` | DINOv3 ViT-B/16 |
| `configs/ct.yaml` | DINOv2, soft-tissue window `[40, 400]`, crop to mask, `image_size: 448` |
| `configs/dinov2.toml` | Same idea as default, TOML syntax |

### `model`

| Key | Default | Meaning |
|---|---|---|
| `name` | `facebook/dinov2-base` | HuggingFace checkpoint |
| `feature_type` | `dinov2` | `dinov2` or `dinov3` (controls register-token handling) |
| `image_size` | `224` | Square resize. Larger → finer `G`, more VRAM |
| `device` | `auto` | `auto` (CUDA if available), `cuda`, or `cpu` |
| `n_register_tokens` | `0` / `4` | Auto: 0 for DINOv2, 4 for DINOv3. Tokens skipped after CLS |
| `do_center_crop` | `false` | Processor center-crop after resize. Leave false unless you want it |

### `intensity`

| Key | Default | Meaning |
|---|---|---|
| `window` | `null` | CT window `[center, width]`, e.g. `[40, 400]` |
| `clip_percentiles` | `null` | Clip to those percentiles, e.g. `[0.5, 99.5]` |

If both are set, **window wins**. After this, each slice is still min–max scaled to 8-bit RGB.

### `extraction`

| Key | Default | Meaning |
|---|---|---|
| `apply_image_mask` | `true` | Zero voxels outside the mask before encoding |
| `apply_patch_mask` | `true` | Zero patch tokens whose cell misses the mask |
| `skip_empty_slices` | `true` | Skip slices with no mask (or all-zero image if no mask) |
| `slice_axis` | `0` | Axis treated as slice (`0` = Z after SimpleITK load) |
| `slice_indices` | `null` | Explicit original-z list, e.g. `[10, 11, 12]` |
| `slice_step` | `1` | Keep every n-th slice when `slice_indices` is null |
| `max_slices` | `null` | Cap how many slices are encoded |
| `crop_to_mask` | `false` | Square crop around the mask centroid |
| `crop_size` | `null` | In-plane side length. `null` = smallest covering square |
| `min_mask_pixels` | `1` | Minimum mask voxels in a slice (or in the crop) to keep it |
| `remove_small_holes` | `false` | Fill small holes in the ROI mask (per slice) |
| `hole_area_threshold` | `512` | Largest hole area (pixels) that `remove_small_holes` fills |
| `mirror_mask_y` | `false` | Union the mask with its flip across the 3D centroid x |

### `output`

| Key | Default | Meaning |
|---|---|---|
| `include_flat_patches` | `false` | Also store `patch_embeddings_flat` as `(N, G*G, D)` |

After `load_config`, fields are normal Python attributes:

```python
config = load_config("configs/default.yaml")
config.model.image_size = 448
config.extraction.crop_to_mask = True
config.intensity.window = [40, 400]
```

---

## 6. Output

`FeatureResult` fields:

| Attribute | Shape | Present |
|---|---|---|
| `cls_embeddings` | `(N, D)` | always |
| `patch_embeddings` | `(N, G, G, D)` | always |
| `patch_embeddings_maxpooled` | `(N, D)` | always |
| `slice_indices` | `(N,)` | always — original z, after crop filtering |
| `patch_mask` | `(N, G, G)` | if a mask was applied to patches |
| `patch_labels` | `(N, G, G)` | if `label_mask` was given |
| `patch_embeddings_flat` | `(N, G*G, D)` | if `include_flat_patches` |
| `metadata` | dict | image path, model, crop window, spacing, full config |

Save / load:

```python
result.save("outputs/case.npz")
from dinomics import FeatureResult
result = FeatureResult.load("outputs/case.npz")
```

The `.npz` also stores `global_embeddings` as an alias of `cls_embeddings`. `metadata` is pickled inside the archive (`allow_pickle=True` on load).

---

## 7. Visualize patch tokens

`plot_pca_features` builds a four-panel figure: **Original** (pre-crop if you pass `original_slice`), **Preprocessed**, **PCA**, **Overlay**.

PCA is fit **per slice** on non-zero patches, reduced to 3 components (whitened), passed through a sigmoid, and painted with the same uniform patch grid used for masking. Zero tokens stay transparent.

```python
from dinomics import plot_pca_features
from dinomics.io import load_volume, align_mask
from dinomics.preprocess import crop_centered_on_mask

fig, axes, heatmap = plot_pca_features(
    cropped_slice,
    result.patch_embeddings[i],
    original_slice=full_volume[z],
    mask_slice=cropped_mask[i],
    title=f"slice {z}",
)
```

The notebook does this for the slice with the most mask voxels.

---

## 8. DINOv2 vs DINOv3

| | DINOv2 | DINOv3 |
|---|---|---|
| Example config | `configs/dinov2.yaml` | `configs/dinov3.yaml` |
| Default checkpoint | `facebook/dinov2-base` | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| `feature_type` | `dinov2` | `dinov3` |
| Register tokens dropped | 0 | 4 |
| Typical `image_size` | 224 (patch 14 → `G=16`) | 256 (patch 16 → `G=16`) |

Set `feature_type` to match the architecture. A wrong value shifts the patch grid (CLS / registers counted as patches).

---

## 9. Suggested setups

**Organ-masked MRI (like the example):** `configs/dinov2.yaml` — crop to mask, apply both masks, default 224 input.

**CT with a soft-tissue window:** `configs/ct.yaml` — `window: [40, 400]`, crop to mask, `image_size: 448` (32×32 patches).

**Whole slice, no ROI:** `configs/default.yaml` and **do not** pass a mask. Leave `crop_to_mask: false`. Non-square slices are letterboxed to a square first.

**Finer patches:** raise `image_size` (must stay compatible with the ViT patch size: multiples of 14 for DINOv2-base, 16 for DINOv3 ViT-B/16). More tokens, more memory.

**Fewer slices:** `slice_step: 2` or `max_slices: 8` while you test.

---

## 10. Common problems

**`crop_to_mask=True requires a mask`**  
`configs/dinov2.yaml` and `configs/ct.yaml` crop by default. Pass a mask, or set `crop_to_mask: false`.

**`No slices were processed`**  
The mask never overlaps the selected slices, or every slice was empty. Check `slice_axis`, `slice_indices`, and that the mask is in the same space as the image.

**CUDA / driver mismatch**  
`nvidia-smi` shows the **driver**. Install an extra ≤ that number. This machine’s older driver needed `cu121`, not a newer CUDA 13 wheel.

**`uv sync` then `import torch` fails**  
You did not pass a CUDA/CPU extra.

**Notebook kernel missing**  
Use `.venv/bin/python`. After `uv sync --extra notebook`, register it if needed:  
`uv run python -m ipykernel install --user --name dinomics --display-name dinomics`.

**Patches look “outside” the organ in PCA**  
Tokens are **squares**. A cell is kept if it overlaps the mask at all. Edge cells can sit mostly outside the organ and still be non-zero. Cells with **no** overlap are zero vectors and stay blank in the figure.

**First slice is slow, later ones are fine**  
Weight download and CUDA warmup. Reuse `DinoExtractor` across images.

**Mask does not match the image**  
Different size, spacing, or origin → automatic nearest-neighbor resample. Prefer a mask already on the image grid.

**Want CPU only for a test**  
`uv sync --extra cpu` and `config.model.device = "cpu"`.

---

## 11. Public Python API

```python
from dinomics import (
    DinoExtractor,       # load once, extract many
    extract_features,    # one-shot
    load_config,         # YAML / TOML / dict
    DinomicsConfig,
    FeatureResult,       # .save() / .load()
    crop_centered_on_mask,
    pca_patch_map,
    plot_pca_features,
)
```

Images and masks accept a path, a SimpleITK image, or a NumPy array.

---

## 12. Minimal checklist

1. `uv sync --extra <cpu|cu118|cu121|cu124|cu126>`
2. Copy a config from `configs/` and edit `crop_to_mask`, `image_size`, and intensity.
3. Call `extract_features(image, config=..., mask=...)` or the CLI.
4. Read `result.cls_embeddings` and `result.patch_embeddings`.
5. Optional: `plot_pca_features` or the example notebook to confirm the ROI and patch grid.
