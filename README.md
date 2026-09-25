# DINOmics 🦖🩻

A medical imaging pipeline that wraps Hugging Face DINOv2/DINOv3 and handles I/O, preprocessing, masking, and aggregation, enabling straightforward extraction of CLS (slice-level) and patch embeddings from 3D volumes and 2D images.

For each processed slice the DINOmics returns:

- **CLS** embeddings — one vector per slice (`cls_embeddings`)
- **Patch** embeddings — a spatial grid of patch tokens (`patch_embeddings`)
- **Max-pooled patches** — a second slice-level descriptor (`patch_embeddings_maxpooled`)

Any grayscale medical image that SimpleITK can read (`.mha`, `.nii`, `.nii.gz`, …) works.

**First time here?** Read the [user guide](docs/user-guide.md)

## Install

DINOmics is managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`). It targets Python 3.11–3.13 (`.python-version` pins 3.12).

PyTorch is installed through a **conflicting extra**. Pick the wheel whose CUDA version is **≤ your driver** (`nvidia-smi`). macOS and machines without an NVIDIA GPU should use `cpu`.

| Extra | PyTorch wheels | Use when |
|---|---|---|
| `cpu` | CPU-only | no NVIDIA GPU |
| `cu118` | CUDA 11.8 | driver 11.8+ |
| `cu121` | CUDA 12.1 | driver 12.1+ |
| `cu124` | CUDA 12.4 | driver 12.4+ |
| `cu126` | CUDA 12.6 | driver 12.6+ |

```bash
nvidia-smi   # "CUDA Version" is the driver; pick an extra at or below that
uv sync --extra cu121
uv sync --extra cu121 --extra notebook   # optional: Jupyter for the example notebook
```

Examples: driver 12.2 → `cu121`; driver 12.8 → `cu126` (highest extra that still fits). Do not combine two CUDA extras.

Activate the environment with `source .venv/bin/activate`, or prefix commands with `uv run`.

## Configuration

Every extraction parameter lives in a YAML or TOML file that you pass into the extractor. Start from `configs/default.yaml` (DINOv2-base) or `configs/dinov3.yaml` / `configs/ct.yaml` and edit what you need.

## Quick start

```python
from dinomics import extract_features, load_config

result = extract_features(
    "example_data/MRI/10000_1000000_t2w.mha",
    config="configs/dinov2.yaml",
    mask="example_data/MRI/10000_1000000.nii.gz",
)

print(result.cls_embeddings.shape)      # (N, 768)
print(result.patch_embeddings.shape)    # (N, 16, 16, 768)
```

Pool to one vector per case, like conventional radiomics, by passing a NumPy reducer (or list of them!). 

```python
import numpy as np
         
case_patches = result.aggregate_patches([np.mean, np.max])  
case_slices = result.aggregate_slices(np.mean)              
```

Reuse one loaded model across many images:

```python
from dinomics import DinoExtractor, load_config

config = load_config("configs/dinov2.yaml")
config.extraction.crop_to_mask = True

extractor = DinoExtractor.from_config(config)
mri = extractor.extract("example_data/MRI/10000_1000000_t2w.mha", mask="example_data/MRI/10000_1000000.nii.gz")
```

CLI:

```bash
uv run dinomics --image example_data/MRI/10000_1000000_t2w.mha \
                --mask example_data/MRI/10000_1000000.nii.gz \
                --config configs/dinov2.yaml \
                --output outputs/mri_dino.npz
```

## Example notebook

`notebooks/feature_extraction.ipynb` runs the extractor on CT, MRI and X-Ray examples and visualizes patch tokens using PCA (as done in the original DINO papers). Give it a whirl!

## Output

| Key | Shape | Meaning |
|---|---|---|
| `cls_embeddings` | `(N, D)` | CLS / slice embedding |
| `patch_embeddings` | `(N, G, G, D)` | Spatial patch tokens |
| `patch_embeddings_maxpooled` | `(N, D)` | Max over patches |
| `patch_mask` | `(N, G, G)` | Patches that overlap the ROI |
| `patch_labels` | `(N, G, G)` | Optional labels from `label_mask` |
| `slice_indices` | `(N,)` | Original slice indices |

`FeatureResult.save(path)` writes a compressed `.npz` (`global_embeddings` is stored as an alias of `cls_embeddings`).


