# dinomics

Extract **DINOv2 / DINOv3** features from 2D medical images and 3D volumes (slice-wise).

For each processed slice the package returns:

- **CLS** embeddings — one vector per slice (`cls_embeddings`)
- **Patch** embeddings — a spatial grid of patch tokens (`patch_embeddings`)
- **Max-pooled patches** — a second slice-level descriptor (`patch_embeddings_maxpooled`)

Any grayscale medical image that SimpleITK can read (`.mha`, `.nii`, `.nii.gz`, …) works.

**First time here?** Read the [user guide](docs/user-guide.md) — install, first extraction, configs, masks, outputs, and common problems.

## Install

The project is managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`). It targets Python 3.11–3.13 (`.python-version` pins 3.12).

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

Every extraction parameter lives in a YAML or TOML file that you pass into the extractor. Start from `configs/default.yaml` (DINOv2-base) or `configs/dinov3.yaml` / `configs/ct.yaml` and edit what you need:

```yaml
model:
  name: facebook/dinov2-base
  feature_type: dinov2          # dinov2 | dinov3
  image_size: 224
  device: auto                  # auto | cuda | cpu

extraction:
  apply_image_mask: true
  apply_patch_mask: true
  skip_empty_slices: true
  crop_to_mask: false           # square crop around the mask centroid
  crop_size: null               # in-plane H=W; expands + pads if the mask does not fit
```

`feature_type: dinov3` drops the CLS token **and** the register tokens before the patch grid (4 registers by default, matching `facebook/dinov3-vitb16-pretrain-lvd1689m`).

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

`notebooks/feature_extraction.ipynb` runs the extractor on the bundled MRI and CT examples and visualizes patch tokens using PCA.

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
