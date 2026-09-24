"""Load and validate dinomics extraction configs from YAML, TOML, or dicts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


@dataclass
class ModelConfig:
    """HuggingFace DINO model and processor settings."""

    name: str = "facebook/dinov2-base"
    feature_type: str = "dinov2"
    image_size: int = 224
    device: str = "auto"
    n_register_tokens: int | None = None
    do_center_crop: bool = False

    def __post_init__(self) -> None:
        self.feature_type = self.feature_type.lower()
        if self.feature_type not in {"dinov2", "dinov3"}:
            raise ValueError(
                f"feature_type must be 'dinov2' or 'dinov3', got {self.feature_type!r}"
            )
        if self.n_register_tokens is None:
            self.n_register_tokens = 4 if self.feature_type == "dinov3" else 0


@dataclass
class IntensityConfig:
    """Optional intensity normalization applied per slice before RGB conversion."""

    clip_percentiles: list[float] | None = None
    window: list[float] | None = None


@dataclass
class ExtractionSettings:
    """How volumes and optional masks are turned into model inputs."""

    apply_image_mask: bool = True
    apply_patch_mask: bool = True
    skip_empty_slices: bool = True
    slice_axis: int = 0
    slice_indices: list[int] | None = None
    slice_step: int = 1
    max_slices: int | None = None
    crop_to_mask: bool = False
    crop_size: int | None = None
    min_mask_pixels: int = 1


@dataclass
class OutputSettings:
    """Optional extras stored alongside the core embeddings."""

    include_flat_patches: bool = False


@dataclass
class DinomicsConfig:
    """Full configuration passed to :func:`dinomics.extract_features`."""

    model: ModelConfig = field(default_factory=ModelConfig)
    intensity: IntensityConfig = field(default_factory=IntensityConfig)
    extraction: ExtractionSettings = field(default_factory=ExtractionSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FLAT_ALIASES = {
    "model_name": ("model", "name"),
    "feature_type": ("model", "feature_type"),
    "image_size": ("model", "image_size"),
    "device": ("model", "device"),
    "n_register_tokens": ("model", "n_register_tokens"),
    "do_center_crop": ("model", "do_center_crop"),
}


def _dataclass_from_dict(cls, data: dict[str, Any] | None):
    if not data:
        return cls()
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in allowed})


def config_from_dict(data: dict[str, Any] | None) -> DinomicsConfig:
    """Build a :class:`DinomicsConfig` from a nested or flat mapping."""
    raw = dict(data or {})

    for alias, (section, key) in _FLAT_ALIASES.items():
        if alias in raw:
            raw.setdefault(section, {})
            if isinstance(raw[section], dict):
                raw[section].setdefault(key, raw[alias])

    return DinomicsConfig(
        model=_dataclass_from_dict(ModelConfig, raw.get("model")),
        intensity=_dataclass_from_dict(IntensityConfig, raw.get("intensity")),
        extraction=_dataclass_from_dict(ExtractionSettings, raw.get("extraction")),
        output=_dataclass_from_dict(OutputSettings, raw.get("output")),
    )


def _read_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("rb") as handle:
        if suffix in {".yaml", ".yml"}:
            loaded = yaml.safe_load(handle) or {}
        elif suffix == ".toml":
            if tomllib is None:
                raise RuntimeError("tomllib is required to load TOML configs")
            loaded = tomllib.load(handle)
        else:
            raise ValueError(
                f"Unsupported config format {path.suffix!r}. Use .yaml, .yml, or .toml"
            )
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping, got {type(loaded)}")
    return loaded


def load_config(source: str | Path | dict[str, Any] | DinomicsConfig | None = None) -> DinomicsConfig:
    """Load an extraction config from a path, mapping, or existing config object.

    Parameters
    ----------
    source
        Path to a YAML/TOML file, a dictionary, a :class:`DinomicsConfig`,
        or ``None`` for package defaults.
    """
    if source is None:
        return DinomicsConfig()
    if isinstance(source, DinomicsConfig):
        return source
    if isinstance(source, dict):
        return config_from_dict(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    return config_from_dict(_read_mapping(path))
