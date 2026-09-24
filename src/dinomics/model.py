"""HuggingFace DINOv2 / DINOv3 loader."""

from __future__ import annotations

import torch
from transformers import AutoImageProcessor, AutoModel

from dinomics.config import ModelConfig


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_dino_model(model_cfg: ModelConfig) -> tuple[AutoImageProcessor, AutoModel, torch.device]:
    """Load the HuggingFace processor and model, and move the model to device."""
    device = resolve_device(model_cfg.device)

    processor = AutoImageProcessor.from_pretrained(model_cfg.name)
    processor.do_center_crop = model_cfg.do_center_crop
    # Force a square canvas so the patch grid is always G x G, including after a
    # rectangular mask crop (shortest_edge resize would otherwise yield a non-square grid).
    size = int(model_cfg.image_size)
    processor.size = {"height": size, "width": size}
    if hasattr(processor, "crop_size"):
        processor.crop_size = {"height": size, "width": size}

    model = AutoModel.from_pretrained(model_cfg.name)
    model.to(device)
    model.eval()
    return processor, model, device
