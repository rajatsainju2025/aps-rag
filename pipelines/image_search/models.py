"""
Lazy-loaded, thread-safe model singletons for SigLIP and DINOv2.
Models stay on GPU after first load; subsequent calls are instant.
"""
import logging
import threading
from typing import ClassVar

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from .config import DEVICE, DINOV2_MODEL_ID, SIGLIP_MODEL_ID

log = logging.getLogger("image_search.models")
_lock = threading.Lock()


class _SigLIP:
    """Wraps SigLIP for text->embedding AND image->embedding."""

    _instance: ClassVar["_SigLIP | None"] = None

    def __new__(cls) -> "_SigLIP":
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._device = torch.device(
                        DEVICE if torch.cuda.is_available() else "cpu"
                    )
                    log.info(
                        "Loading SigLIP model %s on %s …", SIGLIP_MODEL_ID, obj._device
                    )
                    obj.model = (
                        AutoModel.from_pretrained(
                            SIGLIP_MODEL_ID, dtype=torch.float16
                        )
                        .eval()
                        .to(obj._device)
                    )
                    obj.processor = AutoProcessor.from_pretrained(
                        SIGLIP_MODEL_ID, use_fast=True
                    )
                    obj.tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL_ID)
                    log.info("SigLIP model loaded.")
                    cls._instance = obj
        return cls._instance

    @torch.inference_mode()
    def embed_images(self, pil_images: list[Image.Image]) -> NDArray[np.float32]:
        """Return L2-normalized [N, D] float32 numpy array."""
        inputs = self.processor(
            images=pil_images, return_tensors="pt", padding=True
        ).to(self._device)
        feats = self.model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.cpu().float().numpy()

    @torch.inference_mode()
    def embed_text(self, texts: list[str]) -> NDArray[np.float32]:
        """Return L2-normalized [N, D] float32 numpy array."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        ).to(self._device)
        feats = self.model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.cpu().float().numpy()


class _DINOv2:
    """Wraps DINOv2-base for image->embedding (CLS token)."""

    _instance: ClassVar["_DINOv2 | None"] = None

    def __new__(cls) -> "_DINOv2":
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._device = torch.device(
                        DEVICE if torch.cuda.is_available() else "cpu"
                    )
                    log.info(
                        "Loading DINOv2 model %s on %s …", DINOV2_MODEL_ID, obj._device
                    )
                    obj.model = (
                        AutoModel.from_pretrained(
                            DINOV2_MODEL_ID, dtype=torch.float16
                        )
                        .eval()
                        .to(obj._device)
                    )
                    obj.processor = AutoProcessor.from_pretrained(
                        DINOV2_MODEL_ID, use_fast=True
                    )
                    log.info("DINOv2 model loaded.")
                    cls._instance = obj
        return cls._instance

    @torch.inference_mode()
    def embed_images(self, pil_images: list[Image.Image]) -> NDArray[np.float32]:
        """Return L2-normalized [N, D] float32 numpy array from CLS token."""
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self._device)
        outputs = self.model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        cls_emb = torch.nn.functional.normalize(cls_emb, dim=-1)
        return cls_emb.cpu().float().numpy()


def get_siglip() -> _SigLIP:
    """Return the singleton SigLIP model wrapper."""
    return _SigLIP()


def get_dinov2() -> _DINOv2:
    """Return the singleton DINOv2 model wrapper."""
    return _DINOv2()
