"""Monocular Depth Estimation module using MiDaS.

This module wraps the MiDaS_small model (v2.1 with EfficientNet-Lite3 backbone).
Model weights are downloaded automatically on first use by PyTorch Hub.

Usage:

    estimator = DepthEstimator()
    depth_map = estimator.estimate(cv2.imread("photo.jpg"))
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

_VALID_MODELS = {"MiDaS_small"}


def _resolve_device(device: str) -> torch.device:
    """Pick the best available device"""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DepthEstimator:
    """Estimate per-pixel relative depth from a single image using MiDaS.

    Params

    - model_type : str
        For this practical example MiDaS small, but extendable for future 
        alternative architectures
    - device : str
        auto (default) picks CUDA > CPU
    """

    def __init__(
        self,
        model_type: str = "MiDaS_small",
        device: str = "auto",
    ) -> None:

        if model_type not in _VALID_MODELS:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Choose from {_VALID_MODELS}."
            )

        self._device = _resolve_device(device)
        logger.info("Loading MiDaS model '%s' on %s ...", model_type, self._device)

        self._model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self._model.to(self._device)
        self._model.eval()

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        if model_type == "MiDaS_small":
            self._transform = midas_transforms.small_transform
        else:
            self._transform = midas_transforms.dpt_transform

    @torch.no_grad()
    def estimate(self, image: np.ndarray) -> np.ndarray:

        """Compute a relative depth map for a single image.

        Params

        - image : np.ndarray
            BGR image as returned by cv2.imread.

        Output

        - np.ndarray
            Float32 array of shape (H, W) with map values in [0, 1].
        """
        h, w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(rgb).to(self._device)

        prediction = self._model(input_batch)

        depth_map = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 1e-6:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(depth_map)

        return depth_map.astype(np.float32)
