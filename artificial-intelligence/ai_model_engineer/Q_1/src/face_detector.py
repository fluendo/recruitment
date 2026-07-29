"""Face detection module using YOLOv8 trained on WIDER FACE.

The pretrained weights are auto-downloaded on first use so no manual setup is required.

Usage:

    detector = FaceDetector()
    faces = detector.detect(cv2.imread("photo.jpg"))
    for face in faces:
        print(face["bbox"], face["confidence"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_DEFAULT_MODEL_NAME = "yolov8n-face.pt"
_MODEL_URL = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
)


def _resolve_device(device: str) -> str:
    """Pick the best available device"""
    
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _ensure_model(model_path: Path) -> Path:
    """Download the YOLO face model if it does not exist yet."""

    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading YOLOv8n-face weights to %s ...", model_path)
    urlretrieve(_MODEL_URL, str(model_path))
    logger.info("Download complete.")
    return model_path


class FaceDetector:
    """Detect faces in an image using a YOLOv8n-face model.

    Parameters
    
    - model_path : str | Path
        Path to the .pt weights file. 
    - conf_threshold : float
        Minimum confidence score to keep a detection (0-1).
    - device : str
        auto (default) picks CUDA > CPU. 
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = 0.25,
        device: str = "auto",
    ) -> None:

        if model_path is None:
            model_path = _DEFAULT_MODEL_DIR / _DEFAULT_MODEL_NAME
        model_path = Path(model_path)
        _ensure_model(model_path)

        self._device = _resolve_device(device)
        self._conf = conf_threshold
        self._model = YOLO(str(model_path))
        self._model.to(self._device)

    def detect(self, image: np.ndarray) -> list[dict]:
        """Run face detection on a single image.

        Params

        - image : np.ndarray
            BGR image as returned by cv2.imread.

        Output
        
        - list[dict]
            Each dict has:
            - bbox  - [x1, y1, x2, y2] pixel coordinates (list[int]).
            - confidence - detection confidence (float).
        """
        
        results = self._model.predict(
            source=image,
            conf=self._conf,
            verbose=False,
        )

        detections: list[dict] = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0].cpu().numpy())
                detections.append({"bbox": xyxy, "confidence": conf})

        return detections
