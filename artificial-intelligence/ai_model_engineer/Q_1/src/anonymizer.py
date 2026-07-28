"""Face anonymization module.

Applies a privacy-preserving filter to every detected face region in an image.
Three methods are supported:

- **blur** -- Gaussian blur with a configurable kernel size.
- **pixelation** -- downscale then nearest-neighbour upscale to create a mosaic.
- **masking** -- solid-colour rectangle fill.

The module runs *after* face detection (it needs the bounding boxes), so it is
called sequentially rather than registered in the parallel pipeline.

Typical usage:

    anonymizer = Anonymizer(method="blur", blur_kernel_size=51)
    anon_image = anonymizer.anonymize(image, faces)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_VALID_METHODS = {"blur", "pixelation", "masking"}


class Anonymizer:
    """Anonymize face regions in an image.

    Parameters
    ----------
    method : str
        ``'blur'``, ``'pixelation'``, or ``'masking'``.
    blur_kernel_size : int
        Gaussian kernel size (must be odd). Only used when *method* is
        ``'blur'``.
    pixelation_block_size : int
        The face ROI is shrunk to this many pixels on its shortest side before
        being scaled back up with nearest-neighbour interpolation. Only used
        when *method* is ``'pixelation'``.
    mask_color : tuple[int, int, int]
        BGR colour used for the solid fill. Only used when *method* is
        ``'masking'``.
    """

    def __init__(
        self,
        method: str = "blur",
        blur_kernel_size: int = 51,
        pixelation_block_size: int = 10,
        mask_color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Choose from {_VALID_METHODS}."
            )
        self._method = method

        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
            logger.warning(
                "blur_kernel_size must be odd; adjusted to %d.", blur_kernel_size
            )
        self._blur_k = blur_kernel_size
        self._pix_block = max(pixelation_block_size, 2)
        self._mask_color = tuple(mask_color)

    def anonymize(self, image: np.ndarray, faces: list[dict]) -> np.ndarray:
        """Return a copy of *image* with every face region anonymized.

        Parameters
        ----------
        image : np.ndarray
            BGR image (unchanged -- a copy is made internally).
        faces : list[dict]
            Detections as returned by :meth:`FaceDetector.detect`.

        Returns
        -------
        np.ndarray
            Anonymized BGR image.
        """
        canvas = image.copy()
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, canvas.shape[1]), min(y2, canvas.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue

            if self._method == "blur":
                self._apply_blur(canvas, x1, y1, x2, y2)
            elif self._method == "pixelation":
                self._apply_pixelation(canvas, x1, y1, x2, y2)
            else:
                self._apply_masking(canvas, x1, y1, x2, y2)

        logger.info(
            "Anonymized %d face(s) using '%s'.", len(faces), self._method
        )
        return canvas

    def _apply_blur(
        self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> None:
        roi = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (self._blur_k, self._blur_k), 0)

    def _apply_pixelation(
        self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> None:
        roi = img[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        small = cv2.resize(
            roi, (self._pix_block, self._pix_block), interpolation=cv2.INTER_LINEAR
        )
        img[y1:y2, x1:x2] = cv2.resize(
            small, (w, h), interpolation=cv2.INTER_NEAREST
        )

    def _apply_masking(
        self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> None:
        cv2.rectangle(img, (x1, y1), (x2, y2), self._mask_color, cv2.FILLED)
