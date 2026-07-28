"""Visualization utilities for pipeline results.

Provides helpers to draw face detections on an image, colorize depth maps,
and compose a side-by-side comparison that can be saved or displayed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_BOX_COLOR = (0, 255, 0)  # BGR green
_TEXT_COLOR = (0, 255, 0)
_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5


def draw_faces(image: np.ndarray, faces: list[dict]) -> np.ndarray:
    """Draw bounding boxes and confidence labels on a copy of *image*.

    Parameters
    ----------
    image : np.ndarray
        BGR image.
    faces : list[dict]
        Detections as returned by :meth:`FaceDetector.detect`.

    Returns
    -------
    np.ndarray
        Annotated copy of the input image.
    """
    canvas = image.copy()
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        conf = face["confidence"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), _BOX_COLOR, _BOX_THICKNESS)
        label = f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, 1)
        label_y = max(y1 - 6, th + 4)
        cv2.putText(canvas, label, (x1, label_y), _FONT, _FONT_SCALE, _TEXT_COLOR, 1)
    return canvas


def colorize_depth(depth_map: np.ndarray) -> np.ndarray:
    """Convert a ``[0, 1]`` depth map to a BGR heatmap using INFERNO.

    Parameters
    ----------
    depth_map : np.ndarray
        Float32 array of shape ``(H, W)`` in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        BGR uint8 image of the same spatial dimensions.
    """
    depth_uint8 = (depth_map * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)


def blur_background(
    image: np.ndarray,
    depth_map: np.ndarray,
    threshold: float,
    kernel_size: int = 151,
) -> np.ndarray:
    """Blur pixels considered background by the depth threshold.

    Foreground pixels (depth >= *threshold*) are kept sharp; everything else
    receives a Gaussian blur.  This is useful for visually inspecting what the
    depth threshold classifies as foreground vs background.

    Parameters
    ----------
    image : np.ndarray
        BGR image.
    depth_map : np.ndarray
        Normalised depth map ``(H, W)`` in ``[0, 1]``.
    threshold : float
        Depth value separating foreground (>= threshold) from background.
    kernel_size : int
        Gaussian kernel size for the background blur.

    Returns
    -------
    np.ndarray
        BGR image with background blurred.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    fg_mask = (depth_map >= threshold).astype(np.uint8)
    fg_mask_3ch = np.stack([fg_mask] * 3, axis=-1)

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return np.where(fg_mask_3ch, image, blurred)


def compose_grid(
    image: np.ndarray,
    faces: list[dict],
    depth_map: np.ndarray,
    anon_image: np.ndarray,
    depth_threshold: float,
) -> np.ndarray:
    """Build a 2x2 grid with four diagnostic panels.

    Layout::

        ┌──────────────────┬──────────────────┐
        │ Face detections  │ Depth map        │
        ├──────────────────┼──────────────────┤
        │ Background blur  │ Anonymized       │
        └──────────────────┴──────────────────┘

    Parameters
    ----------
    image : np.ndarray
        Original BGR image.
    faces : list[dict]
        All face detections (before depth filtering).
    depth_map : np.ndarray
        Normalised depth map ``(H, W)`` in ``[0, 1]``.
    anon_image : np.ndarray
        Image with foreground faces anonymized.
    depth_threshold : float
        Threshold used for depth filtering (shown on the background-blur panel).

    Returns
    -------
    np.ndarray
        Combined 2x2 BGR image.
    """
    tl = draw_faces(image, faces)
    tr = colorize_depth(depth_map)
    bl = blur_background(image, depth_map, depth_threshold)
    br = anon_image.copy()

    _label(tl, f"Face detections: {len(faces)}")
    _label(tr, "Depth map (MiDaS)", color=(255, 255, 255))
    _label(bl, f"Background blur (thr={depth_threshold:.2f})")
    _label(br, "Anonymized (foreground faces)")

    top = np.hstack([tl, tr])
    bottom = np.hstack([bl, br])
    return np.vstack([top, bottom])


def _label(
    image: np.ndarray,
    text: str,
    color: tuple[int, int, int] = _TEXT_COLOR,
) -> None:
    """Draw a header label in-place on the top-left corner of *image*."""
    cv2.putText(image, text, (10, 25), _FONT, 0.7, color, 2)


def save_and_show(
    output: np.ndarray,
    output_path: str | Path,
    *,
    display: bool = True,
) -> None:
    """Write *output* to disk and optionally open a window.

    Parameters
    ----------
    output : np.ndarray
        BGR image to save.
    output_path : str | Path
        Destination file path.
    display : bool
        If ``True``, show the image in an OpenCV window until a key is pressed.
    """
    cv2.imwrite(str(output_path), output)

    if display:
        cv2.imshow("Pipeline Output", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
