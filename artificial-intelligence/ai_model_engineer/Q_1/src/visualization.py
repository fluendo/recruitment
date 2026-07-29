"""Visualization utilities for pipeline results"""

from __future__ import annotations

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX

_DEFAULTS = {
    "box_color": (0, 255, 0),
    "text_color": (0, 255, 0),
    "box_thickness": 2,
    "font_scale": 0.5,
    "bg_blur_kernel_size": 151,
}

_cfg: dict = {}

def configure(cfg: dict) -> None:
    """Store the visualization config. Missing keys fall back to _DEFAULTS."""

    _cfg.update(cfg)


def _get(key: str):
    """Get a config value by key, falling back to _DEFAULTS if not found."""
    
    val = _cfg.get(key, _DEFAULTS[key])
    return tuple(val) if isinstance(val, list) else val


def draw_faces(image: np.ndarray, faces: list[dict]) -> np.ndarray:
    """Draw bounding boxes and confidence labels on a copy of image.

    Params

    - image : np.ndarray
        BGR image.
    - faces : list[dict]
        Detections as returned by FaceDetector.detect.

    Output

    - np.ndarray
        Copy of the input image with bounding boxes and confidence labels.
    """

    canvas = image.copy()
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        conf = face["confidence"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), _get("box_color"), _get("box_thickness"))
        label = f"{conf:.2f}"
        font_scale = _get("font_scale")
        (tw, th), _ = cv2.getTextSize(label, _FONT, font_scale, 1)
        label_y = max(y1 - 6, th + 4)
        cv2.putText(canvas, label, (x1, label_y), _FONT, font_scale, _get("text_color"), 1)
    return canvas


def colorize_depth(depth_map: np.ndarray) -> np.ndarray:
    """Convert a [0, 1] depth map to a BGR heatmap using INFERNO colormap.

    Params
    
    - depth_map : np.ndarray
        Float32 array of shape (H, W) in [0, 1].

    Output

    - np.ndarray
        BGR uint8 image of the same spatial dimensions.
    """

    depth_uint8 = (depth_map * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)


def refine_depth_mask(
    depth_map: np.ndarray,
    threshold: float,
    close_ksize: int = 15,
    dilate_ksize: int = 7,
    feather_ksize: int = 21,
) -> np.ndarray:
    """Build a cleaned foreground mask from a normalised depth map.

    Applies morphological closing (to fill interior holes), dilation (to
    expand the foreground boundary), and an optional Gaussian feather (to
    produce soft edges for alpha-blending compositing).

    Params

    - depth_map : np.ndarray
        Float32 array of shape (H, W) in [0, 1].
    - threshold : float
        Depth value separating foreground (>= threshold) from background.
    - close_ksize : int
        Kernel size for morphological closing (fills small holes).
    - dilate_ksize : int
        Kernel size for dilation (expands foreground contour outward).
    - feather_ksize : int
        Gaussian kernel size applied to the mask for soft edges.
        Set to 0 to disable feathering and keep a hard mask.

    Output
    
    - np.ndarray
        Float32 mask of shape (H, W) in [0, 1].  Values near 1
        are foreground; values near 0 are background.  When feathering is
        enabled the mask has smooth gradients at object boundaries.
    """

    mask = (depth_map >= threshold).astype(np.uint8) * 255

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize)
    )
    mask = cv2.dilate(mask, dilate_kernel)

    if feather_ksize > 0:
        fk = feather_ksize | 1
        mask = cv2.GaussianBlur(mask, (fk, fk), 0)

    return mask.astype(np.float32) / 255.0


def blur_background(
    image: np.ndarray,
    depth_map: np.ndarray,
    threshold: float,
    kernel_size: int | None = None,
    close_ksize: int = 15,
    dilate_ksize: int = 7,
    feather_ksize: int = 21,
) -> np.ndarray:
    """Blur pixels considered background by the depth threshold.

    Foreground pixels (depth >= *threshold*) are kept sharp; everything else
    receives a Gaussian blur.  A refined depth mask with morphological
    cleanup and feathering is used so that foreground silhouettes are not
    cut by noisy depth boundaries.

    Params
    
    - image : np.ndarray
        BGR image.
    - depth_map : np.ndarray
        Normalised depth map (H, W) in [0, 1].
    - threshold : float
        Depth value separating foreground (>= threshold) from background.
    - kernel_size : int
        Gaussian kernel size for the background blur.
    - close_ksize : int
        Passed to refine_depth_mask.
    - dilate_ksize : int
        Passed to refine_depth_mask.
    - feather_ksize : int
        Passed to refine_depth_mask.

    Output
    
    - np.ndarray
        Copy of the input image with background blurred.
    """

    if kernel_size is None:
        kernel_size = _get("bg_blur_kernel_size")
    if kernel_size % 2 == 0:
        kernel_size += 1

    alpha = refine_depth_mask(depth_map, threshold, close_ksize, dilate_ksize, feather_ksize)
    alpha_3ch = np.stack([alpha] * 3, axis=-1)

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return (image * alpha_3ch + blurred * (1.0 - alpha_3ch)).astype(np.uint8)


def compose_grid(
    image: np.ndarray,
    faces: list[dict],
    depth_map: np.ndarray,
    anon_image: np.ndarray,
    depth_threshold: float,
    mask_cfg: dict | None = None,
) -> np.ndarray:
    """Build a 2x2 grid with four diagnostic panels.

    Params

    - image : np.ndarray
        BGR image.
    - faces : list[dict]
        All face detections (before depth filtering).
    - depth_map : np.ndarray
        Normalised depth map (H, W) in [0, 1].
    - anon_image : np.ndarray
        Image with foreground faces anonymized.
    - depth_threshold : float
        Threshold used for depth filtering (shown on the background-blur panel).
    mask_cfg : dict | None
        Optional dict with keys close_ksize, dilate_ksize,
        feather_ksize forwarded to blur_background.

    Output
    
    - np.ndarray
        Combined 2x2 BGR image with diagnostic panels.
    """

    tl = draw_faces(image, faces)
    tr = colorize_depth(depth_map)
    bl = blur_background(image, depth_map, depth_threshold, **(mask_cfg or {}))
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
    color: tuple[int, int, int] | None = None,
) -> None:
    """Draw a header label in-place on the top-left corner of image

    Params

    - image : np.ndarray
        BGR image.
    - text : str
        Text to draw.
    - color : tuple[int, int, int]
        Color of the text in BGR.
    """

    if color is None:
        color = _get("text_color")
    cv2.putText(image, text, (10, 25), _FONT, 0.7, color, 2)
