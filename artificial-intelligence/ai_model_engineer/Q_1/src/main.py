"""CLI entry-point for the face-detection + depth-estimation pipeline.

Run with::

    python -m src.main --image path/to/photo.jpg

All system parameters are read from a YAML config file (default:
``config.yaml``).  Only ``--image`` and ``--config`` are accepted as CLI
arguments.

Use ``--help`` to see available options.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.anonymizer import Anonymizer
from src.depth_estimator import DepthEstimator
from src.face_detector import FaceDetector
from src.pipeline import Pipeline
from src.visualization import compose_grid, save_and_show


def _filter_foreground_faces(
    faces: list[dict],
    depth_map: np.ndarray,
    threshold: float,
) -> list[dict]:
    """Keep only faces whose median depth is at or above *threshold*.

    Higher depth values mean closer to the camera (MiDaS inverse depth,
    normalised to [0, 1]), so faces with ``median >= threshold`` are
    considered foreground.
    """
    foreground = []
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        roi_depth = depth_map[max(y1, 0):y2, max(x1, 0):x2]
        if roi_depth.size > 0 and float(np.median(roi_depth)) >= threshold:
            foreground.append(face)
    return foreground


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run face detection (YOLO) and depth estimation (MiDaS) "
            "in parallel on a single image, then anonymize detected faces."
        ),
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    return parser


def _load_config(config_path: Path) -> dict:
    """Read and return the YAML configuration, with basic validation."""
    if not config_path.is_file():
        sys.exit(f"Error: config file not found at '{config_path}'")

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    required_sections = ["face_detector", "depth_estimator", "anonymizer", "pipeline", "output"]
    for section in required_sections:
        if section not in cfg:
            sys.exit(f"Error: missing '{section}' section in {config_path}")

    return cfg


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    cfg = _load_config(Path(args.config))

    image_path = Path(args.image)
    if not image_path.is_file():
        sys.exit(f"Error: image not found at '{image_path}'")

    image = cv2.imread(str(image_path))
    if image is None:
        sys.exit(f"Error: could not decode image at '{image_path}'")

    logging.info("Image loaded: %s (%dx%d)", image_path.name, image.shape[1], image.shape[0])

    # --- Read config sections -------------------------------------------------
    cfg_face = cfg["face_detector"]
    cfg_depth = cfg["depth_estimator"]
    cfg_anon = cfg["anonymizer"]
    cfg_pipe = cfg["pipeline"]
    cfg_out = cfg["output"]

    device = cfg_pipe.get("device", "auto")

    # --- Initialise modules ---------------------------------------------------
    detector = FaceDetector(
        model_path=cfg_face.get("model_path"),
        conf_threshold=cfg_face.get("confidence_threshold", 0.25),
        device=device,
    )
    estimator = DepthEstimator(
        model_type=cfg_depth.get("model_type", "MiDaS_small"),
        device=device,
    )
    anonymizer = Anonymizer(
        method=cfg_anon.get("method", "blur"),
        blur_kernel_size=cfg_anon.get("blur_kernel_size", 51),
        pixelation_block_size=cfg_anon.get("pixelation_block_size", 10),
        mask_color=tuple(cfg_anon.get("mask_color", [0, 0, 0])),
    )

    # --- Build and run pipeline -----------------------------------------------
    pipeline = Pipeline(
        face_detector=detector.detect,
        depth_estimator=estimator.estimate,
    )

    logging.info("Running pipeline ...")
    t0 = time.perf_counter()
    results = pipeline.process(image)
    total = time.perf_counter() - t0

    faces = results["face_detector"]
    depth_map = results["depth_estimator"]
    timing = results["_timing"]

    logging.info(
        "Pipeline finished in %.3fs  (face_detector=%.3fs, depth_estimator=%.3fs)",
        total,
        timing["face_detector"],
        timing["depth_estimator"],
    )
    logging.info("Detected %d face(s).", len(faces))

    # --- Depth-filter faces (keep foreground only) -----------------------------
    depth_threshold = cfg_anon.get("depth_threshold", 0.4)
    fg_faces = _filter_foreground_faces(faces, depth_map, depth_threshold)
    logging.info("Foreground faces: %d / %d (threshold=%.2f)", len(fg_faces), len(faces), depth_threshold)

    # --- Anonymize foreground faces -------------------------------------------
    anon_img = anonymizer.anonymize(image, fg_faces)

    # --- Compose 2x2 grid and save -------------------------------------------
    grid = compose_grid(image, faces, depth_map, anon_img, depth_threshold)
    output_path = cfg_out.get("output_path", "output.jpg")
    display = cfg_out.get("display", False)
    save_and_show(grid, output_path, display=display)
    logging.info("Output grid saved to %s", output_path)


if __name__ == "__main__":
    main()
