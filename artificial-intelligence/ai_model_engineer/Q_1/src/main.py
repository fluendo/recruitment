"""CLI entry-point for the video face-anonymization pipeline.

Run with::

    python -m src.main --video path/to/video.mp4

All system parameters are read from a YAML config file (default:
``config.yaml``).  Only ``--video`` and ``--config`` are accepted as CLI
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
from src.visualization import compose_grid, refine_depth_mask


def _filter_foreground_faces(
    faces: list[dict],
    refined_mask: np.ndarray,
) -> list[dict]:
    """Keep only faces that fall predominantly inside the foreground mask.

    Uses the morphologically-refined depth mask instead of raw depth
    values so that noisy pixels at bbox boundaries do not cause
    foreground faces to be dropped.

    A face is kept when the mean mask value inside its bbox is >= 0.5,
    meaning more than half of its pixels are classified as foreground.
    """
    foreground = []
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        roi_mask = refined_mask[max(y1, 0):y2, max(x1, 0):x2]
        if roi_mask.size > 0 and float(np.mean(roi_mask)) >= 0.5:
            foreground.append(face)
    return foreground


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run face detection (YOLO) and depth estimation (MiDaS) "
            "in parallel on each frame of a video, then anonymize "
            "foreground faces.  Outputs a 2x2 diagnostic grid video "
            "and a clean anonymized video."
        ),
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video.",
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


def _open_video(video_path: Path) -> tuple[cv2.VideoCapture, int, float, int, int]:
    """Open a video file and return (capture, total_frames, fps, width, height)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Error: could not open video at '{video_path}'")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, total_frames, fps, width, height


def _create_writer(
    path: str | Path,
    codec: str,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """Create and return a VideoWriter, exiting on failure."""
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        sys.exit(f"Error: could not create video writer for '{path}'")
    return writer


def _format_eta(seconds: float) -> str:
    """Return a human-readable ETA string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    cfg = _load_config(Path(args.config))

    video_path = Path(args.video)
    if not video_path.is_file():
        sys.exit(f"Error: video not found at '{video_path}'")

    # --- Open input video -----------------------------------------------------
    cap, total_frames, fps, width, height = _open_video(video_path)
    logging.info(
        "Video opened: %s (%dx%d, %.2f fps, %d frames)",
        video_path.name, width, height, fps, total_frames,
    )

    # --- Read config sections -------------------------------------------------
    cfg_face = cfg["face_detector"]
    cfg_depth = cfg["depth_estimator"]
    cfg_anon = cfg["anonymizer"]
    cfg_pipe = cfg["pipeline"]
    cfg_out = cfg["output"]

    cfg_mask = cfg.get("depth_mask", {})

    device = cfg_pipe.get("device", "auto")
    depth_threshold = cfg_anon.get("depth_threshold", 0.4)

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

    pipeline = Pipeline(
        face_detector=detector.detect,
        depth_estimator=estimator.estimate,
    )

    # --- Create output video writers ------------------------------------------
    codec = cfg_out.get("codec", "mp4v")
    grid_path = cfg_out.get("grid_path", "output_grid.mp4")
    anon_path = cfg_out.get("anonymized_path", "output_anon.mp4")

    grid_writer = _create_writer(grid_path, codec, fps, width * 2, height * 2)
    anon_writer = _create_writer(anon_path, codec, fps, width, height)

    # --- Frame loop -----------------------------------------------------------
    logging.info("Processing frames (offline, device=%s) ...", device)
    t_start = time.perf_counter()
    last_log_time = t_start
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = pipeline.process(frame)
            faces = results["face_detector"]
            depth_map = results["depth_estimator"]

            refined_mask = refine_depth_mask(depth_map, depth_threshold, **cfg_mask)
            fg_faces = _filter_foreground_faces(faces, refined_mask)
            anon_frame = anonymizer.anonymize(frame, fg_faces)

            grid_frame = compose_grid(
                frame, faces, depth_map, anon_frame, depth_threshold,
                mask_cfg=cfg_mask,
            )
            grid_writer.write(grid_frame)
            anon_writer.write(anon_frame)

            frame_idx += 1

            now = time.perf_counter()
            if frame_idx % 10 == 0 or (now - last_log_time) >= 5.0:
                elapsed = now - t_start
                avg_fps = frame_idx / elapsed
                pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                remaining = (total_frames - frame_idx) / avg_fps if avg_fps > 0 else 0
                logging.info(
                    "Frame %d/%d (%.1f%%)  |  %.2f fps  |  ETA %s",
                    frame_idx, total_frames, pct, avg_fps, _format_eta(remaining),
                )
                last_log_time = now
    finally:
        cap.release()
        grid_writer.release()
        anon_writer.release()

    total_time = time.perf_counter() - t_start
    avg_fps = frame_idx / total_time if total_time > 0 else 0
    logging.info(
        "Done: %d frames in %s (avg %.2f fps)",
        frame_idx, _format_eta(total_time), avg_fps,
    )
    logging.info("Grid video  -> %s", grid_path)
    logging.info("Anon video  -> %s", anon_path)


if __name__ == "__main__":
    main()
