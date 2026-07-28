# Face Detection + Depth Estimation + Anonymization Pipeline

Modular pipeline that runs **YOLOv8 face detection** and **MiDaS monocular depth estimation** in parallel on a single input image, then **anonymizes** every detected face. Built as a practical code snippet for Q1 Question 7 of the AI Model Engineer technical test.

## Architecture

```
Input Image
     │
     ▼
 ┌────────────────────────────────────────────┐
 │         Pipeline (ThreadPoolExecutor)       │
 │                                             │
 │  ┌─────────────────┐  ┌──────────────────┐ │
 │  │  FaceDetector    │  │  DepthEstimator  │ │
 │  │  (YOLOv8n-face)  │  │  (MiDaS small)   │ │
 │  └────────┬────────┘  └────────┬─────────┘ │
 └───────────┼────────────────────┼───────────┘
             │                    │
             ▼                    ▼
        face bboxes          depth map
             │                    │
             ├────────┬───────────┘
             │        ▼
             │   Visualization
             │   (side-by-side debug image)
             ▼
        Anonymizer
   (blur / pixelation / masking)
             │
             ▼
      Anonymized image
```

Both detection modules run concurrently. The anonymizer then applies the chosen filter to every detected face region sequentially (it depends on detection results).

## Project structure

```
Q_1/
├── config.yaml           # All system parameters (single source of truth)
├── requirements.txt      # Python dependencies
├── .gitignore
└── src/
    ├── __init__.py
    ├── main.py            # CLI entry-point
    ├── face_detector.py   # FaceDetector class (YOLO)
    ├── depth_estimator.py # DepthEstimator class (MiDaS)
    ├── pipeline.py        # Parallel orchestrator
    ├── anonymizer.py      # Anonymizer class (blur / pixelation / masking)
    ├── visualization.py   # Drawing and compositing helpers
    └── README.md          # This file
```

## Requirements

- Python 3.9+
- Dependencies listed in `../requirements.txt`

Install with:

```bash
pip install -r requirements.txt
```

## Configuration (`config.yaml`)

All parameters live in a single YAML file at the project root. The only CLI arguments are `--image` (required) and `--config` (optional, defaults to `config.yaml`).

```yaml
face_detector:
  model_path: models/yolov8n-face.pt   # auto-downloaded if missing
  confidence_threshold: 0.25

depth_estimator:
  model_type: MiDaS_small              # MiDaS_small | DPT_Hybrid | DPT_Large

anonymizer:
  method: blur                          # blur | pixelation | masking
  blur_kernel_size: 51                  # odd integer (blur only)
  pixelation_block_size: 10             # block size (pixelation only)
  mask_color: [0, 0, 0]                # BGR colour (masking only)
  depth_threshold: 0.4                 # only foreground faces are anonymized

pipeline:
  device: auto                          # auto | cpu | cuda | mps

output:
  debug_path: output_debug.jpg          # side-by-side detections + depth map
  anonymized_path: output_anon.jpg      # anonymized result
  display: false                        # open OpenCV window?
```

### Configuration fields

| Section | Key | Description |
|---------|-----|-------------|
| `face_detector` | `model_path` | Path to YOLO `.pt` weights (auto-downloaded if missing) |
| `face_detector` | `confidence_threshold` | Minimum detection confidence (0--1) |
| `depth_estimator` | `model_type` | MiDaS variant to load |
| `anonymizer` | `method` | `blur`, `pixelation`, or `masking` |
| `anonymizer` | `blur_kernel_size` | Gaussian kernel size (must be odd) |
| `anonymizer` | `pixelation_block_size` | Mosaic block size in pixels |
| `anonymizer` | `mask_color` | BGR colour for solid-fill masking |
| `anonymizer` | `depth_threshold` | Faces with median depth >= this value are foreground (0--1) |
| `pipeline` | `device` | Inference device |
| `output` | `debug_path` | Where to save the debug side-by-side image |
| `output` | `anonymized_path` | Where to save the anonymized image |
| `output` | `display` | Whether to open an OpenCV window |

## Usage

From the `Q_1/` directory:

```bash
python -m src.main --image path/to/photo.jpg
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | *(required)* | Path to the input image |
| `--config` | `config.yaml` | Path to the YAML configuration file |

### Example

```bash
python -m src.main --image ../imagery/anonym_1.png
```

### Output

Two images are produced:

1. **Debug image** (`output_debug.jpg` by default) -- side-by-side: original with green face bounding boxes on the left, colourised depth map (INFERNO) on the right.
2. **Anonymized image** (`output_anon.jpg` by default) -- original image with every detected face anonymized using the configured method.

## Modules

### FaceDetector

- **Model**: YOLOv8n-face (~6 MB), trained on the WIDER FACE dataset by [akanametov/yolo-face](https://github.com/akanametov/yolov8-face).
- **Auto-download**: if the weights file is not found at the specified path, it is fetched from the GitHub release automatically.
- **Interface**: `detect(image: np.ndarray) -> list[dict]` where each dict contains `bbox` (`[x1, y1, x2, y2]`) and `confidence` (`float`).

### DepthEstimator

- **Model**: MiDaS v2.1 Small (~63 MB, EfficientNet-Lite3 backbone), loaded via `torch.hub`.
- **Auto-download**: PyTorch Hub handles weight download and caching transparently on first run.
- **Interface**: `estimate(image: np.ndarray) -> np.ndarray` returning a `float32` depth map of shape `(H, W)` normalised to `[0, 1]`. Higher values mean closer to the camera (inverse depth).
- **Alternative models**: set `model_type` to `DPT_Hybrid` or `DPT_Large` in `config.yaml` for higher accuracy at the cost of speed.

### Anonymizer

Applies a privacy-preserving filter to each detected face bounding box. Three methods are available:

- **blur** -- Gaussian blur with configurable kernel size.
- **pixelation** -- downscale then nearest-neighbour upscale to create a mosaic effect.
- **masking** -- solid-colour rectangle fill.

**Interface**: `anonymize(image: np.ndarray, faces: list[dict]) -> np.ndarray` returning an anonymized copy of the image.

### Pipeline

- Accepts any number of named callables via keyword arguments.
- Runs them in parallel with `concurrent.futures.ThreadPoolExecutor`.
- Returns a dict mapping module names to their outputs, plus a `_timing` key with per-module wall-clock durations.

### Visualization

- `draw_faces()` -- annotate an image with bounding boxes and confidence labels.
- `colorize_depth()` -- convert a `[0, 1]` depth map to a BGR heatmap.
- `compose_side_by_side()` -- horizontally concatenate the annotated image and the depth heatmap.
- `save_and_show()` -- write the result to disk and optionally display it.

## Device selection

When `device: auto` (default), the best available backend is selected automatically:

1. **CUDA** -- if an NVIDIA GPU with CUDA is detected.
2. **MPS** -- if running on Apple Silicon (macOS).
3. **CPU** -- fallback.

## Extending the pipeline

The pipeline is designed for iterative extension. To add a new parallel module:

1. Create a class with a method that takes `np.ndarray` and returns a result.
2. Pass it as an additional keyword argument when constructing the `Pipeline`:

```python
pipeline = Pipeline(
    face_detector=detector.detect,
    depth_estimator=estimator.estimate,
    my_new_module=new_module.run,      # runs in parallel with the others
)
```

No changes to `pipeline.py` are needed.
