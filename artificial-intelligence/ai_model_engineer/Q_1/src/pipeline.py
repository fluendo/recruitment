"""Runs multiple processing modules on an image at the same time using threads.

Usage::

    pipeline = Pipeline(
        face_detector=FaceDetector(),
        depth_estimator=DepthEstimator(),
    )
    results = pipeline.process(image)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class Pipeline:
    """Run multiple image-processing modules in parallel on a single image.

    Params

    - modules
        Keyword arguments where each key is a module name and each value is a
        callable with the signature (image: np.ndarray) -> Any.  Current entries are 
        face_detector=detector.detect and  depth_estimator=estimator.estimate.
    """

    def __init__(self, **modules: Callable[[np.ndarray], Any]) -> None:
        if not modules:
            raise ValueError("At least one module must be provided.")
        self._modules = modules

    def process(self, image: np.ndarray) -> dict[str, Any]:
        """Process image through every registered module in parallel.

        Output

        - dict[str, Any]
            Mapping of module_name -> module_output.  The dict also
            contains a '_timing' key with per-module wall-clock seconds.
        """
        
        results: dict[str, Any] = {}
        timing: dict[str, float] = {}

        def _run(name: str, fn: Callable) -> tuple[str, Any, float]:
            t0 = time.perf_counter()
            output = fn(image)
            elapsed = time.perf_counter() - t0
            return name, output, elapsed

        with ThreadPoolExecutor(max_workers=len(self._modules)) as pool:
            futures = [
                pool.submit(_run, name, fn)
                for name, fn in self._modules.items()
            ]
            for future in futures:
                name, output, elapsed = future.result()
                results[name] = output
                timing[name] = round(elapsed, 4)
                logger.info("Module '%s' finished in %.3fs", name, elapsed)

        results["_timing"] = timing
        return results
