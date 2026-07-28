"""Pipeline orchestrator that runs processing modules in parallel.

The :class:`Pipeline` accepts any number of named *modules* (callables that
take an image and return a result) and executes them concurrently using a
:class:`~concurrent.futures.ThreadPoolExecutor`.  PyTorch releases the GIL
during tensor computation, so thread-level parallelism provides real speed-up
when modules are GPU-bound.

Currently wired with :class:`FaceDetector` and :class:`DepthEstimator`, but
the design is intentionally generic so that new modules (e.g. anonymiser, ReID)
can be registered without changing the orchestration logic.

Typical usage:

    pipeline = Pipeline(
        face_detector=FaceDetector(),
        depth_estimator=DepthEstimator(),
    )
    results = pipeline.process(image)
    # results == {"face_detector": [...], "depth_estimator": np.ndarray}
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

    Parameters
    ----------
    **modules
        Keyword arguments where each key is a module name and each value is a
        callable with the signature ``(image: np.ndarray) -> Any``.  Typical
        entries are ``face_detector=detector.detect`` and
        ``depth_estimator=estimator.estimate``.
    """

    def __init__(self, **modules: Callable[[np.ndarray], Any]) -> None:
        if not modules:
            raise ValueError("At least one module must be provided.")
        self._modules = modules

    def process(self, image: np.ndarray) -> dict[str, Any]:
        """Process *image* through every registered module in parallel.

        Returns
        -------
        dict[str, Any]
            Mapping of ``module_name -> module_output``.  The dict also
            contains a ``'_timing'`` key with per-module wall-clock seconds.
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
