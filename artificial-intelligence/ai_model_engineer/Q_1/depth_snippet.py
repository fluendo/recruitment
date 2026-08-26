from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol
import numpy as np


class Zone(Enum):
    FOREGROUND = auto()
    BACKGROUND = auto()


@dataclass(frozen=True)
class DepthThresholds:
    """Relative inverse depth: higher value means closer.
    Two thresholds so an estimate sitting on the boundary
    cannot flip the decision every frame."""
    enter_background: float
    exit_background: float

    def __post_init__(self):
        if self.exit_background <= self.enter_background:
            raise ValueError("exit must sit above enter to leave a margin")

    def classify(self, depth: float, previous: Zone) -> Zone:
        if previous is Zone.BACKGROUND:
            return Zone.FOREGROUND if depth > self.exit_background else Zone.BACKGROUND
        return Zone.BACKGROUND if depth < self.enter_background else Zone.FOREGROUND


@dataclass
class BackgroundMap:
    """Static-camera snapshot. Valid until the geometry or the camera moves."""
    depth: np.ndarray
    edges: np.ndarray             # structure at build time, for drift checking
    frame_index: int

    def structural_change(self, frame_edges: np.ndarray) -> float:
        """Differenced on edges, not raw pixels, so a cloud passing
        changes brightness without registering as a geometry change."""
        return float(np.mean(np.abs(frame_edges - self.edges)))


class DepthEstimator(Protocol):
    def infer(self, frame: np.ndarray) -> np.ndarray: ...


@dataclass
class BackgroundDepthLayer:
    estimator: DepthEstimator
    thresholds: DepthThresholds
    refresh_every: int                      # floor; neither test catches everything
    global_change_ratio: float              # spread across the frame -> camera moved
    drift_tolerance: float
    _map: BackgroundMap | None = None
    _zones: dict[int, Zone] = field(default_factory=dict)

    def needs_rebuild(self, frame_index: int, frame_edges: np.ndarray) -> bool:
        if self._map is None:
            return True
        if frame_index - self._map.frame_index >= self.refresh_every:
            return True
        change = self._map.structural_change(frame_edges)
        if change > self.global_change_ratio:
            return True                     # scene rearranged or camera moved
        return change > self.drift_tolerance  # slow mount drift, no visible event

    def rebuild(self, frame, frame_edges, frame_index, occupied: bool) -> None:
        if occupied:
            return                          # never throw a standing person into the map
        self._map = BackgroundMap(self.estimator.infer(frame), frame_edges, frame_index)

    def zone_of(self, track_id: int, depth: float) -> Zone:
        previous = self._zones.get(track_id, Zone.FOREGROUND)
        current = self.thresholds.classify(depth, previous)
        self._zones[track_id] = current
        return current


@dataclass(frozen=True)
class Track:
    track_id: int
    box: tuple[int, int, int, int]
    is_protected: bool
    identity_resolved: bool

    @property
    def face_width(self) -> int:
        x1, _, x2, _ = self.box
        return x2 - x1


@dataclass
class AnonymisationPolicy:
    """Identity outranks depth in both directions."""
    depth_layer: BackgroundDepthLayer

    def strength_for(self, track: Track) -> float:
        # scaled to the face, since a radius that destroys a 30px face
        # leaves a 150px one perfectly readable
        return max(1.0, track.face_width / 10.0)

    def decide(self, track: Track, depth: float) -> float | None:
        if track.is_protected or not track.identity_resolved:
            return self.strength_for(track)          # blur on doubt
        if self.depth_layer.zone_of(track.track_id, depth) is Zone.BACKGROUND:
            return self.strength_for(track)          # scene rule, clearance doesn't exempt
        return None
