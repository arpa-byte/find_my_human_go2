from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CameraTargetState:
    """
    Camera-only target state for Phase 3A.

    This is an internal Python data structure for now.
    Later, the same fields can be mapped to ROS messages for
    visualization, fusion, or logging.
    """
    timestamp_sec: float = 0.0
    target_visible: bool = False

    # Bounding box in image pixel coordinates: [x1, y1, x2, y2]
    bbox: List[int] = field(default_factory=lambda: [0, 0, 0, 0])

    # Center pixel in image coordinates: [u, v]
    center_pixel: List[int] = field(default_factory=lambda: [0, 0])

    # Estimated depth in meters
    depth_m: float = 0.0

    # 3D position in the camera optical frame: [x, y, z] in meters
    position_cam: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Detector confidence or target confidence
    confidence: float = 0.0

    # Fixed target label or later track identifier
    track_id: str = "target"

    # Optional note for debugging or failure state labeling
    status: str = "not_initialized"


def empty_target_state() -> CameraTargetState:
    """
    Returns a default 'no target visible' state.
    """
    return CameraTargetState()
