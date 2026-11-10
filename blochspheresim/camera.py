"""Orbit camera and matrix helpers shared with the renderer."""

from __future__ import annotations

import math

import numpy as np

from .config import Controls


BLOCH_TO_OPENGL = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
WINDOW_ASPECT = 1280.0 / 800.0
FIELD_OF_VIEW_DEGREES = 45.0
NEAR_PLANE = 0.1
FAR_PLANE = 100.0
MAX_ELEVATION = math.pi / 2.0 - 0.01


def blochToOpenGl(point: tuple[float, float, float]) -> tuple[float, float, float]:
    """Map a Bloch-space point into OpenGL space using the shared convention matrix."""
    mapped = BLOCH_TO_OPENGL @ np.asarray(point, dtype=np.float64)
    return (float(mapped[0]), float(mapped[1]), float(mapped[2]))


def toGlBytes(matrix: np.ndarray) -> bytes:
    """Serialize a column-vector matrix as little-endian float32 column-major bytes."""
    return np.ascontiguousarray(matrix.T, dtype="f4").tobytes()


def perspectiveMatrix(aspect: float) -> np.ndarray:
    """Build a right-handed perspective projection with the fixed field of view."""
    focal = 1.0 / math.tan(math.radians(FIELD_OF_VIEW_DEGREES) / 2.0)
    return np.array(
        [
            [focal / max(aspect, 1e-6), 0.0, 0.0, 0.0],
            [0.0, focal, 0.0, 0.0],
            [
                0.0,
                0.0,
                (FAR_PLANE + NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE),
                (2.0 * FAR_PLANE * NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE),
            ],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )


def lookAtMatrix(eye: np.ndarray) -> np.ndarray:
    """Build a view matrix that looks from `eye` at the origin with +y up."""
    forward = -eye / np.linalg.norm(eye)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    side = np.cross(forward, up)
    side /= np.linalg.norm(side)
    upward = np.cross(side, forward)

    return np.array(
        [
            [side[0], side[1], side[2], -float(side @ eye)],
            [upward[0], upward[1], upward[2], -float(upward @ eye)],
            [-forward[0], -forward[1], -forward[2], float(forward @ eye)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class Camera:
    """Orbit camera expressed as azimuth, elevation, and distance from the origin."""

    def __init__(self, controls: Controls) -> None:
        """Start on the +OpenGL-Z axis at the configured default distance."""
        self.azimuth = 0.0
        self.elevation = 0.0
        self.distance = controls.default_distance
        self.aspect = WINDOW_ASPECT
        self.pan_sensitivity = controls.pan_sensitivity
        self.scroll_sensitivity = controls.scroll_sensitivity
        self.min_distance = controls.min_distance
        self.max_distance = controls.max_distance

    def applyControls(self, controls: Controls) -> None:
        """Adopt live sensitivities and limits, clamping the current distance."""
        self.pan_sensitivity = controls.pan_sensitivity
        self.scroll_sensitivity = controls.scroll_sensitivity
        self.min_distance = controls.min_distance
        self.max_distance = controls.max_distance
        self.distance = _clamp(self.distance, self.min_distance, self.max_distance)

    def reset(self, controls: Controls) -> None:
        """Return to the default orientation and distance."""
        self.azimuth = 0.0
        self.elevation = 0.0
        self.distance = _clamp(controls.default_distance, controls.min_distance, controls.max_distance)

    def orbit(self, dx: float, dy: float) -> None:
        """Orbit by a pixel delta, keeping elevation away from the poles."""
        self.azimuth -= dx * self.pan_sensitivity
        self.elevation += dy * self.pan_sensitivity
        self.elevation = _clamp(self.elevation, -MAX_ELEVATION, MAX_ELEVATION)

    def zoom(self, notches: float) -> None:
        """Scale the distance exponentially per wheel notch, then clamp it."""
        self.distance *= math.exp(-notches * self.scroll_sensitivity)
        self.distance = _clamp(self.distance, self.min_distance, self.max_distance)

    def setAspect(self, aspect: float) -> None:
        """Store the current viewport aspect ratio."""
        self.aspect = max(aspect, 1e-6)

    def getEye(self) -> np.ndarray:
        """Return the camera position in OpenGL space."""
        cos_elevation = math.cos(self.elevation)
        return self.distance * np.array(
            [
                cos_elevation * math.sin(self.azimuth),
                math.sin(self.elevation),
                cos_elevation * math.cos(self.azimuth),
            ],
            dtype=np.float64,
        )

    def getViewMatrix(self) -> np.ndarray:
        """Return the world-to-view matrix."""
        return lookAtMatrix(self.getEye())

    def getProjectionMatrix(self) -> np.ndarray:
        """Return the projection matrix for the stored aspect ratio."""
        return perspectiveMatrix(self.aspect)

    def getViewProjectionMatrix(self) -> np.ndarray:
        """Return projection times view, ready for the vertex shader."""
        return self.getProjectionMatrix() @ self.getViewMatrix()


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a number into an inclusive range."""
    return max(lower, min(upper, value))
