"""Tests for the orbit camera, projection, and matrix upload."""

from __future__ import annotations

import math
import unittest

import numpy as np

from blochspheresim.camera import (
    MAX_ELEVATION,
    Camera,
    blochToOpenGl,
    toGlBytes,
)
from blochspheresim.config import Controls
from blochspheresim.labels import projectPoint


class ConventionTest(unittest.TestCase):
    """Bloch to OpenGL mapping and matrix serialization follow the documented convention."""

    def testAxisMapping(self) -> None:
        """Map each Bloch axis onto its documented OpenGL axis."""
        for source, expected in (
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
            ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        ):
            with self.subTest(source=source):
                mapped = blochToOpenGl(source)
                for actual, target in zip(mapped, expected):
                    self.assertAlmostEqual(actual, target, places=12)

    def testTranslationLandsInFourthGlslColumn(self) -> None:
        """Put translations in the fourth GLSL column after upload."""
        matrix = np.eye(4)
        matrix[:3, 3] = (1.0, 2.0, 3.0)
        floats = np.frombuffer(toGlBytes(matrix), dtype="f4")
        self.assertEqual(floats[12], 1.0)
        self.assertEqual(floats[13], 2.0)
        self.assertEqual(floats[14], 3.0)
        self.assertEqual(floats[15], 1.0)


class CameraTest(unittest.TestCase):
    """Orbit and zoom respect their limits and project correctly."""

    def setUp(self) -> None:
        """Create a camera with default controls."""
        self.camera = Camera(Controls())

    def testOrbitClampsElevation(self) -> None:
        """Keep the camera off the poles."""
        self.camera.orbit(0.0, 1.0e9)
        self.assertAlmostEqual(self.camera.elevation, MAX_ELEVATION, places=12)

        self.camera.orbit(0.0, -1.0e9)
        self.assertAlmostEqual(self.camera.elevation, -MAX_ELEVATION, places=12)

    def testOrbitChangesAzimuth(self) -> None:
        """Update azimuth while preserving distance."""
        self.camera.orbit(100.0, 0.0)
        self.assertAlmostEqual(self.camera.azimuth, -0.5, places=12)
        self.assertAlmostEqual(self.camera.distance, 3.0, places=12)

    def testZoomClampsDistance(self) -> None:
        """Clamp zoom to the configured distance range."""
        self.camera.zoom(1000.0)
        self.assertAlmostEqual(self.camera.distance, 1.5, places=12)

        self.camera.zoom(-1000.0)
        self.assertAlmostEqual(self.camera.distance, 10.0, places=12)

    def testEyeStaysAtConstantDistance(self) -> None:
        """Keep the eye at the stored orbit distance."""
        self.camera.orbit(200.0, 120.0)
        self.assertAlmostEqual(float(np.linalg.norm(self.camera.getEye())), 3.0, places=12)

    def testOriginProjectsToScreenCentre(self) -> None:
        """Project the origin to the middle of the viewport."""
        self.camera.setAspect(1280.0 / 800.0)
        point = projectPoint(
            (0.0, 0.0, 0.0),
            self.camera.getViewProjectionMatrix(),
            1280,
            800,
        )
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point[0], 640.0, places=6)
        self.assertAlmostEqual(point[1], 400.0, places=6)

    def testPointBehindCameraIsRejected(self) -> None:
        """Reject points behind the camera."""
        point = projectPoint(
            (0.0, 0.0, 10.0),
            self.camera.getViewProjectionMatrix(),
            1280,
            800,
        )
        self.assertIsNone(point)

    def testApplyControlsClampsDistance(self) -> None:
        """Clamp the distance when new limits arrive."""
        controls = Controls(min_distance=1.5, max_distance=5.0, default_distance=3.0)
        self.camera.distance = 9.0
        self.camera.applyControls(controls)
        self.assertAlmostEqual(self.camera.distance, 5.0, places=12)

    def testResetRestoresDefaultDistance(self) -> None:
        """Restore orientation and distance on reset."""
        controls = Controls()
        self.camera.orbit(50.0, 20.0)
        self.camera.zoom(-4.0)
        self.camera.reset(controls)
        self.assertEqual(self.camera.azimuth, 0.0)
        self.assertEqual(self.camera.elevation, 0.0)
        self.assertAlmostEqual(self.camera.distance, controls.default_distance, places=12)

    def testElevationClampIsBelowPole(self) -> None:
        """Keep the elevation clamp below the pole."""
        self.assertLess(MAX_ELEVATION, math.pi / 2.0)


if __name__ == "__main__":
    unittest.main()
