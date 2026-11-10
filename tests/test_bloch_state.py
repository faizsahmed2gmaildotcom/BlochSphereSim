"""Tests for the Bloch-vector physics."""

from __future__ import annotations

import math
import unittest

from blochspheresim.bloch_state import BlochState


def _angleDegrees(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    """Return the angle between two unit vectors in degrees."""
    dot = sum(a * b for a, b in zip(first, second))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


class RwaTest(unittest.TestCase):
    """Rotating-wave evolution is exact and dt-independent."""

    def testNormStaysUnit(self) -> None:
        """Keep the vector unit-length through the evolution."""
        state = BlochState(integration_steps_per_period=64)
        for dt in (0.001, 0.37, 2.5, 11.0):
            state.advanceTime(dt)
            self.assertAlmostEqual(_length(state.getVector()), 1.0, places=12)

    def testOneStepEqualsManySteps(self) -> None:
        """Match one large step to many small steps."""
        single = BlochState()
        single.advanceTime(1.234)

        many = BlochState()
        for _ in range(100):
            many.advanceTime(0.01234)

        self.assertLess(_angleDegrees(single.getVector(), many.getVector()), 1e-7)
        self.assertAlmostEqual(single.lab_time, many.lab_time, places=12)

    def testPureLarmorRotatingViewIsStationary(self) -> None:
        """Cancel pure Larmor precession in the rotating view."""
        state = BlochState(b_z=3.0, b_1=0.0)
        state.theta = math.pi / 2.0
        state.phi = 0.0

        state.advanceTime(0.7)

        rotating = state.getDisplayVector(True)
        self.assertLess(_angleDegrees(rotating, (1.0, 0.0, 0.0)), 1e-6)

        expected = (math.cos(-3.0 * 0.7), math.sin(-3.0 * 0.7), 0.0)
        self.assertLess(_angleDegrees(state.getVector(), expected), 1e-6)
        self.assertAlmostEqual(state.frame_phase, 3.0 * 0.7, places=12)

    def testRabiPiPulseFlipsZ(self) -> None:
        """Flip the state after a Rabi pi pulse."""
        state = BlochState(b_z=3.0, b_1=0.3)
        state.advanceTime(math.pi / 0.3)

        self.assertAlmostEqual(state.getVector()[2], -1.0, places=9)

    def testRfPhaseSelectsDriveAxis(self) -> None:
        """Drive about the y' axis when the RF phase is pi/2."""
        state = BlochState(b_z=3.0, b_1=0.3, rf_phase=math.pi / 2.0)
        state.advanceTime((math.pi / 2.0) / 0.3)

        rotating = state.getDisplayVector(True)
        self.assertLess(_angleDegrees(rotating, (-1.0, 0.0, 0.0)), 1e-9)

    def testFramePhaseAccumulatesAcrossFieldChange(self) -> None:
        """Accumulate frame phase across a field edit."""
        state = BlochState()
        state.advanceTime(0.1)
        self.assertAlmostEqual(state.frame_phase, 0.3, places=12)

        state.setFields(1.0, 1.0, 0.3, 0.0, True, 512)
        state.advanceTime(0.1)
        self.assertAlmostEqual(state.frame_phase, 0.4, places=12)

    def testNonPositiveTimeIsNoOp(self) -> None:
        """Ignore zero, negative, and non-finite time steps."""
        state = BlochState()
        initial = state.getVector()
        for dt in (0.0, -1.0, float("nan"), float("inf")):
            state.advanceTime(dt)
        self.assertEqual(state.getVector(), initial)
        self.assertEqual(state.lab_time, 0.0)


class FullModelTest(unittest.TestCase):
    """Full-model integration matches the exact model in its limits."""

    def testNormStaysUnit(self) -> None:
        """Keep the vector unit-length through the evolution."""
        state = BlochState(rwa=False, integration_steps_per_period=64)
        for dt in (0.001, 0.37, 2.5):
            state.advanceTime(dt)
            self.assertAlmostEqual(_length(state.getVector()), 1.0, places=12)

    def testOneStepEqualsManySteps(self) -> None:
        """Match one large step to many small steps."""
        single = BlochState(rwa=False)
        single.advanceTime(0.2)

        many = BlochState(rwa=False)
        for _ in range(200):
            many.advanceTime(0.001)

        self.assertLess(_angleDegrees(single.getVector(), many.getVector()), 1e-6)

    def testConvergence(self) -> None:
        """Converge as the integration step count grows."""
        coarse = BlochState(rwa=False, integration_steps_per_period=128)
        fine = BlochState(rwa=False, integration_steps_per_period=4096)
        coarse.advanceTime(10.0)
        fine.advanceTime(10.0)

        self.assertLess(_angleDegrees(coarse.getVector(), fine.getVector()), 0.01)

    def testStaticFieldLimit(self) -> None:
        """Rotate exactly about a static field when b_z is zero."""
        for rwa in (True, False):
            with self.subTest(rwa=rwa):
                state = BlochState(b_z=0.0, b_1=0.3, rwa=rwa)
                state.advanceTime(math.pi / (2.0 * 0.3))
                self.assertAlmostEqual(state.getVector()[2], -1.0, places=9)

    def testDiffersFromRwaByCounterRotatingTerm(self) -> None:
        """Show the counter-rotating difference shrinking with b_1."""
        full = BlochState(rwa=False)
        approximate = BlochState(rwa=True)
        full.advanceTime(10.0)
        approximate.advanceTime(10.0)

        difference = _angleDegrees(full.getVector(), approximate.getVector())
        self.assertGreater(difference, 1.0)
        self.assertLess(difference, 10.0)

        weak_full = BlochState(b_1=0.03, rwa=False)
        weak_approximate = BlochState(b_1=0.03, rwa=True)
        weak_full.advanceTime(10.0)
        weak_approximate.advanceTime(10.0)
        weak_difference = _angleDegrees(weak_full.getVector(), weak_approximate.getVector())
        self.assertLess(weak_difference, difference)

    def testZeroFieldIsNoOp(self) -> None:
        """Do nothing when both fields vanish."""
        for rwa in (True, False):
            with self.subTest(rwa=rwa):
                state = BlochState(b_z=0.0, b_1=0.0, rwa=rwa)
                state.theta = math.pi / 3.0
                state.phi = 0.4
                state.advanceTime(2.0)
                self.assertAlmostEqual(state.theta, math.pi / 3.0, places=12)
                self.assertAlmostEqual(state.phi, 0.4, places=12)


def _length(vector: tuple[float, float, float]) -> float:
    """Return the Euclidean length of a 3-vector."""
    return math.sqrt(sum(component * component for component in vector))


if __name__ == "__main__":
    unittest.main()
