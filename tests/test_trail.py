"""Tests for the ring-buffer trail, fade function, and sampling grid."""

from __future__ import annotations

import math
import unittest

import numpy as np

from blochspheresim.bloch_state import BlochState
from blochspheresim.config import BlochVisuals
from blochspheresim.trail import Trail, TrailRecorder, computeAlpha


def _visuals(
    solid: float = 1.0,
    fading: float = 1.0,
    interval: float = 0.1,
) -> BlochVisuals:
    """Return trail visuals with explicit durations and cadence."""
    return BlochVisuals(
        trail_solid_duration=solid,
        trail_fading_duration=fading,
        trail_sample_interval=interval,
    )


def _angleDegrees(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    """Return the angle between two unit vectors in degrees."""
    dot = sum(a * b for a, b in zip(first, second))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


class FadeTest(unittest.TestCase):
    """The fade is opaque through the head, linear through the tail, then zero."""

    def testSolidHead(self) -> None:
        """Stay fully opaque inside the solid duration."""
        self.assertEqual(computeAlpha(0.0, 1.0, 2.0), 1.0)
        self.assertEqual(computeAlpha(1.0, 1.0, 2.0), 1.0)

    def testLinearFade(self) -> None:
        """Fade linearly across the tail."""
        self.assertAlmostEqual(computeAlpha(2.0, 1.0, 2.0), 0.5, places=12)
        self.assertAlmostEqual(computeAlpha(1.5, 1.0, 1.0), 0.5, places=12)

    def testPastLifetimeIsZero(self) -> None:
        """Hide samples older than the total duration."""
        self.assertEqual(computeAlpha(3.0, 1.0, 2.0), 0.0)
        self.assertEqual(computeAlpha(10.0, 1.0, 2.0), 0.0)

    def testZeroFadeCutsOff(self) -> None:
        """Use a hard cutoff when the fade duration is zero."""
        self.assertEqual(computeAlpha(1.0, 1.0, 0.0), 1.0)
        self.assertEqual(computeAlpha(1.0001, 1.0, 0.0), 0.0)

    def testNegativeAgeIsHidden(self) -> None:
        """Hide samples dated in the future."""
        self.assertEqual(computeAlpha(-0.1, 1.0, 2.0), 0.0)


class TrailRingTest(unittest.TestCase):
    """The ring keeps the newest samples in chronological order."""

    def testWrapKeepsNewest(self) -> None:
        """Overwrite the oldest samples once capacity is reached."""
        trail = Trail(3)
        for index in range(5):
            trail.record((float(index), 0.0, 0.0), float(index))

        rows = trail.getChronological()
        self.assertEqual(trail.count, 3)
        self.assertEqual([row[0] for row in rows], [2.0, 3.0, 4.0])
        self.assertEqual(trail.getNewest(), ((4.0, 0.0, 0.0), 4.0))

    def testSamplesSinceRevision(self) -> None:
        """Return only samples recorded after a revision marker."""
        trail = Trail(4)
        for index in range(3):
            trail.record((float(index), 0.0, 0.0), float(index))

        rows = trail.getSamplesSince(2)
        self.assertEqual([row[0] for row in rows], [2.0])

    def testResizeKeepsNewest(self) -> None:
        """Shrink and grow without losing the newest samples or their order."""
        trail = Trail(5)
        for index in range(5):
            trail.record((float(index), 0.0, 0.0), float(index))

        trail.resize(3)
        self.assertEqual([row[0] for row in trail.getChronological()], [2.0, 3.0, 4.0])

        trail.resize(6)
        trail.record((5.0, 0.0, 0.0), 5.0)
        trail.record((6.0, 0.0, 0.0), 6.0)
        self.assertEqual(
            [row[0] for row in trail.getChronological()],
            [2.0, 3.0, 4.0, 5.0, 6.0],
        )

    def testClearBumpsGeneration(self) -> None:
        """Empty the ring and signal a full re-upload."""
        trail = Trail(4)
        trail.record((1.0, 2.0, 3.0), 0.5)
        generation = trail.generation

        trail.clear()

        self.assertEqual(trail.count, 0)
        self.assertIsNone(trail.getNewest())
        self.assertEqual(trail.generation, generation + 1)

    def testGetNewestOnEmpty(self) -> None:
        """Return None before anything is recorded."""
        self.assertIsNone(Trail(2).getNewest())

    def testZeroCapacityRejected(self) -> None:
        """Refuse an unusable ring."""
        with self.assertRaises(ValueError):
            Trail(0)


class TrailRecorderTest(unittest.TestCase):
    """Sampling happens on a fixed simulated-time grid for both frames."""

    def testOneLargeStepSamplesEveryBoundary(self) -> None:
        """Record every grid boundary crossed by a single large step."""
        recorder = TrailRecorder(_visuals())
        state = BlochState()
        recorder.advance(state, 0.35)

        self.assertEqual(recorder.lab.count, 3)
        born = recorder.lab.getChronological()[:, 3]
        for actual, expected in zip(born, (0.1, 0.2, 0.3)):
            self.assertAlmostEqual(float(actual), expected, places=5)

    def testManySmallStepsMatchOneLargeStep(self) -> None:
        """Produce the same grid regardless of how the time is chunked."""
        recorder = TrailRecorder(_visuals())
        state = BlochState()
        for _ in range(35):
            recorder.advance(state, 0.01)

        self.assertEqual(recorder.lab.count, 3)
        born = recorder.lab.getChronological()[:, 3]
        for actual, expected in zip(born, (0.1, 0.2, 0.3)):
            self.assertAlmostEqual(float(actual), expected, places=5)

    def testChunkingPreservesPhysics(self) -> None:
        """Advance through the recorder without changing the final state."""
        recorder = TrailRecorder(_visuals(interval=0.05))
        recorded = BlochState()
        recorder.advance(recorded, 1.0)

        direct = BlochState()
        direct.advanceTime(1.0)

        self.assertLess(_angleDegrees(recorded.getVector(), direct.getVector()), 1e-7)

    def testFullModelChunkingStaysClose(self) -> None:
        """Keep full-model accuracy close when sampling splits a frame."""
        visuals = _visuals(interval=0.1)
        recorder = TrailRecorder(visuals)
        recorded = BlochState(rwa=False)
        recorder.advance(recorded, 0.2)

        direct = BlochState(rwa=False)
        direct.advanceTime(0.2)

        self.assertLess(_angleDegrees(recorded.getVector(), direct.getVector()), 1e-4)

    def testBothFramesShareBirthTimes(self) -> None:
        """Record lab and rotating histories at identical instants."""
        recorder = TrailRecorder(_visuals())
        state = BlochState()
        recorder.advance(state, 0.35)

        lab = recorder.lab.getChronological()
        rotating = recorder.rotating.getChronological()
        np.testing.assert_allclose(lab[:, 3], rotating[:, 3], atol=1e-6)

    def testRotatingSamplesAreCaptureTimeProjections(self) -> None:
        """Rotate each lab sample by the frame phase it was captured at."""
        recorder = TrailRecorder(_visuals())
        state = BlochState()
        recorder.advance(state, 0.35)

        lab = recorder.lab.getChronological()
        rotating = recorder.rotating.getChronological()
        for lab_row, rotating_row in zip(lab, rotating):
            phase = 3.0 * float(lab_row[3])
            cosine = math.cos(phase)
            sine = math.sin(phase)
            expected = np.array(
                [
                    cosine * lab_row[0] - sine * lab_row[1],
                    sine * lab_row[0] + cosine * lab_row[1],
                    lab_row[2],
                ]
            )
            np.testing.assert_allclose(rotating_row[:3], expected, atol=1e-5)

    def testDisabledTrailStillAdvances(self) -> None:
        """Advance physics without allocating or recording when disabled."""
        recorder = TrailRecorder(_visuals(solid=0.0, fading=0.0))
        state = BlochState()
        recorder.advance(state, 0.5)

        direct = BlochState()
        direct.advanceTime(0.5)

        self.assertFalse(recorder.enabled)
        self.assertIsNone(recorder.lab)
        self.assertLess(_angleDegrees(state.getVector(), direct.getVector()), 1e-9)

    def testIntervalChangeKeepsHistoryAndResetsGrid(self) -> None:
        """Keep existing samples but start the new cadence immediately."""
        recorder = TrailRecorder(_visuals(interval=0.1))
        state = BlochState()
        recorder.advance(state, 0.25)
        before = recorder.lab.count

        recorder.applyVisuals(_visuals(interval=0.2))
        self.assertEqual(recorder.lab.count, before)

        recorder.advance(state, 0.2)
        self.assertEqual(recorder.lab.count, before + 1)
        self.assertAlmostEqual(
            float(recorder.lab.getNewest()[1]),
            state.lab_time,
            places=5,
        )

    def testClearEmptiesBothFrames(self) -> None:
        """Clear both histories and reset the clock and grid."""
        recorder = TrailRecorder(_visuals())
        state = BlochState()
        recorder.advance(state, 0.35)

        recorder.clear()

        self.assertEqual(recorder.lab.count, 0)
        self.assertEqual(recorder.rotating.count, 0)
        self.assertEqual(recorder.now, 0.0)

    def testNowTracksSimulatedTime(self) -> None:
        """Expose the current simulated time for fading."""
        recorder = TrailRecorder(_visuals())
        state = BlochState()
        recorder.advance(state, 0.42)
        self.assertAlmostEqual(recorder.now, state.lab_time, places=9)


if __name__ == "__main__":
    unittest.main()
