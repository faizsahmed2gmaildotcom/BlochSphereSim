"""Tests for configuration loading, validation, and hot reload."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from blochspheresim.config import (
    BlochVisuals,
    ConfigError,
    ConfigWatcher,
    buildConfig,
    collectPhysicsWarnings,
    loadConfig,
)


class ConfigDefaultsTest(unittest.TestCase):
    """Defaults apply when files or keys are missing."""

    def testMissingFileUsesDefaults(self) -> None:
        """Fall back to defaults when the file does not exist."""
        with tempfile.TemporaryDirectory() as directory:
            config = loadConfig(Path(directory) / "config.yaml")
        self.assertEqual(config.bloch_visuals, BlochVisuals())
        self.assertEqual(config.physics.gamma, 1.0)
        self.assertTrue(config.physics.rwa)

    def testMissingKeysUseDefaults(self) -> None:
        """Fill in defaults for absent keys."""
        config = buildConfig({"physics": {"b_z": 5.0}})
        self.assertEqual(config.physics.b_z, 5.0)
        self.assertEqual(config.physics.b_1, 0.3)
        self.assertEqual(config.bloch_visuals.latitude_count, 7)
        self.assertEqual(config.controls.default_distance, 3.0)


class ConfigValidationTest(unittest.TestCase):
    """Out-of-range numbers snap or clamp; malformed values raise."""

    def testCountsSnap(self) -> None:
        """Snap latitude to odd and longitude to even."""
        config = buildConfig({"bloch_visuals": {"latitude_count": 8, "longitude_count": 7}})
        self.assertEqual(config.bloch_visuals.latitude_count, 7)
        self.assertEqual(config.bloch_visuals.longitude_count, 6)

        config = buildConfig({"bloch_visuals": {"latitude_count": 0, "longitude_count": 1}})
        self.assertEqual(config.bloch_visuals.latitude_count, 1)
        self.assertEqual(config.bloch_visuals.longitude_count, 2)

    def testVisualClamps(self) -> None:
        """Clamp visual numbers into valid ranges."""
        config = buildConfig(
            {
                "bloch_visuals": {
                    "timespeed": 2.0,
                    "font_size": 0,
                    "label_offset": -1.0,
                    "max_frame_rate": 0,
                    "latitude_opacity": 3.0,
                }
            }
        )
        visuals = config.bloch_visuals
        self.assertEqual(visuals.timespeed, 1.0)
        self.assertEqual(visuals.font_size, 8)
        self.assertEqual(visuals.label_offset, 0.0)
        self.assertEqual(visuals.max_frame_rate, 1)
        self.assertEqual(visuals.latitude_opacity, 1.0)

        config = buildConfig({"bloch_visuals": {"timespeed": -1.0}})
        self.assertEqual(config.bloch_visuals.timespeed, 0.0)

    def testDefaultDistanceClampedIntoRange(self) -> None:
        """Pull the default distance into the allowed range."""
        config = buildConfig({"controls": {"default_distance": 50.0}})
        self.assertEqual(config.controls.default_distance, config.controls.max_distance)

    def testInvalidDistanceRangeRaises(self) -> None:
        """Reject impossible distance limits."""
        with self.assertRaises(ConfigError):
            buildConfig({"controls": {"min_distance": 5.0, "max_distance": 2.0}})
        with self.assertRaises(ConfigError):
            buildConfig({"controls": {"min_distance": 0.0}})

    def testIntegrationStepsFloor(self) -> None:
        """Enforce the minimum integration step count."""
        config = buildConfig({"physics": {"integration_steps_per_period": 4}})
        self.assertEqual(config.physics.integration_steps_per_period, 16)

    def testMalformedValuesRaise(self) -> None:
        """Reject wrong types and non-finite numbers."""
        malformed = (
            {"physics": {"b_z": "three"}},
            {"physics": {"rwa": 1}},
            {"physics": {"rwa": "yes"}},
            {"bloch_visuals": {"font_size": 12.5}},
            {"bloch_visuals": "wide"},
            {"controls": {"pan_sensitivity": -0.1}},
            {"physics": {"gamma": float("nan")}},
        )
        for mapping in malformed:
            with self.subTest(mapping=mapping), self.assertRaises(ConfigError):
                buildConfig(mapping)

    def testBadYamlRaises(self) -> None:
        """Report invalid YAML as a config error."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("physics: [unclosed\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                loadConfig(path)


class ConfigWatcherTest(unittest.TestCase):
    """The watcher reloads only on mtime changes and survives bad edits."""

    def testDetectsMtimeChange(self) -> None:
        """Reload after the file mtime changes."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("physics:\n  b_z: 1.0\n", encoding="utf-8")
            watcher = ConfigWatcher(path, poll_interval=0.0)

            self.assertIsNone(watcher.poll(0.0))

            self._touchWithNewContent(path, "physics:\n  b_z: 2.0\n")
            config = watcher.poll(0.0)
            self.assertIsNotNone(config)
            self.assertEqual(config.physics.b_z, 2.0)

    def testBadEditKeepsLastConfig(self) -> None:
        """Keep the previous config when a reload fails."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("physics:\n  b_z: 1.0\n", encoding="utf-8")
            watcher = ConfigWatcher(path, poll_interval=0.0)

            self._touchWithNewContent(path, "physics: [unclosed\n")
            self.assertIsNone(watcher.poll(0.0))

    def _touchWithNewContent(self, path: Path, content: str) -> None:
        """Write content and force a strictly newer mtime."""
        path.write_text(content, encoding="utf-8")
        stamp = path.stat().st_mtime_ns + 1_000_000_000
        os.utime(path, ns=(stamp, stamp))


class PhysicsWarningTest(unittest.TestCase):
    """Warnings flag questionable physics settings."""

    def testStrongDriveWarns(self) -> None:
        """Warn when the drive breaks the rotating-wave assumption."""
        config = buildConfig({"physics": {"b_z": 1.0, "b_1": 0.5, "rwa": True}})
        self.assertTrue(collectPhysicsWarnings(config.physics))

    def testZeroCarrierWarns(self) -> None:
        """Warn about the zero-carrier rotating-wave case."""
        config = buildConfig({"physics": {"b_z": 0.0, "rwa": True}})
        self.assertTrue(collectPhysicsWarnings(config.physics))

    def testDefaultsAreQuiet(self) -> None:
        """Stay silent for the default settings."""
        self.assertEqual(collectPhysicsWarnings(buildConfig({}).physics), [])

    def testSlowFullModelIsQuiet(self) -> None:
        """Stay silent when the full model is fast enough."""
        config = buildConfig({"physics": {"rwa": False}})
        self.assertEqual(collectPhysicsWarnings(config.physics), [])

    def testFastFullModelWarns(self) -> None:
        """Warn when the full model cannot keep up."""
        config = buildConfig({"physics": {"rwa": False, "b_z": 1.0e5}})
        self.assertTrue(collectPhysicsWarnings(config.physics))


if __name__ == "__main__":
    unittest.main()
