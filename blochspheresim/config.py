"""Typed, validated configuration for the Bloch sphere simulator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")
MIN_FONT_SIZE = 8
MIN_INTEGRATION_STEPS = 16
MAX_ESTIMATED_STEP_RATE = 250_000.0
MIN_TRAIL_SAMPLE_INTERVAL = 1.0e-4
MAX_TRAIL_POINTS = 100_000


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or physically invalid."""


@dataclass(frozen=True)
class BlochVisuals:
    """Numeric knobs for the sphere, labels, timing, and frame rate."""

    latitude_count: int = 7
    longitude_count: int = 8
    latitude_opacity: float = 0.25
    longitude_opacity: float = 0.55
    label_offset: float = 0.05
    font_size: int = 20
    timespeed: float = 1.0
    max_frame_rate: int = 60
    trail_solid_duration: float = 0.5
    trail_fading_duration: float = 1.5
    trail_sample_interval: float = 0.016667


@dataclass(frozen=True)
class Controls:
    """Camera sensitivities and distance limits."""

    pan_sensitivity: float = 0.005
    scroll_sensitivity: float = 0.1
    default_distance: float = 3.0
    min_distance: float = 1.5
    max_distance: float = 10.0


@dataclass(frozen=True)
class Physics:
    """Magnetic field and integration settings."""

    gamma: float = 1.0
    b_z: float = 3.0
    b_1: float = 0.3
    rf_phase: float = 0.0
    rwa: bool = True
    integration_steps_per_period: int = 512


@dataclass(frozen=True)
class Config:
    """Complete validated configuration."""

    bloch_visuals: BlochVisuals = BlochVisuals()
    controls: Controls = Controls()
    physics: Physics = Physics()


def buildConfig(raw: dict[str, Any]) -> Config:
    """Validate a raw YAML mapping into frozen configuration dataclasses."""
    visuals = _buildVisuals(_readSection(raw, "bloch_visuals"))
    controls = _buildControls(_readSection(raw, "controls"))
    physics = _buildPhysics(_readSection(raw, "physics"))
    return Config(bloch_visuals=visuals, controls=controls, physics=physics)


def loadConfig(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate a YAML config file, using defaults when it is absent."""
    path = Path(path)
    if not path.exists():
        return buildConfig({})

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"could not read {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {path}: {error}") from error

    if raw is None:
        return buildConfig({})
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return buildConfig(raw)


def collectPhysicsWarnings(physics: Physics) -> list[str]:
    """Return human-readable warnings about questionable field or step settings."""
    warnings: list[str] = []

    if physics.rwa and physics.b_z == 0.0:
        warnings.append(
            "physics.rwa is on with b_z = 0; using the exact static-field rotation instead"
        )
    elif physics.rwa and abs(physics.b_1) > 0.25 * abs(physics.b_z):
        warnings.append(
            "physics.b_1 exceeds 0.25 * |b_z|; rotating-wave accuracy is degraded"
        )

    if not physics.rwa:
        step_rate = _estimateStepRate(physics)
        if step_rate > MAX_ESTIMATED_STEP_RATE:
            warnings.append(
                f"full-model integration needs about {step_rate:,.0f} steps per simulated "
                "second; the simulation may not keep up"
            )

    return warnings


def collectTrailWarnings(visuals: BlochVisuals) -> list[str]:
    """Return a warning when the requested trail was clamped to the point limit."""
    if computeTrailCapacity(visuals) < MAX_TRAIL_POINTS:
        return []
    return [
        f"trail capacity limited to {MAX_TRAIL_POINTS:,} points; "
        f"using a sample interval of {visuals.trail_sample_interval:.6f} simulated seconds"
    ]


def computeTrailCapacity(visuals: BlochVisuals) -> int:
    """Return the ring capacity for one trail, or zero when the trail is disabled."""
    total = visuals.trail_solid_duration + visuals.trail_fading_duration
    if total <= 0.0:
        return 0
    return min(MAX_TRAIL_POINTS, math.ceil(total / visuals.trail_sample_interval) + 2)


class ConfigWatcher:
    """Reload a config file when its modification time changes."""

    def __init__(self, path: Path = DEFAULT_CONFIG_PATH, poll_interval: float = 0.5) -> None:
        """Store the path and remember the mtime of the currently loaded file."""
        self._path = Path(path)
        self._poll_interval = poll_interval
        self._last_mtime = self._readMtime()
        self._elapsed = 0.0

    def poll(self, dt_real: float) -> Config | None:
        """Return a reloaded Config at most every poll interval, or None if unchanged."""
        self._elapsed += dt_real
        if self._elapsed < self._poll_interval:
            return None
        self._elapsed = 0.0

        mtime = self._readMtime()
        if mtime is None or mtime == self._last_mtime:
            return None
        self._last_mtime = mtime

        try:
            return loadConfig(self._path)
        except ConfigError as error:
            print(f"config reload failed: {error}", file=sys.stderr)
            return None

    def _readMtime(self) -> int | None:
        """Return the file's nanosecond mtime, or None when it cannot be read."""
        try:
            return self._path.stat().st_mtime_ns
        except OSError:
            return None


def _buildVisuals(section: dict[str, Any]) -> BlochVisuals:
    """Validate and snap the `bloch_visuals` section."""
    defaults = BlochVisuals()
    latitude_count = _snapOdd(_readInt(section, "latitude_count", defaults.latitude_count))
    longitude_count = _snapEven(_readInt(section, "longitude_count", defaults.longitude_count))

    solid = max(0.0, _readFloat(section, "trail_solid_duration", defaults.trail_solid_duration))
    fading = max(0.0, _readFloat(section, "trail_fading_duration", defaults.trail_fading_duration))
    interval = max(
        MIN_TRAIL_SAMPLE_INTERVAL,
        _readFloat(section, "trail_sample_interval", defaults.trail_sample_interval),
    )
    interval = _clampTrailInterval(solid + fading, interval)

    return BlochVisuals(
        latitude_count=latitude_count,
        longitude_count=longitude_count,
        latitude_opacity=_clamp(_readFloat(section, "latitude_opacity", defaults.latitude_opacity), 0.0, 1.0),
        longitude_opacity=_clamp(_readFloat(section, "longitude_opacity", defaults.longitude_opacity), 0.0, 1.0),
        label_offset=max(0.0, _readFloat(section, "label_offset", defaults.label_offset)),
        font_size=max(MIN_FONT_SIZE, _readInt(section, "font_size", defaults.font_size)),
        timespeed=_clamp(_readFloat(section, "timespeed", defaults.timespeed), 0.0, 1.0),
        max_frame_rate=max(1, _readInt(section, "max_frame_rate", defaults.max_frame_rate)),
        trail_solid_duration=solid,
        trail_fading_duration=fading,
        trail_sample_interval=interval,
    )


def _clampTrailInterval(total: float, interval: float) -> float:
    """Raise the sample interval so one trail never exceeds the point limit."""
    if total <= 0.0:
        return interval
    if math.ceil(total / interval) + 2 <= MAX_TRAIL_POINTS:
        return interval
    return total / (MAX_TRAIL_POINTS - 2)


def _buildControls(section: dict[str, Any]) -> Controls:
    """Validate and clamp the `controls` section."""
    defaults = Controls()
    pan_sensitivity = _readFloat(section, "pan_sensitivity", defaults.pan_sensitivity)
    scroll_sensitivity = _readFloat(section, "scroll_sensitivity", defaults.scroll_sensitivity)
    min_distance = _readFloat(section, "min_distance", defaults.min_distance)
    max_distance = _readFloat(section, "max_distance", defaults.max_distance)

    if pan_sensitivity < 0.0 or scroll_sensitivity < 0.0:
        raise ConfigError("pan_sensitivity and scroll_sensitivity must be non-negative")
    if min_distance <= 0.0:
        raise ConfigError("min_distance must be greater than zero")
    if max_distance <= min_distance:
        raise ConfigError("max_distance must be greater than min_distance")

    default_distance = _readFloat(section, "default_distance", defaults.default_distance)
    return Controls(
        pan_sensitivity=pan_sensitivity,
        scroll_sensitivity=scroll_sensitivity,
        default_distance=_clamp(default_distance, min_distance, max_distance),
        min_distance=min_distance,
        max_distance=max_distance,
    )


def _buildPhysics(section: dict[str, Any]) -> Physics:
    """Validate the `physics` section."""
    defaults = Physics()
    return Physics(
        gamma=_readFloat(section, "gamma", defaults.gamma),
        b_z=_readFloat(section, "b_z", defaults.b_z),
        b_1=_readFloat(section, "b_1", defaults.b_1),
        rf_phase=_readFloat(section, "rf_phase", defaults.rf_phase),
        rwa=_readBool(section, "rwa", defaults.rwa),
        integration_steps_per_period=max(
            MIN_INTEGRATION_STEPS,
            _readInt(section, "integration_steps_per_period", defaults.integration_steps_per_period),
        ),
    )


def _estimateStepRate(physics: Physics) -> float:
    """Estimate RK4 steps per simulated second for the configured fields."""
    angular_rate = abs(physics.gamma) * (abs(physics.b_z) + 2.0 * abs(physics.b_1))
    return physics.integration_steps_per_period * angular_rate / (2.0 * math.pi)


def _readSection(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a nested mapping, treating a missing or null section as empty."""
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping")
    return value


def _readFloat(section: dict[str, Any], key: str, default: float) -> float:
    """Read a finite float from a section, rejecting booleans and strings."""
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{key}' must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ConfigError(f"'{key}' must be finite, got {value!r}")
    return value


def _readInt(section: dict[str, Any], key: str, default: int) -> int:
    """Read an integer from a section, accepting only integral numbers."""
    value = section.get(key, default)
    if isinstance(value, bool):
        raise ConfigError(f"'{key}' must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ConfigError(f"'{key}' must be an integer, got {value!r}")


def _readBool(section: dict[str, Any], key: str, default: bool) -> bool:
    """Read a strict boolean from a section."""
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"'{key}' must be a boolean, got {value!r}")
    return value


def _snapOdd(value: int) -> int:
    """Return the nearest odd integer not below one."""
    return max(1, value if value % 2 else value - 1)


def _snapEven(value: int) -> int:
    """Return the nearest even integer not below two."""
    return max(2, value if value % 2 == 0 else value - 1)


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a number into an inclusive range."""
    return max(lower, min(upper, value))
