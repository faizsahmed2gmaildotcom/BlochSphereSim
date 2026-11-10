"""Time-sampled position histories for the Bloch-vector trail."""

from __future__ import annotations

import numpy as np

from .bloch_state import BlochState
from .camera import blochToOpenGl
from .config import BlochVisuals, computeTrailCapacity

MAX_SAMPLES_PER_ADVANCE = 4096
SAMPLE_EPSILON = 1.0e-12
TRAIL_SAMPLE_WIDTH = 4


def computeAlpha(age: float, solid: float, fade: float) -> float:
    """Return the trail opacity for a sample of the given age in simulated seconds."""
    if age < 0.0:
        return 0.0
    if age <= solid:
        return 1.0
    if fade <= 0.0:
        return 0.0
    if age >= solid + fade:
        return 0.0
    return (solid + fade - age) / fade


class Trail:
    """Fixed-capacity chronological ring of position/birth-time samples."""

    def __init__(self, capacity: int) -> None:
        """Allocate an empty ring with room for `capacity` samples."""
        if capacity < 1:
            raise ValueError("trail capacity must be at least 1")

        self._samples = np.zeros((capacity, TRAIL_SAMPLE_WIDTH), dtype="f4")
        self._capacity = capacity
        self._head = 0
        self._count = 0
        self._revision = 0
        self._generation = 0

    @property
    def capacity(self) -> int:
        """Return the maximum number of retained samples."""
        return self._capacity

    @property
    def count(self) -> int:
        """Return the number of samples currently stored."""
        return self._count

    @property
    def revision(self) -> int:
        """Return how many samples have ever been recorded."""
        return self._revision

    @property
    def generation(self) -> int:
        """Return a counter bumped whenever the stored history is cleared or resized."""
        return self._generation

    def record(self, vector: tuple[float, float, float], born: float) -> None:
        """Convert a Bloch-space sample to OpenGL and append it, overwriting the oldest when full."""
        self._samples[self._head] = (*blochToOpenGl(vector), born)
        self._head = (self._head + 1) % self._capacity
        self._count = min(self._count + 1, self._capacity)
        self._revision += 1

    def resize(self, capacity: int) -> None:
        """Change the capacity while keeping the newest samples in order."""
        if capacity < 1:
            raise ValueError("trail capacity must be at least one")
        if capacity == self._capacity:
            return

        rows = self.getChronological()[-capacity:]
        samples = np.zeros((capacity, TRAIL_SAMPLE_WIDTH), dtype="f4")
        samples[:len(rows)] = rows

        self._samples = samples
        self._capacity = capacity
        self._count = len(rows)
        self._head = len(rows) % capacity
        self._generation += 1

    def clear(self) -> None:
        """Discard every sample while keeping the allocation."""
        self._head = 0
        self._count = 0
        self._generation += 1

    def getNewest(self) -> tuple[tuple[float, float, float], float] | None:
        """Return (in OpenGL space) the most recent position and its birth time, or None when empty."""
        if self._count == 0:
            return None

        row = self._samples[(self._head - 1) % self._capacity]
        return (float(row[0]), float(row[1]), float(row[2])), float(row[3])

    def getSampleStart(self, sample_count: int) -> int:
        """Return the ring index of the oldest of the newest `sample_count` samples."""
        return (self._head - sample_count) % self._capacity

    def getChronological(self) -> np.ndarray:
        """Return all stored samples oldest-first as an `(n, 4)` float32 copy."""
        if self._count == 0:
            return np.empty((0, TRAIL_SAMPLE_WIDTH), dtype="f4")

        indices = np.arange(self.getSampleStart(self._count), self.getSampleStart(self._count) + self._count)
        return self._samples[indices % self._capacity].copy()

    def getSamplesSince(self, revision: int) -> np.ndarray:
        """Return the samples recorded after `revision`, oldest-first, up to capacity."""
        available = min(self._revision - revision, self._count)
        if available <= 0:
            return np.empty((0, TRAIL_SAMPLE_WIDTH), dtype="f4")

        indices = np.arange(self.getSampleStart(available), self.getSampleStart(available) + available)
        return self._samples[indices % self._capacity].copy()


class TrailRecorder:
    """Advance the physics on a fixed simulated-time grid and record both frame histories."""

    def __init__(self, visuals: BlochVisuals) -> None:
        """Store the trail settings and allocate both rings when the trail is enabled."""
        self._solid = visuals.trail_solid_duration
        self._fading = visuals.trail_fading_duration
        self._interval = visuals.trail_sample_interval
        self._elapsed = 0.0
        self._now = 0.0
        self._lab: Trail | None = None
        self._rotating: Trail | None = None

        self._allocate(computeTrailCapacity(visuals))

    @property
    def enabled(self) -> bool:
        """Return whether trail sampling and drawing are active."""
        return self._lab is not None and self._rotating is not None

    @property
    def lab(self) -> Trail | None:
        """Return the lab-frame history, or None when the trail is disabled."""
        return self._lab

    @property
    def rotating(self) -> Trail | None:
        """Return the rotating-frame history, or None when the trail is disabled."""
        return self._rotating

    @property
    def now(self) -> float:
        """Return the simulated time of the last advance, used for fading."""
        return self._now

    @property
    def solid_duration(self) -> float:
        """Return the fully opaque head duration in simulated seconds."""
        return self._solid

    @property
    def fading_duration(self) -> float:
        """Return the linear fade duration in simulated seconds."""
        return self._fading

    def advance(self, state: BlochState, dt_sim: float) -> None:
        """Advance the state and record a sample on every grid boundary it crosses."""
        if not self.enabled:
            state.advanceTime(dt_sim)
            self._now = state.lab_time
            return

        remaining = dt_sim
        recorded = 0
        while remaining > SAMPLE_EPSILON and recorded < MAX_SAMPLES_PER_ADVANCE:
            to_boundary = self._interval - self._elapsed
            if remaining + SAMPLE_EPSILON >= to_boundary:
                state.advanceTime(to_boundary)
                remaining -= to_boundary
                self._elapsed = 0.0
                self._record(state)
                recorded += 1
            else:
                state.advanceTime(remaining)
                self._elapsed += remaining
                remaining = 0.0

        if remaining > 0.0:
            state.advanceTime(remaining)
            self._elapsed = (self._elapsed + remaining) % self._interval

        self._now = state.lab_time

    def applyVisuals(self, visuals: BlochVisuals) -> None:
        """Adopt new durations and sampling cadence, keeping the newest history."""
        if visuals.trail_sample_interval != self._interval:
            self._elapsed = 0.0

        self._solid = visuals.trail_solid_duration
        self._fading = visuals.trail_fading_duration
        self._interval = visuals.trail_sample_interval
        self._allocate(computeTrailCapacity(visuals))

    def clear(self) -> None:
        """Discard both histories and restart the sampling grid."""
        if self._lab is not None:
            self._lab.clear()
        if self._rotating is not None:
            self._rotating.clear()

        self._elapsed = 0.0
        self._now = 0.0

    def _allocate(self, capacity: int) -> None:
        """Create or resize both rings to the requested capacity, or disable them."""
        if capacity <= 0:
            self._lab = None
            self._rotating = None
            self._elapsed = 0.0
            return

        if self._lab is None or self._rotating is None:
            self._lab = Trail(capacity)
            self._rotating = Trail(capacity)
            self._elapsed = 0.0
            return

        self._lab.resize(capacity)
        self._rotating.resize(capacity)

    def _record(self, state: BlochState) -> None:
        """Capture both reference frames at the current simulated instant."""
        born = state.lab_time
        if self._lab is not None:
            self._lab.record(state.getVector(), born)
        if self._rotating is not None:
            self._rotating.record(state.getDisplayVector(True), born)
