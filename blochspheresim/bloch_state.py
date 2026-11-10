"""Bloch-vector evolution under static and radio-frequency magnetic fields."""

from __future__ import annotations

import math


class BlochState:
    """Single-qubit state stored as lab-frame theta/phi with a driven magnetic field.

    The lab field is the linearly polarized drive from the NMR reference:

        B(t) = (2 * b_1 * cos(w_L * t + rf_phase), 0, b_z),   w_L = gamma * b_z

    and the Bloch vector obeys `dv/dt = gamma * v x B(t)`.

    With `rwa` on, the evolution uses the rotating-wave solution: in the frame rotating at the
    Larmor frequency, the drive becomes the static field `b_1 * (cos rf_phase, sin rf_phase, 0)`,
    so the state rotates about that axis at the Rabi frequency `gamma * b_1`. That solution is
    exact for the rotating-wave model and works for any `dt`.

    With `rwa` off, the full time-dependent model is integrated with fourth-order Runge-Kutta.
    It has no elementary closed form (the linearly driven two-level system reduces to a
    Whittaker-Hill problem), so accuracy is bounded by `integration_steps_per_period` rather
    than by `dt` alone.

    A zero `b_z` makes the drive static; both models then use the exact rotation about the
    fixed field, because the rotating-wave factor-of-two cancellation is invalid at zero
    carrier frequency.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        b_z: float = 3.0,
        b_1: float = 0.3,
        rf_phase: float = 0.0,
        rwa: bool = True,
        integration_steps_per_period: int = 512,
    ) -> None:
        """Create the |0> state at rest with the given field settings."""
        self.theta = 0.0
        self.phi = 0.0
        self.gamma = gamma
        self.b_z = b_z
        self.b_1 = b_1
        self.rf_phase = rf_phase
        self.rwa = rwa
        self.integration_steps_per_period = integration_steps_per_period
        self.lab_time = 0.0
        self.frame_phase = 0.0

    def advanceTime(self, dt: float) -> None:
        """Advance the state by `dt` simulated seconds; non-positive or non-finite is a no-op."""
        if not math.isfinite(dt) or dt <= 0.0:
            return

        if self.b_z == 0.0:
            self._advanceStaticField(dt)
        elif self.rwa:
            self._advanceRwa(dt)
        else:
            self._advanceFull(dt)

    def getVector(self) -> tuple[float, float, float]:
        """Return the Bloch vector in the lab frame."""
        sin_theta = math.sin(self.theta)
        return (
            sin_theta * math.cos(self.phi),
            sin_theta * math.sin(self.phi),
            math.cos(self.theta),
        )

    def getDisplayVector(self, rotating: bool) -> tuple[float, float, float]:
        """Return the lab-frame vector in Bloch space, or the vector viewed in the rotating frame."""
        vector = self.getVector()
        if not rotating:
            return vector
        return self._rotateZ(vector, self.frame_phase)

    def setFields(
        self,
        gamma: float,
        b_z: float,
        b_1: float,
        rf_phase: float,
        rwa: bool,
        integration_steps_per_period: int,
    ) -> None:
        """Replace the field settings without disturbing the accumulated phases."""
        self.gamma = gamma
        self.b_z = b_z
        self.b_1 = b_1
        self.rf_phase = rf_phase
        self.rwa = rwa
        self.integration_steps_per_period = integration_steps_per_period

    def reset(self) -> None:
        """Return to the |0> state with the field settings untouched."""
        self.theta = 0.0
        self.phi = 0.0
        self.lab_time = 0.0
        self.frame_phase = 0.0

    def _advanceRwa(self, dt: float) -> None:
        """Apply one exact rotating-wave rotation of angle `gamma * b_1 * dt`."""
        larmor_rate = self.gamma * self.b_z
        rabi_rate = self.gamma * self.b_1

        vector = self._rotateZ(self.getVector(), self.frame_phase)
        if rabi_rate != 0.0:
            axis = (math.cos(self.rf_phase), math.sin(self.rf_phase), 0.0)
            vector = self._rotateAboutAxis(vector, axis, -rabi_rate * dt)

        self.frame_phase += larmor_rate * dt
        vector = self._rotateZ(vector, -self.frame_phase)

        self._setFromVector(vector)
        self.lab_time += dt

    def _advanceFull(self, dt: float) -> None:
        """Integrate the full time-dependent model with equal RK4 substeps."""
        max_rate = abs(self.gamma) * (abs(self.b_z) + 2.0 * abs(self.b_1))
        if max_rate == 0.0:
            self.lab_time += dt
            return

        max_step = 2.0 * math.pi / (self.integration_steps_per_period * max_rate)
        step_count = max(1, math.ceil(dt / max_step))
        step = dt / step_count

        vector = self.getVector()
        time = self.lab_time
        for _ in range(step_count):
            vector = self._rk4(vector, time, step)
            time += step

        self._setFromVector(vector)
        self.lab_time = time
        self.frame_phase += self.gamma * self.b_z * dt

    def _advanceStaticField(self, dt: float) -> None:
        """Rotate exactly about the field when the carrier frequency is zero."""
        field = (2.0 * self.b_1 * math.cos(self.rf_phase), 0.0, 0.0)
        magnitude = math.sqrt(sum(component * component for component in field))
        if magnitude == 0.0 or self.gamma == 0.0:
            self.lab_time += dt
            return

        axis = tuple(component / magnitude for component in field)
        vector = self._rotateAboutAxis(self.getVector(), axis, -self.gamma * magnitude * dt)

        self._setFromVector(vector)
        self.lab_time += dt

    def _rk4(
        self,
        vector: tuple[float, float, float],
        time: float,
        step: float,
    ) -> tuple[float, float, float]:
        """Take one renormalized fourth-order Runge-Kutta step."""
        half_step = 0.5 * step

        k1 = self._fieldDerivative(vector, time)
        k2 = self._fieldDerivative(_addScaled(vector, k1, half_step), time + half_step)
        k3 = self._fieldDerivative(_addScaled(vector, k2, half_step), time + half_step)
        k4 = self._fieldDerivative(_addScaled(vector, k3, step), time + step)

        combined = tuple(
            k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index] for index in range(3)
        )
        result = _addScaled(vector, combined, step / 6.0)
        return _normalize(result)

    def _fieldDerivative(
        self,
        vector: tuple[float, float, float],
        time: float,
    ) -> tuple[float, float, float]:
        """Return `gamma * v x B(t)` for the current field settings."""
        field_x = 2.0 * self.b_1 * math.cos(self.gamma * self.b_z * time + self.rf_phase)
        field_z = self.b_z
        x, y, z = vector
        return (
            self.gamma * (y * field_z),
            self.gamma * (z * field_x - x * field_z),
            self.gamma * (-y * field_x),
        )

    def _setFromVector(self, vector: tuple[float, float, float]) -> None:
        """Store a vector as theta/phi, renormalizing first."""
        x, y, z = _normalize(vector)
        self.theta = math.acos(max(-1.0, min(1.0, z)))
        self.phi = math.atan2(y, x)

    def _rotateZ(
        self,
        vector: tuple[float, float, float],
        angle: float,
    ) -> tuple[float, float, float]:
        """Rotate a vector about the z axis."""
        cosine = math.cos(angle)
        sine = math.sin(angle)
        x, y, z = vector
        return (cosine * x - sine * y, sine * x + cosine * y, z)

    def _rotateAboutAxis(
        self,
        vector: tuple[float, float, float],
        axis: tuple[float, float, float],
        angle: float,
    ) -> tuple[float, float, float]:
        """Rotate a vector about a unit axis using Rodrigues' formula."""
        cosine = math.cos(angle)
        sine = math.sin(angle)
        cross = _cross(axis, vector)
        projection = axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2]
        return tuple(
            vector[index] * cosine
            + cross[index] * sine
            + axis[index] * projection * (1.0 - cosine)
            for index in range(3)
        )


def _cross(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the cross product of two 3-vectors."""
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _addScaled(
    vector: tuple[float, float, float],
    direction: tuple[float, float, float],
    scale: float,
) -> tuple[float, float, float]:
    """Return `vector + scale * direction`."""
    return (
        vector[0] + scale * direction[0],
        vector[1] + scale * direction[1],
        vector[2] + scale * direction[2],
    )


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return a unit vector, falling back to +z when the input is degenerate."""
    length = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    if length == 0.0:
        return (0.0, 0.0, 1.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)
