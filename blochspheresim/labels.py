"""Axis label definitions, expansion state, and screen-space projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxisLabel:
    """One labelled axis endpoint with its collapsed and expanded text."""

    axis: tuple[float, float, float]
    collapsed: str
    expanded: str


AXIS_LABELS: tuple[AxisLabel, ...] = (
    AxisLabel((0.0, 0.0, 1.0), "|0>", "|0>"),
    AxisLabel((0.0, 0.0, -1.0), "|1>", "|1>"),
    AxisLabel((1.0, 0.0, 0.0), "|+>", "(|0> + |1>)/sqrt(2)"),
    AxisLabel((-1.0, 0.0, 0.0), "|->", "(|0> - |1>)/sqrt(2)"),
    AxisLabel((0.0, 1.0, 0.0), "|+i>", "(|0> + i|1>)/sqrt(2)"),
    AxisLabel((0.0, -1.0, 0.0), "|-i>", "(|0> - i|1>)/sqrt(2)"),
)


class LabelSet:
    """Runtime expansion state for the fixed axis labels."""

    def __init__(self) -> None:
        """Start every label collapsed."""
        self._expanded = [False] * len(AXIS_LABELS)

    def toggleAt(self, index: int) -> None:
        """Flip the expansion state of one label."""
        self._expanded[index] = not self._expanded[index]

    def isExpanded(self, index: int) -> bool:
        """Return whether a label is currently expanded."""
        return self._expanded[index]

    def getText(self, index: int) -> str:
        """Return the text to draw for one label."""
        label = AXIS_LABELS[index]
        return label.expanded if self._expanded[index] else label.collapsed


def projectPoint(
    point: tuple[float, float, float],
    view_projection: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float] | None:
    """Project a world point into pixel coordinates, or None when it is behind the camera."""
    clip = view_projection @ np.array([*point, 1.0], dtype=np.float64)
    if clip[3] <= 0.0:
        return None

    ndc = clip[:3] / clip[3]
    return (ndc[0] + 1.0) * 0.5 * width, (1.0 - ndc[1]) * 0.5 * height
