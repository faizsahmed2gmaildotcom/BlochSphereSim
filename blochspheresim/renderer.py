"""ModernGL renderer for the wireframe sphere, state vector, and text overlay."""

from __future__ import annotations

import math

import moderngl
import numpy as np
import pygame

from .camera import Camera, blochToOpenGl, toGlBytes
from .config import BlochVisuals
from .labels import AXIS_LABELS, LabelSet, projectPoint
from .trail import Trail, TrailRecorder


BACKGROUND_COLOR = (0.04, 0.05, 0.08)
LATITUDE_COLOR = (0.55, 0.65, 0.80)
LONGITUDE_COLOR = (0.60, 0.75, 0.90)
AXIS_COLOR = (0.70, 0.70, 0.70, 0.35)
STATE_COLOR = (1.0, 0.55, 0.15, 1.0)
TIP_COLOR = (1.0, 0.75, 0.20, 1.0)
TRAIL_COLOR = (0.95, 0.45, 0.12, 0.85)
TRAIL_HEAD_COLOR = (0.95, 0.50, 0.15, 1.0)
TEXT_COLOR = (255, 255, 255, 255)
POINT_SIZE = 10.0
TRAIL_STRIDE_BYTES = 16
LATITUDE_SEGMENTS = 128
LONGITUDE_SEGMENTS = 64
LABEL_MARGIN = 4

LINE_VERTEX_SHADER = """
#version 330 core

in vec3 in_pos;
in vec4 in_color;

uniform mat4 u_mvp;

out vec4 v_color;

void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    gl_PointSize = %f;
    v_color = in_color;
}
""" % POINT_SIZE

LINE_FRAGMENT_SHADER = """
#version 330 core

in vec4 v_color;
out vec4 f_color;

void main() {
    f_color = v_color;
}
"""

TEXT_VERTEX_SHADER = """
#version 330 core

in vec2 in_pos;
in vec2 in_uv;

out vec2 v_uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

TEXT_FRAGMENT_SHADER = """
#version 330 core

uniform sampler2D u_texture;

in vec2 v_uv;
out vec4 f_color;

void main() {
    f_color = texture(u_texture, v_uv);
}
"""

TRAIL_VERTEX_SHADER = """
#version 330 core

in vec3 in_pos;
in float in_born;

uniform mat4 u_mvp;
uniform float u_now;
uniform float u_solid;
uniform float u_fade;

out float v_alpha;

void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);

    float age = u_now - in_born;
    float alpha = 1.0;
    if (age < 0.0) {
        alpha = 0.0;
    } else if (age > u_solid) {
        alpha = u_fade > 0.0 ? clamp((u_solid + u_fade - age) / u_fade, 0.0, 1.0) : 0.0;
    }
    v_alpha = alpha;
}
"""

TRAIL_FRAGMENT_SHADER = """
#version 330 core

uniform vec4 u_color;

in float v_alpha;
out vec4 f_color;

void main() {
    f_color = vec4(u_color.rgb, u_color.a * v_alpha);
}
"""


class _TrailBuffers:
    """GL resources and upload bookkeeping for one trail history."""

    def __init__(
        self,
        buffer: moderngl.Buffer,
        vao: moderngl.VertexArray,
        capacity: int,
    ) -> None:
        """Store the resources and mark the history as not yet uploaded."""
        self.buffer = buffer
        self.vao = vao
        self.capacity = capacity
        self.revision = -1
        self.generation = -1


class Renderer:
    """Own the GL programs, geometry, and the camera-facing text overlay."""

    def __init__(
        self,
        context: moderngl.Context,
        visuals: BlochVisuals,
        size: tuple[int, int],
    ) -> None:
        """Create programs and buffers for the given window size."""
        self._context = context
        self._context.enable(moderngl.PROGRAM_POINT_SIZE)
        self._visuals = visuals
        self._size = size
        self._font: pygame.font.Font | None = None
        self._text_cache: dict[tuple[str, int], pygame.Surface] = {}
        self._static_buffer: moderngl.Buffer | None = None
        self._static_vao: moderngl.VertexArray | None = None
        self._text_texture: moderngl.Texture | None = None
        self._text_surface = pygame.Surface(size, pygame.SRCALPHA)

        self._line_program = context.program(
            vertex_shader=LINE_VERTEX_SHADER,
            fragment_shader=LINE_FRAGMENT_SHADER,
        )
        self._text_program = context.program(
            vertex_shader=TEXT_VERTEX_SHADER,
            fragment_shader=TEXT_FRAGMENT_SHADER,
        )
        self._trail_program = context.program(
            vertex_shader=TRAIL_VERTEX_SHADER,
            fragment_shader=TRAIL_FRAGMENT_SHADER,
        )
        self._trail_buffers: dict[str, _TrailBuffers] = {}

        self._state_buffer = context.buffer(reserve=2 * 7 * 4)
        self._state_vao = context.vertex_array(
            self._line_program,
            [(self._state_buffer, "3f 4f", "in_pos", "in_color")],
        )
        self._tip_buffer = context.buffer(reserve=7 * 4)
        self._tip_vao = context.vertex_array(
            self._line_program,
            [(self._tip_buffer, "3f 4f", "in_pos", "in_color")],
        )
        self._connector_buffer = context.buffer(reserve=2 * 7 * 4)
        self._connector_vao = context.vertex_array(
            self._line_program,
            [(self._connector_buffer, "3f 4f", "in_pos", "in_color")],
        )

        self._buildOverlay()
        self.applyVisuals(visuals)
        self.resize(size)

    def applyVisuals(self, visuals: BlochVisuals) -> None:
        """Adopt visual settings, rebuilding geometry and the font cache if needed."""
        if visuals.font_size != self._visuals.font_size:
            self._font = None
            self._text_cache.clear()

        self._visuals = visuals
        self._buildStaticGeometry()

    def resize(self, size: tuple[int, int]) -> None:
        """Resize the viewport and the full-window text texture."""
        self._size = size
        self._context.viewport = (0, 0, size[0], size[1])
        self._text_surface = pygame.Surface(size, pygame.SRCALPHA)

        if self._text_texture is not None:
            self._text_texture.release()
        self._text_texture = self._context.texture(size, 4)
        self._text_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._text_texture.swizzle = "BGRA"

    def draw(
        self,
        camera: Camera,
        state_vector: tuple[float, float, float],
        labels: LabelSet,
        trails: TrailRecorder | None = None,
        rotating: bool = False,
    ) -> list[tuple[int, tuple[int, int, int, int]]]:
        """Draw one frame and return the clickable label rectangles with their indices."""
        self._context.clear(*BACKGROUND_COLOR)
        view_projection = camera.getViewProjectionMatrix()

        active_trail: Trail | None = None
        active_buffers: _TrailBuffers | None = None
        if trails is not None and trails.enabled:
            lab_buffers = self._syncTrail("lab", trails.lab)
            rotating_buffers = self._syncTrail("rotating", trails.rotating)
            active_trail = trails.rotating if rotating else trails.lab
            active_buffers = rotating_buffers if rotating else lab_buffers

        self._context.enable(moderngl.DEPTH_TEST)
        self._drawStatic(view_projection)
        if trails is not None and active_trail is not None and active_buffers is not None:
            self._drawTrail(
                view_projection,
                active_buffers,
                active_trail,
                state_vector,
                trails.now,
                trails.solid_duration,
                trails.fading_duration,
            )
        self._drawState(view_projection, state_vector)
        self._context.disable(moderngl.DEPTH_TEST)

        return self._drawLabels(view_projection, labels)

    def _syncTrail(self, kind: str, trail: Trail | None) -> _TrailBuffers | None:
        """Upload new samples for one history and return its GPU state."""
        if trail is None or trail.count == 0:
            return None

        buffers = self._ensureTrailBuffers(kind, trail.capacity)
        if buffers.generation != trail.generation:
            rows = trail.getChronological()
        elif buffers.revision != trail.revision:
            rows = trail.getSamplesSince(buffers.revision)
        else:
            rows = None

        if rows is not None and len(rows) > 0:
            self._writeTrailRows(buffers, rows, trail.getSampleStart(len(rows)))

        buffers.generation = trail.generation
        buffers.revision = trail.revision
        return buffers

    def _ensureTrailBuffers(self, kind: str, capacity: int) -> _TrailBuffers:
        """Return the mirrored buffers for a history, reallocating when capacity changes."""
        buffers = self._trail_buffers.get(kind)
        if buffers is not None and buffers.capacity == capacity:
            return buffers

        if buffers is not None:
            buffers.vao.release()
            buffers.buffer.release()

        buffer = self._context.buffer(reserve=2 * capacity * TRAIL_STRIDE_BYTES)
        vao = self._context.vertex_array(
            self._trail_program,
            [(buffer, "3f 1f", "in_pos", "in_born")],
        )
        buffers = _TrailBuffers(buffer, vao, capacity)
        self._trail_buffers[kind] = buffers
        return buffers

    def _writeTrailRows(
        self,
        buffers: _TrailBuffers,
        rows: np.ndarray,
        first_index: int,
    ) -> None:
        """Write rows to a ring and its mirror, splitting at most once per copy."""
        position = 0
        index = first_index
        total = len(rows)

        while position < total:
            run = min(total - position, buffers.capacity - index)
            chunk = np.ascontiguousarray(rows[position : position + run], dtype="f4").tobytes()
            buffers.buffer.write(chunk, offset=index * TRAIL_STRIDE_BYTES)
            buffers.buffer.write(chunk, offset=(index + buffers.capacity) * TRAIL_STRIDE_BYTES)
            position += run
            index = 0

    def _drawTrail(
        self,
        view_projection: np.ndarray,
        buffers: _TrailBuffers,
        trail: Trail,
        state_vector: tuple[float, float, float],
        now: float,
        solid: float,
        fading: float,
    ) -> None:
        """Draw the active history plus a connector from its newest sample to the vector."""
        self._context.depth_mask = False

        if trail.count >= 2:
            program = self._trail_program
            program["u_mvp"].write(toGlBytes(view_projection))
            program["u_now"].value = now
            program["u_solid"].value = solid
            program["u_fade"].value = fading
            program["u_color"].value = TRAIL_COLOR
            buffers.vao.render(
                moderngl.LINE_STRIP,
                vertices=trail.count,
                first=trail.getSampleStart(trail.count),
            )

        newest = trail.getNewest()
        if newest is not None:
            start = newest[0]
            end = blochToOpenGl(state_vector)
            connector = np.array(
                [*start, *TRAIL_HEAD_COLOR, *end, *TRAIL_HEAD_COLOR],
                dtype="f4",
            )
            self._connector_buffer.write(connector.tobytes())
            self._line_program["u_mvp"].write(toGlBytes(view_projection))
            self._connector_vao.render(moderngl.LINES)

        self._context.depth_mask = True

    def _buildOverlay(self) -> None:
        """Create the fullscreen quad used to composite the text texture."""
        vertices = np.array(
            [
                -1.0, -1.0, 0.0, 1.0,
                1.0, -1.0, 1.0, 1.0,
                -1.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 0.0,
            ],
            dtype="f4",
        )
        self._overlay_buffer = self._context.buffer(vertices.tobytes())
        self._overlay_vao = self._context.vertex_array(
            self._text_program,
            [(self._overlay_buffer, "2f 2f", "in_pos", "in_uv")],
        )

    def _buildStaticGeometry(self) -> None:
        """Rebuild the latitude, longitude, and axis line buffers."""
        vertices: list[float] = []
        self._appendLatitudes(vertices)
        self._appendLongitudes(vertices)
        self._appendAxes(vertices)

        data = np.asarray(vertices, dtype="f4").tobytes()
        if self._static_vao is not None:
            self._static_vao.release()
        if self._static_buffer is not None:
            self._static_buffer.release()

        self._static_buffer = self._context.buffer(data)
        self._static_vao = self._context.vertex_array(
            self._line_program,
            [(self._static_buffer, "3f 4f", "in_pos", "in_color")],
        )

    def _appendLatitudes(self, vertices: list[float]) -> None:
        """Append evenly spaced latitude circles in Bloch space."""
        count = self._visuals.latitude_count
        color = (*LATITUDE_COLOR, self._visuals.latitude_opacity)

        for index in range(1, count + 1):
            theta = math.pi * index / (count + 1)
            radius = math.sin(theta)
            height = math.cos(theta)
            for segment in range(LATITUDE_SEGMENTS):
                first = 2.0 * math.pi * segment / LATITUDE_SEGMENTS
                second = 2.0 * math.pi * (segment + 1) / LATITUDE_SEGMENTS
                self._appendVertex(
                    vertices,
                    (radius * math.cos(first), radius * math.sin(first), height),
                    color,
                )
                self._appendVertex(
                    vertices,
                    (radius * math.cos(second), radius * math.sin(second), height),
                    color,
                )

    def _appendLongitudes(self, vertices: list[float]) -> None:
        """Append evenly spaced longitude meridians in Bloch space."""
        count = self._visuals.longitude_count
        color = (*LONGITUDE_COLOR, self._visuals.longitude_opacity)

        for index in range(count):
            phi = 2.0 * math.pi * index / count
            for segment in range(LONGITUDE_SEGMENTS):
                first = math.pi * segment / LONGITUDE_SEGMENTS
                second = math.pi * (segment + 1) / LONGITUDE_SEGMENTS
                self._appendVertex(
                    vertices,
                    (math.sin(first) * math.cos(phi), math.sin(first) * math.sin(phi), math.cos(first)),
                    color,
                )
                self._appendVertex(
                    vertices,
                    (math.sin(second) * math.cos(phi), math.sin(second) * math.sin(phi), math.cos(second)),
                    color,
                )

    def _appendAxes(self, vertices: list[float]) -> None:
        """Append the three axis diameters through the sphere."""
        axes = (
            ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            ((0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
            ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
        )
        for start, end in axes:
            self._appendVertex(vertices, start, AXIS_COLOR)
            self._appendVertex(vertices, end, AXIS_COLOR)

    def _appendVertex(
        self,
        vertices: list[float],
        point: tuple[float, float, float],
        color: tuple[float, float, float, float],
    ) -> None:
        """Append one interleaved position and color to a vertex list."""
        x, y, z = blochToOpenGl(point)
        vertices.extend((x, y, z, color[0], color[1], color[2], color[3]))

    def _drawStatic(self, view_projection: np.ndarray) -> None:
        """Draw the sphere wireframe and axes."""
        self._line_program["u_mvp"].write(toGlBytes(view_projection))
        if self._static_vao is not None:
            self._static_vao.render(moderngl.LINES)

    def _drawState(
        self,
        view_projection: np.ndarray,
        state_vector: tuple[float, float, float],
    ) -> None:
        """Draw the state vector from the origin and its tip."""
        origin = blochToOpenGl((0.0, 0.0, 0.0))
        tip = blochToOpenGl(state_vector)
        line = np.array([*origin, *STATE_COLOR, *tip, *STATE_COLOR], dtype="f4")
        self._state_buffer.write(line.tobytes())
        self._state_vao.render(moderngl.LINES)

        point = np.array([*tip, *TIP_COLOR], dtype="f4")
        self._tip_buffer.write(point.tobytes())
        self._tip_vao.render(moderngl.POINTS)

    def _drawLabels(
        self,
        view_projection: np.ndarray,
        labels: LabelSet,
    ) -> list[tuple[int, pygame.Rect]]:
        """Render the axis labels into the overlay texture and return hit rectangles."""
        width, height = self._size
        self._text_surface.fill((0, 0, 0, 0))
        scale = 1.0 + self._visuals.label_offset
        rectangles: list[tuple[int, pygame.Rect]] = []

        for index, label in enumerate(AXIS_LABELS):
            anchor = tuple(component * scale for component in label.axis)
            point = projectPoint(blochToOpenGl(anchor), view_projection, width, height)
            if point is None:
                continue

            text = labels.getText(index)
            text_surface = self._getTextSurface(text)
            rectangle = text_surface.get_rect(
                center=(round(point[0]), round(point[1]))
            )
            self._text_surface.blit(text_surface, rectangle)
            rectangles.append((index, rectangle.inflate(2 * LABEL_MARGIN, 2 * LABEL_MARGIN)))

        if self._text_texture is not None:
            self._text_texture.write(self._text_surface.get_view("1"))
            self._text_texture.use(0)
            self._text_program["u_texture"] = 0
            self._overlay_vao.render(moderngl.TRIANGLE_STRIP)

        return rectangles

    def _getTextSurface(self, text: str) -> pygame.Surface:
        """Render and cache one label string at the configured font size."""
        key = (text, self._visuals.font_size)
        surface = self._text_cache.get(key)
        if surface is None:
            if self._font is None:
                self._font = pygame.font.Font(None, self._visuals.font_size)
            surface = self._font.render(text, True, TEXT_COLOR)
            self._text_cache[key] = surface
        return surface
