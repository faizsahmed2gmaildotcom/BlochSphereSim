"""Application window, event loop, and config hot reload."""

from __future__ import annotations

from pathlib import Path

import moderngl
import pygame

from .bloch_state import BlochState
from .camera import Camera
from .config import (
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigError,
    ConfigWatcher,
    buildConfig,
    collectPhysicsWarnings,
    loadConfig,
)
from .labels import LabelSet
from .renderer import Renderer


WINDOW_SIZE = (1280, 800)
WINDOW_CAPTION = "Bloch Sphere Simulator"
LAB_FRAME = "Lab Frame"
ROTATING_FRAME = "Rotating Frame"
MAX_REAL_STEP = 1.0


class App:
    """Own the window, GL context, simulation state, and input handling."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        """Load configuration, create the window, and build the simulation."""
        self._config_path = Path(config_path)
        self._config = self._loadInitialConfig()
        self._initializePygame()

        self._context = moderngl.create_context()
        self._context.enable(moderngl.BLEND)
        self._context.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self._renderer = Renderer(self._context, self._config.bloch_visuals, WINDOW_SIZE)
        self._camera = Camera(self._config.controls)
        self._state = self._createState(self._config)
        self._labels = LabelSet()
        self._watcher = ConfigWatcher(self._config_path)

        self._rotating = False
        self._middle_drag = False
        self._running = True
        self._label_rects: list[tuple[int, tuple[int, int, int, int]]] = []
        self._printed_warnings: set[str] = set()

        self._updateCaption()
        self._printWarnings(self._config)

    def run(self) -> None:
        """Run the main loop until the window closes."""
        clock = pygame.time.Clock()
        try:
            while self._running:
                max_frame_rate = self._config.bloch_visuals.max_frame_rate
                dt_real = clock.tick(max_frame_rate) / 1000.0
                dt_sim = min(dt_real, MAX_REAL_STEP) * self._config.bloch_visuals.timespeed

                self._state.advanceTime(dt_sim)
                self._handleEvents()
                self._pollConfig(dt_real)
                self._draw()
        finally:
            pygame.quit()

    def _initializePygame(self) -> None:
        """Create the resizable OpenGL window."""
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 1)
        self._screen = pygame.display.set_mode(
            WINDOW_SIZE,
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
        )
        pygame.display.set_caption(WINDOW_CAPTION)

    def _loadInitialConfig(self) -> Config:
        """Load config, falling back to validated defaults on failure."""
        try:
            return loadConfig(self._config_path)
        except ConfigError as error:
            print(f"config load failed, using defaults: {error}")
            return buildConfig({})

    def _createState(self, config: Config) -> BlochState:
        """Build the Bloch state from the physics section."""
        physics = config.physics
        return BlochState(
            gamma=physics.gamma,
            b_z=physics.b_z,
            b_1=physics.b_1,
            rf_phase=physics.rf_phase,
            rwa=physics.rwa,
            integration_steps_per_period=physics.integration_steps_per_period,
        )

    def _handleEvents(self) -> None:
        """Dispatch one frame of pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                self._handleKeyDown(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handleMouseButtonDown(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    self._middle_drag = False
            elif event.type == pygame.MOUSEMOTION:
                if self._middle_drag:
                    self._camera.orbit(event.rel[0], event.rel[1])
            elif event.type == pygame.MOUSEWHEEL:
                self._camera.zoom(event.y)
            elif event.type == pygame.VIDEORESIZE:
                self._resizeIfChanged(event.size)
            elif event.type == pygame.WINDOWRESIZED:
                self._resizeIfChanged((event.x, event.y))

    def _handleKeyDown(self, event: pygame.event.Event) -> None:
        """Handle the frame toggle, reset, and quit keys."""
        if event.key == pygame.K_ESCAPE:
            self._running = False
        elif event.key == pygame.K_f:
            self._rotating = not self._rotating
            self._updateCaption()
        elif event.key == pygame.K_r:
            self._state.reset()
            self._camera.reset(self._config.controls)

    def _handleMouseButtonDown(self, event: pygame.event.Event) -> None:
        """Start a middle-drag orbit or toggle a clicked label."""
        if event.button == 2:
            self._middle_drag = True
        elif event.button == 1:
            self._toggleLabelAt(event.pos)

    def _toggleLabelAt(self, position: tuple[int, int]) -> None:
        """Toggle the first label whose click rectangle contains the position."""
        for index, rectangle in self._label_rects:
            if _rectContains(rectangle, position):
                self._labels.toggleAt(index)
                return

    def _pollConfig(self, dt_real: float) -> None:
        """Apply a hot-reloaded config when the watcher reports one."""
        new_config = self._watcher.poll(dt_real)
        if new_config is not None:
            self._applyConfig(new_config)

    def _applyConfig(self, config: Config) -> None:
        """Apply a validated config without disturbing the simulation phases."""
        physics = config.physics
        self._renderer.applyVisuals(config.bloch_visuals)
        self._camera.applyControls(config.controls)
        self._state.setFields(
            physics.gamma,
            physics.b_z,
            physics.b_1,
            physics.rf_phase,
            physics.rwa,
            physics.integration_steps_per_period,
        )
        self._config = config
        self._printWarnings(config)

    def _resizeIfChanged(self, size: tuple[int, int]) -> None:
        """Resize only when the requested size differs from the current window."""
        target = (max(size[0], 1), max(size[1], 1))
        if tuple(self._screen.get_size()) == target:
            return
        self._resize(target)

    def _resize(self, size: tuple[int, int]) -> None:
        """Resize the window, viewport, projection, and text overlay."""
        self._screen = pygame.display.set_mode(
            size,
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
        )
        self._context.viewport = (0, 0, size[0], size[1])
        self._camera.setAspect(size[0] / max(size[1], 1))
        self._renderer.resize(size)

    def _draw(self) -> None:
        """Render the current state and refresh the display."""
        self._label_rects = self._renderer.draw(
            self._camera,
            self._state.getDisplayVector(self._rotating),
            self._labels,
        )
        pygame.display.flip()

    def _updateCaption(self) -> None:
        """Show the active reference frame in the window title."""
        frame = ROTATING_FRAME if self._rotating else LAB_FRAME
        pygame.display.set_caption(f"{WINDOW_CAPTION} - {frame}")

    def _printWarnings(self, config: Config) -> None:
        """Print each physics warning at most once per session."""
        for message in collectPhysicsWarnings(config.physics):
            if message not in self._printed_warnings:
                print(f"warning: {message}")
                self._printed_warnings.add(message)


def _rectContains(rectangle: tuple[int, int, int, int], position: tuple[int, int]) -> bool:
    """Return whether a pygame-style rectangle contains a point."""
    left, top, width, height = rectangle
    x, y = position
    return left <= x < left + width and top <= y < top + height
