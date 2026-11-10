# AGENTS.md

Project-specific instructions for agents working in this repository.

## Project

Single-qubit Bloch sphere simulator: ModernGL rendering in a pygame-ce window, Python 3.14.
Physics and rendering are independent modules; the loop is single-threaded (physics then render).

## Commands

- Run: `uv run python -m blochspheresim`
- Test: `uv run python -m unittest discover -s tests -v`
- Add a dependency: `uv add <package>` (always use `uv`, never bare `pip`)
- Do not swap `pygame-ce` for `pygame`: the stock pygame build has no font module. The import
  name remains `pygame`.

## Layout

```
blochspheresim/
  app.py           window, loop, input, hot reload, label picking
  bloch_state.py   physics only; no pygame/moderngl imports
  camera.py        orbit camera, matrices, Bloch -> OpenGL mapping
  config.py        dataclasses, YAML validation, ConfigWatcher
  labels.py        axis label definitions, expansion state, projection
  renderer.py      GL programs, geometry, text overlay, trail pass
  trail.py         ring-buffer histories, fixed-grid sampler, fade math
  __main__.py      entry point
config.yaml        runtime settings (hot-reloaded)
tests/             stdlib unittest; no headless-GL tests
plans/             specs, final plan, deferred work (gitignored)
research/          physics reference PDF (gitignored)
```

## Style

Follow `faiz-code-style`:

- files/variables/attributes `snake_case`; classes `PascalCase`; methods/functions `camelCase`;
  constants `MACRO_CASE`.
- Every class and method gets a docstring. Comment only non-obvious physics/GL conventions.
- One major task per method; split minor tasks with blank lines.

## Physics invariants

- Lab field: `B(t) = (2*b_1*cos(w_L*t + rf_phase), 0, b_z)`, `w_L = gamma*b_z`;
  motion `dv/dt = gamma * v x B(t)`.
- `rwa: true` uses the exact rotating-wave rotation (compose `Rz(frame_phase)`, Rabi rotation about
  `(cos rf_phase, sin rf_phase, 0)`, `Rz(-frame_phase)`). It must stay exact/composable for any `dt`.
- `rwa: false` uses RK4 on the full model with equal substeps (`n = ceil(dt/h_max)`, `h = dt/n`) and
  renormalizes every step. Keep the integrator in plain floats/tuples; numpy 3-vectors are ~100x
  slower here.
- `frame_phase` is accumulated, never recomputed from `w_L * lab_time`, so config edits stay
  continuous.
- `b_z == 0` always uses the exact static-field rotation, regardless of `rwa`.
- `timespeed` scales `dt` in `App`, not in `BlochState`; Larmor and RF stay resonant.

## Config rules

- Config controls numbers only (plus the `rwa` boolean). Labels and text are hardcoded.
- New settings belong in `config.py` (dataclass + validation), `config.yaml` (documented default),
  and `tests/test_config.py`.
- Apply rules: visuals/camera/physics are live; `default_distance` only at startup and on `R`.
- Bad reloads must keep the last good config and print once; never crash the loop.

## Rendering conventions

- Bloch -> OpenGL matrix is `[[0,1,0],[0,0,1],[1,0,0]]` (X->GL Z, Y->GL X, Z->GL Y), applied on the
  CPU. This is the transpose of the matrix printed in `plans/specs.md`; the transpose is intentional
  and required for +Z to be the north pole. `tests/test_camera.py` pins it.
- Text is a screen-space overlay drawn after 3D (depth test off), so labels always face the camera;
  never render text in world space.
- `moderngl.PROGRAM_POINT_SIZE` must stay enabled for the state-vector tip (`gl_PointSize`).
- Overlay quad maps v=0 to the top of the screen (surface row 0 = texture v=0), and the texture uses
  swizzle `"BGRA"`. Changing either flips/miscolors the text.
- Keep the sphere wireframe only; no solid surface or depth occlusion of labels.

## Trail invariants

- Sample by **simulated time**, never by frame or distance. `TrailRecorder.advance` sub-steps
  `BlochState.advanceTime` onto the `trail_sample_interval` grid; physics must end at the same
  `lab_time` it would without sampling. RWA chunking is exact; full-model chunking stays <1e-6 deg.
- Each frame has its own fixed-capacity ring (`Trail`) storing `(x, y, z, born)` float32 in Bloch
  space. Lab records `getVector()`; rotating records `getDisplayVector(True)` **at capture time**.
  Never re-project stored lab history with the current `frame_phase`: it smears into a ring.
- Both histories are recorded every sample so `F` can switch instantly. Only the active one is
  drawn; toggling frames must not touch buffers.
- Fade is a shader function of `now - born`: opaque through `trail_solid_duration`, linear to zero
  across `trail_fading_duration`, hard cut when the fade is zero. `computeAlpha` pins the same rule.
- Capacity is `ceil((solid + fade) / interval) + 2`, capped at `MAX_TRAIL_POINTS = 100_000` by
  raising the interval in config validation. `0` total duration means no allocation and no drawing.
- Resize keeps the newest points and bumps `generation`; recording bumps `revision`. The renderer
  uploads incrementally via `revision`, full-rebuilds on `generation`, and mirrors each write into a
  `2 * capacity` VBO so one `GL_LINE_STRIP` covers the wrap.
- `R` clears both trails and the sampling accumulator. `timespeed: 0` freezes both state and fade.
- Trail drawing uses `depth_mask = False` (depth test stays on) and is drawn before the state vector
  so it can never occlude the vector. A two-vertex head connector joins the newest sample to the
  current vector.

## Testing

- Stdlib `unittest` only. Physics/config/camera logic must be testable without a window.
- Run the full suite before finishing. Manual smoke test: window opens, sphere and labels render,
  `F` toggles frames, `R` resets, clicks expand labels, editing `config.yaml` applies live, and the
  trail fades in simulated time (freezing `timespeed` must freeze the fade too).
- When changing camera/matrix code, rerun `testAxisMapping` and `testTranslationLandsInFourthGlslColumn`.

## Deferred scope

RF pulses, gates, detuning, relaxation, and ensembles are documented in
`plans/future-extensions/rf-control.md`; do not implement them without being asked.
