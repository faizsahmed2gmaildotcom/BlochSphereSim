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
  renderer.py      GL programs, geometry, text overlay
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

## Testing

- Stdlib `unittest` only. Physics/config/camera logic must be testable without a window.
- Run the full suite before finishing. Manual smoke test: window opens, sphere and labels render,
  `F` toggles frames, `R` resets, clicks expand labels, editing `config.yaml` applies live.
- When changing camera/matrix code, rerun `testAxisMapping` and `testTranslationLandsInFourthGlslColumn`.

## Deferred scope

RF pulses, gates, detuning, relaxation, and ensembles are documented in
`plans/future-extensions/rf-control.md`; do not implement them without being asked.
