# Bloch Sphere Simulator

A single-qubit Bloch sphere simulator with a ModernGL renderer, driven by a static z-field and a linearly polarized radio-frequency field. Runs in a resizable 1280x800 pygame window.

- Wireframe unit sphere with configurable latitude/longitude counts and opacity.
- Camera-facing, clickable axis labels (`|0>`, `|+>`, `|+i>`, ...).
- Lab-frame and rotating-frame views.
- Exact rotating-wave evolution, or the full time-dependent model via RK4.
- Live-updating `config.yaml` (edit and save while the simulator is running).
- Simulation speed from 0 to 1x, plus a configurable frame-rate cap.

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- A desktop with OpenGL 3.3+ (Linux Wayland/X11, macOS, or Windows)

Dependencies (`moderngl`, `numpy`, `pygame-ce`, `pyyaml`) are declared in `pyproject.toml` and pinned in `uv.lock`.

## Install and run

```sh
git clone https://github.com/faizsahmed2gmaildotcom/BlochSphereSim.git && cd BlochSphereSim
uv sync
uv run python -m blochspheresim
```

The simulator loads `config.yaml` from the current directory. Run the command from the repository
root, or copy `config.yaml` next to where you launch it.

## Controls

| Input               | Action                                                    |
| ------------------- | --------------------------------------------------------- |
| Middle-click + drag | Orbit the camera around the sphere at a constant distance |
| Scroll wheel        | Zoom in/out, clamped to the configured distance limits    |
| `F`                 | Toggle between the lab frame and the rotating frame       |
| `R`                 | Reset the quantum state to `+\|0>`                        |
| `Esc`               | Quit                                                      |
| Left-click a label  | Expand or collapse that label                             |

Labels start collapsed. Clicking `+Z`/`-Z` changes nothing visible because their collapsed and expanded forms are both `|0>`/`|1>`, per the quantum-state table.

At the default camera angle, `+X` and `-X` project onto the same screen point (front and back of the sphere); orbit the camera to reach the hidden one. The same happens to any opposite pair when
viewed along its axis.

## Interface

- The state vector is drawn from the origin to the current Bloch vector, with a dot at the tip.
- A trailing line follows the vector: fully opaque for the configured solid duration, then fading
  linearly to transparent. Separate lab and rotating histories are recorded continuously, so `F`
  switches between complete trails with no redraw or popping.
- The window title shows the active frame: "Lab Frame" or "Rotating Frame".
- The three axes are dim gray; the sphere wireframe uses low-opacity latitude/longitude circles.
- Zooming in far enough clips the labels, since they sit just outside the sphere surface.

## Settings (`config.yaml`)

Every value below is a number (except `rwa`) and can be changed at runtime. The file is polled for changes about twice per second; invalid edits are rejected and the last good configuration stays active.

### `bloch_visuals`

| Setting             | Default | Meaning                                                                 |
| ------------------- | ------- | ----------------------------------------------------------------------- |
| `latitude_count`    | `7`     | Latitude circles; snapped to an odd number                              |
| `longitude_count`   | `8`     | Meridians; snapped to an even number                                    |
| `latitude_opacity`  | `0.25`  | Latitude alpha, 0-1                                                     |
| `longitude_opacity` | `0.55`  | Meridian alpha, 0-1                                                     |
| `label_offset`      | `0.05`  | Label distance beyond the surface, as a fraction of the radius          |
| `font_size`         | `20`    | Label font size; minimum 8                                              |
| `timespeed`         | `1.0`   | Simulated seconds per real second, clamped to 0-1 (0 freezes the state) |
| `max_frame_rate`    | `60`    | Main-loop frame-rate cap                                                |
| `trail_solid_duration`  | `0.5`      | Simulated seconds of fully opaque trail behind the vector          |
| `trail_fading_duration` | `1.5`      | Simulated seconds of linear fade after the solid head              |
| `trail_sample_interval` | `0.016667` | Simulated seconds between trail samples (about 60 per second)      |

The trail duration is measured in **simulated** time, so it is unaffected by `timespeed`, frame
rate, or camera distance. Setting both durations to `0` disables the trail. Durations are clamped to
`>= 0`, and the sample interval is clamped to at least `1e-4` simulated seconds. If a request would
need more than 100,000 samples per history, the interval is raised automatically and a warning is
printed.

Editing durations while running applies immediately: shrinking keeps the newest points, and the
same stored samples follow the new solid/fade split. Changing the sample interval keeps the existing
history and starts sampling at the new cadence right away. `R` clears both trails along with the
state reset.

### `controls`

| Setting              | Default | Meaning                                     |
| -------------------- | ------- | ------------------------------------------- |
| `pan_sensitivity`    | `0.005` | Orbit radians per pixel of middle-drag      |
| `scroll_sensitivity` | `0.1`   | Logarithmic distance change per wheel notch |
| `default_distance`   | `3.0`   | Camera distance at startup and after `R`    |
| `min_distance`       | `1.5`   | Closest allowed distance from the centre    |
| `max_distance`       | `10.0`  | Farthest allowed distance from the centre   |

`default_distance` is applied only at startup and on `R`; editing it mid-session does not move the
camera.

### `physics`

| Setting                        | Default | Meaning                                                         |
| ------------------------------ | ------- | --------------------------------------------------------------- |
| `gamma`                        | `1.0`   | Gyromagnetic ratio, rad/s per field unit                        |
| `b_z`                          | `3.0`   | Static field along z                                            |
| `b_1`                          | `0.3`   | RF amplitude                                                    |
| `rf_phase`                     | `0.0`   | RF phase in radians (`pi/2` drives about the `y'` axis)         |
| `rwa`                          | `true`  | Rotating-wave model on/off                                      |
| `integration_steps_per_period` | `512`   | RK4 steps per fastest field period, full model only; minimum 16 |

Values are in simulation units by default: the lab Larmor period is about 2.09 s and a Rabi
pi-pulse takes about 10.47 s, so both frames are readable at `timespeed: 1`. A commented real-proton
example is included at the bottom of `config.yaml`; with real constants the lab view aliases and the
rotating frame is the usable one.

## Physics notes

The lab field is the linearly polarized drive

```
B(t) = ( 2 * b_1 * cos(w_L * t + rf_phase), 0, b_z ),    w_L = gamma * b_z
```

and the Bloch vector follows `dv/dt = gamma * v x B(t)`.

- **`rwa: true`** (default): the drive is solved in the frame rotating at the Larmor frequency,
  where it becomes the static field `b_1 * (cos rf_phase, sin rf_phase, 0)`. The state rotates about
  that axis at the Rabi frequency `gamma * b_1`. This solution is exact for the rotating-wave model
  for any frame time step, so simulation speed does not affect accuracy.
- **`rwa: false`**: the full linearly driven model is integrated with fourth-order Runge-Kutta and
  renormalized every step. Longer simulated times are fine; very large fields are not. Accuracy is
  governed by `integration_steps_per_period`, and the simulator prints a one-time warning when the
  estimated step rate becomes too high to keep up.
- **`b_z: 0`** makes the drive static; the simulator uses the exact rotation about that fixed field
  regardless of `rwa`.
- Warnings are printed at load/reload if `rwa` is on with `b_1 > 0.25 * |b_z|` (rotating-wave
  accuracy degrades) or with `b_z == 0`.

## Tests

```sh
uv run python -m unittest discover -s tests -v
```

The suite covers state evolution (norm, RWA exactness, RK4 convergence, static-field limit),
configuration validation and hot reload, trail ring/fade/sampling behaviour, and the camera/projection
conventions. There are no automated GL tests; verify rendering by running the simulator.

## Troubleshooting

- **No window / GL context error**: check that your GPU driver supports OpenGL 3.3+ and that a
  desktop session is available.
- **`pygame` font errors**: use the `pygame-ce` dependency from `uv.lock`; stock `pygame` builds
  without SDL2_ttf cannot render labels. The import name is still `pygame`.
- **Config edits ignored**: the file must be valid YAML, and the simulator only polls it while
  running. Errors are printed to the terminal and the previous configuration stays active.
