# BiGym Desktop Demo Recorder

Browser-based desktop teleoperation and success-gated demonstration recording for
`DishwasherUnloadCutleryLong`. The recorder supports manual keyboard control and an
optional privileged-state controller that executes through normal MuJoCo `env.step`
calls. It never teleports task objects or writes task completion state.

The browser shows the native D455 head view (`848x480`) and two D405 wrist views
(`640x480`) in one screen. It mirrors base, arm, and gripper commands as virtual key
presses and displays live task milestones and reset randomization.

## Install

From the repository root:

```bash
pip install -e '.[desktop-recorder]'
```

The privileged controller also needs the official 20 Hz lightweight demonstrations.
Pass their version directory with `--expert-demo-root`; the expected layout is:

```text
<demo-root>/DishwasherUnloadCutleryLong/
  JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/
    lightweight/20hz/*.safetensors
```

## Start

For a local machine with a display-capable MuJoCo backend:

```bash
MUJOCO_GL=egl python -m tools.desktop_demo_recorder.server \
  --host 127.0.0.1 \
  --port 8765 \
  --expert-demo-root "$HOME/.bigym/demonstrations/0.9.0"
```

Open <http://127.0.0.1:8765>. The server intentionally binds to loopback by default.
For a remote simulator, use an SSH tunnel instead of exposing the recorder publicly:

```bash
ssh -N -L 8765:127.0.0.1:8765 user@remote-host
```

The server starts paused. `New episode` resets the selected seed, `Start recording`
starts the 20 Hz loop, and `Save successful demo` is enabled only after the task emits
a positive reward. `Start automatic demonstration` resets the same seed and runs the
privileged closed-loop controller while recording the resulting interaction.

## Controls

- `W/S`: base forward/back
- `A/D`: base lateral motion
- `R/F`: base up/down
- `Q/E`: base yaw
- `Space`: emergency pause
- `T/G`, `Y/H`, `U/J`, `I/K`, `O/L`: left-arm joint jog pairs
- `Shift` plus the same pairs: right-arm joint jog pairs
- `Z/X`: left gripper open/close
- `C/V`: right gripper open/close

## Privileged closed-loop controller

The controller uses a validated expert trajectory as a continuous motion skeleton and
reads privileged simulator state at every 20 Hz physics step. Its feedback layers:

1. retarget the floating base to the live cutlery pose;
2. servo the left pinch point during the contact window;
3. apply a damped MuJoCo-Jacobian correction to the five left-arm targets in transport;
4. filter short contact-state dropouts;
5. gate and retry grasp, placement, drawer closing, tray closing, and door closing from
   physical task state.

Phase boundaries in `reference_milestones.json` were extracted from 30 demonstrations
that pass the current task predicate. Regenerate the file after changing simulator or
demonstration versions:

```bash
python -m tools.desktop_demo_recorder.extract_reference_milestones \
  --validation-manifest demos/DishwasherUnloadCutleryLong/privileged_batch/manifest.jsonl \
  --output tools/desktop_demo_recorder/reference_milestones.json
```

## Expanded randomization

The stock task selects one of two discrete cutlery sites and rotates it by up to 30
degrees. Each recorder reset keeps those physically supported regions but adds
continuous XY jitter, expands rotation to 40 degrees in either direction, and varies
the lower dishwasher tray from `0.95` to `1.0`. The door remains fully open because
values near `0.9` obstruct the lower tray and change the task's initial-state contract.

Randomization is deterministic for a given seed and is written into each saved
manifest. In the development validation, the controller completed 9 of 10 expanded
randomization episodes (`510100` through `510109`); one far-holder pose slipped during
transport. This is measured coverage, not a guarantee for arbitrary placements.

## Batch validation

Replay official actions through the current simulator and save only physically
successful demonstrations:

```bash
MUJOCO_GL=egl python -m tools.desktop_demo_recorder.collect_privileged_batch \
  --expert-demo-root "$HOME/.bigym/demonstrations/0.9.0"
```

The command writes a resumable `manifest.jsonl`, `run_summary.json`, and validated
`.safetensors` files. Run `smoke_ws.py` against a live server to verify the three camera
streams, manual controls, pause path, and save-success gate.

The D-Robotics mark in the operator UI is sourced from the
[D-Robotics developer site](https://developer.d-robotics.cc/) and remains the property
of its owner.
