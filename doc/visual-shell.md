# BiGym 3DGS kitchen visual shell

The optional scene shell wraps an existing BiGym/MuJoCo task with a realistic
room background. MuJoCo still owns the robot, table, props, collisions, contacts,
task state, and simulation clock. A DISCOVERSE-compatible native Gaussian
renderer draws only the static kitchen surroundings, then MuJoCo segmentation
places the existing task geometry back in front.

The release contains three finished PLY layers:

- `walls_fixed_kitchen.ply`: walls, windows, fixed kitchen facilities, and
  perimeter decoration;
- `floor_perimeter.ply`: photographed perimeter floor;
- `ceiling_lights.ply`: ceiling and fixed lights.

It intentionally contains no source image, source video, combined duplicate
PLY, OBJ, collision mesh, robot, table, or central furniture.

## 1. Requirements

The base simulator remains unchanged. Native 3DGS rendering additionally
requires:

- Linux with an NVIDIA CUDA GPU (validated on one A800);
- a CUDA-compatible PyTorch installation;
- `gaussian_renderer==0.2.0`.

Install the optional dependency after selecting the correct PyTorch/CUDA build:

```bash
pip install -e '.[visual-shell]'
```

The A800 result validates the CUDA path only; it is not evidence for AMD
ROCm/Radeon compatibility.

## 2. One-click install and verification

This command downloads the versioned release once, verifies the archive and
every contained file, rejects unsafe archive paths, and caches the result under
`~/.bigym/visual_shells/`:

```bash
python -m bigym.utils.visual_shell_assets install dl3dv-kitchen-mujoco
```

Available names and an installed bundle can be inspected with:

```bash
python -m bigym.utils.visual_shell_assets list
python -m bigym.utils.visual_shell_assets verify \
  ~/.bigym/visual_shells/dl3dv-kitchen-v1
```

Set `BIGYM_VISUAL_SHELL_CACHE=/large/local/path` to move the cache. For an
air-gapped A800 node, copy the unchanged release bundle to the node, set
`BIGYM_VISUAL_SHELL_BUNDLE=/path/dl3dv-kitchen-shell-v1.tar.gz`, and run the same
install command. The bundle must still match the pinned size and SHA-256.

Pinned release bundle:

- bytes: `226124687`;
- SHA-256:
  `bc72810aca750b3392e416ba696b81258218bbbcd54886cf1375da22e9b0dc26`.

## 3. One-line loading in BiGym H1

Pass the registered name to any `BiGymEnv` subclass. First use installs the
asset automatically:

```python
from bigym.action_modes import JointPositionActionMode
from bigym.envs.dishwasher_plates import DishwasherLoadPlates
from bigym.utils.observation_config import CameraConfig, ObservationConfig

cameras = [
    CameraConfig("head", rgb=True, resolution=(480, 848), fovy=58.0),
    CameraConfig("left_wrist", rgb=True, resolution=(480, 640), fovy=58.0),
    CameraConfig("right_wrist", rgb=True, resolution=(480, 640), fovy=58.0),
]

env = DishwasherLoadPlates(
    action_mode=JointPositionActionMode(
        floating_base=True,
        absolute=True,
    ),
    control_frequency=20,
    observation_config=ObservationConfig(cameras=cameras),
    visual_shell_profile="dl3dv-kitchen-bigym-h1",
    visual_shell_strict=True,
)

observations, info = env.reset(seed=0)
assert observations["rgb_head"].shape == (3, 480, 848)
assert observations["rgb_left_wrist"].shape == (3, 480, 640)
assert observations["rgb_right_wrist"].shape == (3, 480, 640)
```

The `d/replay_generation` helper exposes the same switch:

```python
from d.replay_generation.env_utils import build_env

env = build_env(
    "DishwasherLoadPlates",
    visual_shell_profile="dl3dv-kitchen-bigym-h1",
)
```

Keep `visual_shell_strict=True` for formal collection. A renderer or asset error
then stops the run instead of silently mixing native and Gaussian backgrounds.
Omitting `visual_shell_profile` preserves the original behavior.

## 4. Choose the correct profile

| Profile | Intended camera | Current quality status |
| --- | --- | --- |
| `dl3dv-kitchen-mujoco` | Cameras kept inside the captured trajectory | Reconstruction passed; visual review still required |
| `dl3dv-kitchen-bigym-h1` | Initial H1 head and both wrist cameras | Diagnostic: visibly softer because the cameras extrapolate from the capture path |

The MCMC reconstruction itself reached PSNR 35.16, SSIM 0.959, and LPIPS 0.131
at 30,000 steps. Those held-out metrics do not override the profile-specific
camera coverage. The H1 head camera is about 0.247 m and 63.5 degrees from its
nearest captured view; therefore this profile is not marked
formal-collection-ready.

## 5. Use with another MuJoCo scene

A Gaussian shell can wrap another MJCF only after that scene receives its own
alignment. Do not reuse the H1 matrix blindly.

1. Keep the existing MJCF, bodies, geoms, and physics unchanged.
2. Identify the MuJoCo floor plane, task center, forward direction, and the
   cameras that will collect data.
3. Fit an upright 7-DoF Sim(3) from the Gaussian frame to those anchors.
4. Write both `gaussian_to_mujoco` and `mujoco_to_gaussian` 4x4 matrices to a
   new alignment JSON.
5. Copy a profile JSON, point `alignment.path` at the new file, and list native
   floor geom names in `integration_contract.native_background_geom_names`.
6. Pass the local profile path as `visual_shell_profile=/absolute/profile.json`.
7. Verify head/wrist cameras against the captured camera envelope before
   collecting formal data.

The Gaussian layers remain renderer assets, not MJCF assets. `GaussianVisualShell`
checks that the static shell was not mapped to a MuJoCo body.

## 6. Compositing and ground behavior

For every RGB observation:

1. MuJoCo renders the native scene and a segmentation image.
2. The MuJoCo camera pose is transformed into the Gaussian coordinate frame.
3. The native renderer draws the three static PLY layers.
4. Pixels belonging to the robot, table, props, and task geometry overwrite the
   background.
5. The H1 profile neutralizes visible blue native-floor leakage without adding
   collision geometry.

Depth observations, actions, state dimensions, contacts, and task success logic
are unchanged. The visual shell contributes zero bodies, geoms, collisions, and
contacts.

## 7. Troubleshooting

- `Gaussian visual shell requires CUDA`: run on the CUDA worker and verify the
  installed PyTorch build with `python -c 'import torch; print(torch.cuda.is_available())'`.
- `bundle SHA-256 mismatch`: delete neither the source repository nor other
  caches. Move only the named invalid cache aside and reinstall from the pinned
  release.
- Background is soft or smeared: the camera is outside the captured trajectory.
  Re-align the scene or acquire a shell whose camera path covers the target
  head/wrist poses; increasing render resolution cannot recover missing views.
- Robot/table disappears behind the room: ensure the MuJoCo geom appears in the
  segmentation image and is not listed as a native background geom.
- Formal collection falls back to plain MuJoCo: keep strict mode enabled; any
  fallback is a collection failure.

## 8. Provenance and license

The shell was reconstructed from DL3DV-10K scene
`951f9db189a7023708b7798e147e04048a84ce039c5761e8ecb1aa65dcb2da86`
at dataset revision
`abb4dab0d4b6d93c32e6d901c06c35bad03210fb`. The release does not contain the
original photographs or video.

Use is subject to the
[DL3DV-10K Terms of Use](https://github.com/DL3DV-10K/Dataset/blob/main/DL3DV-10K_term_of_use.pdf)
and [DL3DV license notice](https://github.com/DL3DV-10K/Dataset/blob/main/License.md).
Review and accept those terms before using the derived asset. The shell is
provided only for non-commercial research and no additional source-data rights
are granted.
