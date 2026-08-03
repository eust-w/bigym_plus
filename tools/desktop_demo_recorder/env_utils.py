"""Environment construction and robot-state helpers for the desktop recorder."""

from __future__ import annotations

from typing import Any

import numpy as np

from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.envs.dishwasher_cutlery import DishwasherUnloadCutleryLong
from bigym.utils.observation_config import CameraConfig, ObservationConfig


SUPPORTED_TASK = "DishwasherUnloadCutleryLong"


def build_env(
    task: str,
    control_frequency: int = 20,
    with_cameras: bool = True,
) -> DishwasherUnloadCutleryLong:
    """Build the supported task with native D455/D405 image dimensions."""
    if task != SUPPORTED_TASK:
        raise ValueError(
            f"Desktop privileged collection currently supports {SUPPORTED_TASK}, "
            f"not {task}"
        )
    cameras = []
    if with_cameras:
        cameras = [
            CameraConfig(name="head", rgb=True, resolution=(480, 848), fovy=58.0),
            CameraConfig(
                name="external",
                rgb=True,
                resolution=(480, 848),
                pos=(0.0, -1.55, 2.55),
                fovy=58.0,
            ),
            CameraConfig(
                name="left_wrist", rgb=True, resolution=(480, 640), fovy=58.0
            ),
            CameraConfig(
                name="right_wrist", rgb=True, resolution=(480, 640), fovy=58.0
            ),
        ]
    return DishwasherUnloadCutleryLong(
        action_mode=JointPositionActionMode(
            floating_base=True,
            absolute=True,
            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
        ),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(cameras=cameras),
        render_mode=None,
    )


def get_state(robot: Any, observation: dict[str, Any], mojo: Any) -> np.ndarray:
    """Return floating base, ten arm joints, and two gripper positions."""
    base = np.asarray(
        observation["proprioception_floating_base"], dtype=np.float32
    ).reshape(-1)
    arms = np.asarray(
        [
            float(np.asarray(mojo.physics.bind(actuator.joint).qpos).reshape(-1)[0])
            for actuator in robot.limb_actuators
        ],
        dtype=np.float32,
    )
    grippers = np.asarray(robot.qpos_grippers, dtype=np.float32).reshape(-1)
    return np.concatenate([base, arms, grippers])
