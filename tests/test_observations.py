"""Test observations."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyquaternion import Quaternion

from bigym.action_modes import JointPositionActionMode
from bigym.envs.reach_target import ReachTarget

from bigym.utils.observation_config import CameraConfig, ObservationConfig


@pytest.mark.parametrize(
    "rgb",
    [True, False],
)
@pytest.mark.parametrize(
    "depth",
    [True, False],
)
@pytest.mark.parametrize(
    "resolution",
    [(4, 4), (8, 8)],
)
class TestObservations:
    """Test observations."""

    def test_visual_observations_exist(self, rgb, depth, resolution):
        """Test visual observations exist."""
        camera_name = "head"
        env = ReachTarget(
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            observation_config=ObservationConfig(
                cameras=[
                    CameraConfig(
                        name=camera_name,
                        rgb=rgb,
                        depth=depth,
                        resolution=resolution,
                    )
                ],
            ),
        )
        observation, _ = env.reset()
        assert (f"rgb_{camera_name}" in observation) == rgb
        if rgb:
            assert observation[f"rgb_{camera_name}"].shape == (3, *resolution)
        assert (f"depth_{camera_name}" in observation) == depth
        if depth:
            assert observation[f"depth_{camera_name}"].shape == resolution


def test_no_visual_observations():
    """Test no visual observations."""
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        observation_config=ObservationConfig(),
    )
    observation, _ = env.reset()
    assert not np.all([key.startswith("rgb_") for key in observation.keys()])
    assert not np.all([key.startswith("depth_") for key in observation.keys()])


def test_camera_pose_config():
    """Test camera pose is correctly set via observation config."""
    camera_name = "external"
    camera_pos = np.random.uniform(-1, 1, 3).tolist()
    camera_quat = Quaternion.random().elements.tolist()
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig(
                    name=camera_name,
                    pos=camera_pos,
                    quat=camera_quat,
                )
            ],
        ),
    )
    env.reset()
    camera = env._cameras_map[camera_name][1]
    assert_allclose(camera.get_position(), camera_pos)
    assert np.allclose(camera.get_quaternion(), camera_quat) or np.allclose(
        camera.get_quaternion(), np.negative(camera_quat)
    )


def test_camera_fovy_config():
    """Test camera field of view is set via observation config."""
    camera_name = "external"
    fovy = 55.0
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        observation_config=ObservationConfig(
            cameras=[CameraConfig(name=camera_name, fovy=fovy)]
        ),
    )

    camera_id = env._cameras_map[camera_name][0]
    assert env.mojo.model.cam_fovy[camera_id] == pytest.approx(fovy)


def test_large_camera_resolution_expands_offscreen_buffer():
    """Test observations can exceed MuJoCo's default offscreen width."""
    camera_name = "head"
    resolution = (32, 641)
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        observation_config=ObservationConfig(
            cameras=[CameraConfig(name=camera_name, resolution=resolution)]
        ),
    )

    observation, _ = env.reset()

    assert env.mojo.model.vis.global_.offheight >= resolution[0]
    assert env.mojo.model.vis.global_.offwidth >= resolution[1]
    assert observation[f"rgb_{camera_name}"].shape == (3, *resolution)
