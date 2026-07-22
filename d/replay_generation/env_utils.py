"""bigym_plus env 构建 + state 提取 (D405/D455 相机, fovy=58, 20Hz).

采集传感器规格 (与 bigym_plus.md 对齐):
  手部相机 D405 (left_wrist / right_wrist): H=480 W=640 fovy=58
  头部相机 D455 (head):                     H=480 W=848 fovy=58
控制频率 20Hz. fovy 经 CameraConfig.fovy (bigym_plus 对 bigym 的扩展) 在
BiGymEnv._initialize_cameras 里写入 model.cam_fovy, 不改 h1.xml 物理默认 (60).
异分辨率 (480×640 与 480×848) 由 bigym 原生支持: obs_renderers 按 resolution
tuple 建 dict, 每个唯一分辨率一个独立 mujoco.Renderer (见 bigym_env._initialize_renderers).

state = floating base(动态 DOF) + arm(10) + gripper(2), 与 action 同构同序.
"""
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import TASKS, FREQ, resolve_task  # noqa: E402

# (env_camera_name, lerobot_feature_key, (H, W), fovy)
# head=D455 (480×848), wrists=D405 (480×640)
CAMERAS = [
    ("head", "cam_high", (480, 848), 58.0),
    ("left_wrist", "cam_left_wrist", (480, 640), 58.0),
    ("right_wrist", "cam_right_wrist", (480, 640), 58.0),
]

# 默认 D405/D455 原生分辨率; 评测若需降采样 (如 128×128) 传 cameras 覆盖.
DEFAULT_CAMERA_CONFIGS = None  # lazy build (需要 bigym import)


def _default_cameras():
    from bigym.utils.observation_config import CameraConfig
    return [
        CameraConfig(name=name, rgb=True, resolution=res, fovy=fovy)
        for name, _lr_key, res, fovy in CAMERAS
    ]


def build_env(
    task_key,
    control_frequency=FREQ,
    with_cameras=True,
    render_mode=None,
    cameras=None,
):
    """构建指定任务的 BiGym env.

    :param task_key: PascalCase env_class 或 snake_case (经 resolve_task 归一).
    :param control_frequency: 控制频率, 默认 20Hz.
    :param with_cameras: 是否挂观测相机 (D405/D455).
    :param render_mode: None / "rgb_array" (评测回灌视频用 rgb_array).
    :param cameras: 自定义 CameraConfig 列表; 默认 D405/D455 原生分辨率.
    """
    from bigym.action_modes import JointPositionActionMode
    from bigym.utils.observation_config import ObservationConfig

    info = TASKS[resolve_task(task_key)]
    m = importlib.import_module(info["env_module"])
    env_cls = getattr(m, info["env_class"])
    cams = cameras if cameras is not None else (_default_cameras() if with_cameras else [])
    return env_cls(
        action_mode=JointPositionActionMode(
            floating_base=True, absolute=True, floating_dofs=info["floating_dofs"]
        ),
        control_frequency=control_frequency,
        observation_config=ObservationConfig(cameras=cams),
        render_mode=render_mode,
    )


def get_state(robot, obs, mojo):
    """state = floating base(动态 DOF) + arm(10) + gripper(2). 维度随任务 (15 或 16).

    与 bigym_to_opendm.bigym_def.get_state 同实现, 保证训练/推理 state 对齐.
    """
    base = np.asarray(obs["proprioception_floating_base"], dtype=np.float32).reshape(-1)
    arm = np.asarray(
        [
            float(np.asarray(mojo.physics.bind(act.joint).qpos).flatten()[0])
            for act in robot.limb_actuators
        ],
        dtype=np.float32,
    )
    grip = np.asarray(robot.qpos_grippers, dtype=np.float32).reshape(-1)
    return np.concatenate([base, arm, grip])


if __name__ == "__main__":
    # smoke: 构建 ReachTargetDual, 打印相机分辨率 + fovy + state 维度
    env = build_env("ReachTargetDual")
    mojo = env.action_mode._mojo
    obs, _ = env.reset(seed=0)
    print("action_space:", env.action_space.shape)
    for name, lr_key, res, fovy in CAMERAS:
        img = obs[f"rgb_{name}"]
        cam_id = env._cameras_map[name][0]
        actual_fovy = float(mojo.model.cam_fovy[cam_id])
        print(f"  {name} ({lr_key}): obs={img.shape} expect=({3},{res[0]},{res[1]}) "
              f"fovy={actual_fovy} expect={fovy}")
    state = get_state(env.robot, obs, mojo)
    print("state:", state.shape, state[:5])
    env.close()
