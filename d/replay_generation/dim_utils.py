"""bigym_plus 统一维度转换 (数据生成与评测共用, 保证两者一致).

所有任务对齐到 16 维: base[X, Y, Z, RZ] + arm(10) + gripper(2).
3-DOF 任务 (ReachTarget / ReachTargetSingle / ReachTargetDual / MovePlate /
MoveTwoPlates, base 只有 X, Y, RZ 共 15 维) 在 base 的 index 2 位 pad 一个 Z,
pad 值固定 0.0 (state 和 action 都用同一个值). 4-DOF 任务 (16 维) 原样.

数据生成 (data_collection/generate_dataset.py):
  state 和 action 都用 pad_to_16 写进 dataset, 一律 16 维.

评测 (d/eval/eval_bigym.py):
  env 给的原始 state (15 或 16 维) 用 pad_to_16 补到 16 维喂 VLA;
  VLA 返回的 16 维 action 用 drop_z 去掉 index 2 的 Z (仅 3-DOF 任务),
  再 env.step (3-DOF env 只收 15 维). 4-DOF 任务 action 原样 step.

生成与评测 import 同一个本模块, pad 值与转换逻辑同源, 改一处两边对齐.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import action_dim  # noqa: E402  单一数据源 (d/tasks.py)

UNIFORM_DIM = 16          # base(4: X,Y,Z,RZ) + arm(10) + gripper(2)
BASE_Z_INDEX = 2          # Z 在 base 内的位置 (统一后 base = [X, Y, Z, RZ])
Z_PAD_VALUE = 0.0         # 3-DOF pad Z 的固定值, 生成与评测一致


def is_3dof(task_key):
    """3-DOF 任务 (base 无 Z, 共 15 维). 其余 35 个任务 4-DOF (16 维)."""
    return action_dim(task_key) == 15


def pad_to_16(vec, task_key):
    """15 维 -> 16 维: base[x, y, rz] 在 index 2 插 Z_PAD_VALUE -> [x, y, z, rz].
    16 维 (4-DOF) 原样返回. state 和 action 都用此函数."""
    vec = np.asarray(vec, dtype=np.float32)
    if not is_3dof(task_key):
        return vec
    base3 = vec[:3]               # [x, y, rz]
    rest = vec[3:]                # arm(10) + gripper(2)
    base4 = np.concatenate([base3[:2], [Z_PAD_VALUE], base3[2:]]).astype(np.float32)
    return np.concatenate([base4, rest])


def drop_z(action16, task_key):
    """16 维 -> 15 维: 去掉 base 的 index 2 (Z) -> [x, y, rz] + arm + gripper.
    仅 3-DOF 评测 env.step 用 (3-DOF env 只收 15 维). 4-DOF 原样返回."""
    a = np.asarray(action16, dtype=np.float32)
    if not is_3dof(task_key):
        return a
    base4 = a[:4]                 # [x, y, z, rz]
    rest = a[4:]                  # arm(10) + gripper(2)
    base3 = np.delete(base4, BASE_Z_INDEX)   # [x, y, rz]
    return np.concatenate([base3, rest])


if __name__ == "__main__":
    # 自检: 3-DOF pad/drop 往返一致, 4-DOF 原样
    for t in ["ReachTarget", "DishwasherOpen"]:
        d = action_dim(t)
        print(f"{t} dim={d} 3dof={is_3dof(t)}")
        if d == 15:
            v = np.arange(15, dtype=np.float32)
            p = pad_to_16(v, t)
            print(f"  pad_to_16: Z@index2={p[2]} (应 0.0), 往返一致={np.allclose(drop_z(p, t), v)}")
        else:
            v = np.arange(16, dtype=np.float32)
            print(f"  pad_to_16 原样={np.allclose(pad_to_16(v, t), v)}, drop_z 原样={np.allclose(drop_z(v, t), v)}")
