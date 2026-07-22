# 数据生成

把 BiGym 的 40 个人类演示任务,从作者提供的轨迹数据回放渲染,输出 LeRobot Dataset v3 数据集. 相似任务合并成 4 个大组,每组一个数据集,组内多个任务用 lerobot 原生的语言指令(task)机制区分. 也支持按单个任务独立生成.

运行环境为 python 3.12 + lerobot 0.6.0(conda 环境 amd_bigym_plus312).

## 生成模式

| 模式 | 参数 | 产物 |
|---|---|---|
| 单任务 | --task <任务名> | 每个任务一个数据集,repo_id=bigym_plus/<任务 snake 名> |
| 组合并 | --group <组名\|all> | 每组一个数据集,repo_id=bigym_plus/<组名>,组内多任务用各自语言指令区分 |

40 任务定义在 tasks.py(任务名、语言指令、自由度、演示数,单一数据源). 4 个组定义在 groups.py:

| 组名 | 中文 | 任务数 | 演示数 | 原始维度 |
|---|---|---|---|---|
| reach | 简单到达 | 3 | 140 | 15 (3-DOF) |
| long_horizon | 长程复合 | 3 | 120 | 16 (4-DOF) |
| dishwasher | 洗碗机 | 10 | 504 | 16 (4-DOF) |
| tabletop | 桌面操作 | 24 | 1104 | 22 个 16 + 2 个 15 |
| 合计 | — | 40 | 1868 | — |

## 维度统一(全局 16 维)

state 与 action 同构,顺序为 base + 双臂关节 + 双夹爪:

| 区间 | 含义 | 维度 |
|---|---|---|
| base | 浮动基座 X, Y, Z, RZ | 3 或 4 |
| arm | 双臂关节角(左肩3 + 左肘 + 左腕 + 右同) | 10 |
| gripper | 左右夹爪开合 | 2 |

35 个任务是 4-DOF(维度 16,base 含 Z);5 个任务(ReachTarget、ReachTargetSingle、ReachTargetDual、MovePlate、MoveTwoPlates)是 3-DOF(维度 15,base 无 Z).

为让数据集维度一致、每个位置语义全局相同,统一对齐到 16 维,在 base 的 index 2 位补一个 Z,值固定 0.0(state 和 action 都用同一个值). 转换在 dim_utils.py,评测端共用同一模块.

回放时 env.step() 仍用原始维度(15 或 16)的 action,与任务环境匹配;16 维统一只作用于写进数据集的 state/action.

## 相机

3 个相机,控制频率 20 Hz:

| 相机 | 型号 | 分辨率 | fovy |
|---|---|---|---|
| 头部 (cam_high) | D455 | 480 x 848 | 58 |
| 左腕 (cam_left_wrist) | D405 | 480 x 640 | 58 |
| 右腕 (cam_right_wrist) | D405 | 480 x 640 | 58 |

## 用法

所有命令在 replay_generation/ 目录下执行,设好 CUDA_VISIBLE_DEVICES 和 MUJOCO_GL=egl.

下载演示数据(首次运行前,BiGym 官方轨迹 v0.9.0,缓存到 ~/.bigym):

```
python download_demos.py
```

验证管线(每个任务只跑 1 条演示):

```
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python generate_dataset.py --group reach --probe
```

单任务生成:

```
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python generate_dataset.py --task ReachTargetDual
```

单组全量:

```
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python generate_dataset.py --group reach
```

4 组全量:

```
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python generate_dataset.py --group all
```

限制每个任务的演示条数(调试):

```
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python generate_dataset.py --group all --amount 12
```

主要参数:

| 参数 | 作用 |
|---|---|
| --task <名> | 单任务模式(与 --group 互斥) |
| --group <组\|all> | 组合并模式 |
| --amount N | 每个任务的演示条数上限(默认全量) |
| --probe | 每任务只跑 1 条,验证管线 |
| --root <目录> | 数据集根目录(默认 data,相对当前目录) |
| --force | 删已有目录从头建 |
| --vcodec <编码> | 视频编码(默认 libsvtav1,不可用改 h264) |

断点续传:默认可恢复. 数据集目录已存在时自动用 LeRobotDataset.resume() 追加,配合 progress.json 跳过已写入的(任务,演示序号). 中断后重跑同一条命令即可接着采集. 想从头来加 --force(删整个目录重建).

## 产物结构

每组或每任务生成一个目录(默认在 replay_generation/data/ 下),内部是标准 LeRobot v3 布局:

```
data/<名>/
├── meta/
│   ├── info.json          数据集元信息 (fps, features, total_episodes)
│   ├── stats.json         state/action 的均值/方差/范围
│   ├── tasks.parquet      任务指令表 (task_index 对应语言指令)
│   └── episodes/          每 episode 元数据
├── data/                  state/action 等数值 (parquet 分块)
├── videos/                3 个相机的视频 (mp4 分块)
│   ├── observation.images.cam_high/
│   ├── observation.images.cam_left_wrist/
│   └── observation.images.cam_right_wrist/
└── progress.json          断点续传用 (已写入的任务/演示序号)
```
