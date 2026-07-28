# bigym_plus/d

把 BiGym 的 40 个人类演示任务回放渲染成 LeRobot Dataset v3 数据集,并提供配套的 HTTP 评测客户端. 数据生成与评测共用同一套任务定义、环境构建和维度对齐逻辑,保证训练数据与推理输入一致.

## 目录结构

| 目录 | 作用 |
|---|---|
| replay_generation/ | 数据生成:回放 BiGym 演示,输出 LeRobot Dataset v3 |
| eval/ | 评测:调 HTTP 推理服务,回灌 BiGym 环境算成功率 |

replay_generation/tasks.py 是 40 任务的单一数据源(任务名、语言指令、自由度、演示数),replay_generation/env_utils.py 提供环境构建与 state 提取,replay_generation/dim_utils.py 提供 16 维统一转换. 评测端 import 这三个模块,与生成端同源,改一处两边对齐.

## 运行环境

数据生成与评测共用单一 conda 环境:

| 项 | 值 |
|---|---|
| 环境名 | amd_bigym_plus312 |
| python | 3.12.13 |
| lerobot | 0.6.0 |
| bigym | 4.1.0 |
| mujoco | 3.10.0 |
| torch | 2.11.0 + cu128 |

bigym 以 editable 模式指向本仓库 fork(含相机 fovy 扩展). 渲染需 EGL,运行前设 MUJOCO_GL=egl 并指定 GPU 卡(CUDA_VISIBLE_DEVICES). 完整依赖清单见 requirements.txt.

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

3 个相机,控制频率 20 Hz:

| 相机 | 型号 | 分辨率 | fovy |
|---|---|---|---|
| 头部 (cam_high) | D455 | 480 x 848 | 58 |
| 左腕 (cam_left_wrist) | D405 | 480 x 640 | 58 |
| 右腕 (cam_right_wrist) | D405 | 480 x 640 | 58 |

## 数据生成

把 BiGym 的 40 个人类演示任务,从作者提供的轨迹数据回放渲染,输出 LeRobot Dataset v3 数据集. 相似任务可合并成 4 个大组,每组一个数据集,组内多个任务用 lerobot 原生的语言指令(task)机制区分;也支持按单个任务独立生成.

### 生成模式

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

### replay 成功率过滤

BiGym 官方演示回放时并非条条成功(整体约 72%,各任务不均). generate_dataset.py 默认只保留回放成功的轨迹:每条 demo 重放完后读 env.reward,reward<=0(未达任务成功)的 demo 直接丢弃,不写入数据集(用 lerobot 原生的 clear_episode_buffer 清掉已缓冲的帧和临时图片,不留痕迹). 判定口径与 check_demo_success.py 审计脚本一致(reward∈{0,1}).

加 --keep-failures 可关闭过滤,保留所有 demo(含回放失败的,旧行为),用于对照或排查.

### 用法

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
| --keep-failures | 不过滤回放失败的 demo(默认丢弃未达 success 的) |

断点续传:默认可恢复. 数据集目录已存在时自动用 LeRobotDataset.resume() 追加,配合 progress.json 跳过已写入的(任务,演示序号). 中断后重跑同一条命令即可接着采集. 想从头来加 --force(删整个目录重建).

### 产物结构

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

## 评测

通过 HTTP 调用 VLA 推理服务, 回灌 BiGym 环境算任务成功率, 并录制回灌视频. 与数据生成共用 tasks.py / env_utils.py / dim_utils.py, 保证喂给模型的 state 和写进数据集的 state 一致.

### 评测流程

评测客户端只负责发请求和回灌环境, VLA 推理(含 action chunk 的 slice/replan)由外部推理服务完成, 客户端零 slice 逻辑. 推理服务需另行启动, 监听 /process_frame 并返回 action chunk.

每个 episode:

1. reset 环境, 取当前 obs 的 3 个相机图 + state.
2. state 经 dim_utils.pad_to_16 补到 16 维(3-DOF 任务在 base index 2 补 Z=0), 与训练数据 state 同维同义.
3. POST 到推理服务的 /process_frame, 带 language prompt、16 维 state、3 张原始分辨率 PNG 图, 取回 action chunk.
4. 执行服务返回的全部动作(服务侧已按 replan 截断); 每个动作先经 dim_utils.drop_z 去掉 3-DOF 任务 pad 的 Z(还原成 15 维), 再 clip 到 action_space 边界, 然后 env.step. 动作队列空时重新请求.
5. 检测成功(reward > 0.5 或 task_success), 记录.

### 与训练数据的一致性

| 项 | 训练数据(生成端) | 评测端 |
|---|---|---|
| language prompt | TASKS[task]["prompt"] | 同源 TASKS[task]["prompt"] |
| state | pad_to_16 后 16 维 | pad_to_16 后 16 维 |
| action | pad_to_16 后 16 维 | VLA 出 16 维, drop_z 还原后 step |
| 相机 | head D455 + 双腕 D405, 原始分辨率 | 同(发原始分辨率 PNG, 不降采样) |

### 用法

先起好 VLA HTTP 服务(监听 /process_frame), 再在 eval/ 目录下运行. 产物落进 {output-dir}/{snake 任务名}/, 含全部 episode 回灌视频 + 一个 results.json:

```
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python eval_bigym.py \
    --task ReachTargetDual \
    --base-url http://localhost:7891 \
    --output-dir eval_videos \
    --n-episodes 50 \
    --tag lora
```

主要参数:

| 参数 | 作用 |
|---|---|
| --task <名> | 任务名(PascalCase, 40 任务之一) |
| --base-url <url> | VLA HTTP 服务地址 |
| --output-dir <目录> | 输出根目录; 脚本自动在其下建 snake_case 任务子目录, 放全部视频 + results.json |
| --n-episodes N | 评测 episode 数(默认 50) |
| --max-steps N | 覆盖 max_step_limit.yaml 的 per-task 步数上限 |
| --seed0 N | 起始随机种子(默认 0, 每 episode 加 1) |
| --tag <标记> | 视频文件名标记(默认 eval) |

action chunk 的 replan/slice 不在客户端控制, 由推理服务侧的配置决定(如 --inference-config.replan-step N). 每个任务的最大步数从 replay_generation/max_step_limit.yaml 读取(单源, maxstep = ceil(任务最长 20Hz 演示步数 × 1.5)).

### 产物

全部落进单一目录 {output-dir}/{snake 任务名}/, 上面命令产出 eval_videos/reach_target_dual/:

```
eval_videos/reach_target_dual/
├── ReachTargetDual_lora_ep00_ok.mp4
├── ReachTargetDual_lora_ep01_no.mp4
├── ...
├── ReachTargetDual_lora_ep49_no.mp4
└── results.json
```

- 回灌视频: 每 episode 一个 mp4, 文件名 {env}_{tag}_ep{NN}_{ok|no}.mp4, ok/no 标该 episode 成功/失败. 录的是渲染相机(第三方全局视角), 不同于喂模型的 head/双腕三个观测相机; 先用 opencv mp4v 录制, 结束转码成 H.264 + yuv420p + baseline(浏览器/系统预览/移动端通用可播放).
- results.json: 汇总结果, 含 task / base_url / success_rate / successes(每 episode 布尔) / n_episodes / max_steps.
