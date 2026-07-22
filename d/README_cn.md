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

## 快速上手

数据生成(在 replay_generation/ 下):

```
python download_demos.py
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python generate_dataset.py --group reach --probe
```

评测(在 eval/ 下,先起好 VLA HTTP 服务):

```
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python eval_bigym.py \
    --task ReachTargetDual --base-url http://localhost:7891 \
    --n-episodes 50 --output results/reach_dual.json
```
