# 评测

通过 HTTP 调用 VLA 推理服务,回灌 BiGym 环境算任务成功率,可录制回灌视频. 与数据生成共用 tasks.py / env_utils.py / dim_utils.py,保证喂给模型的 state 和写进数据集的 state 一致.

运行环境为 python 3.12(conda 环境 amd_bigym_plus312),需 opencv-python、requests、av.

## 评测流程

每个 episode:

1. reset 环境,取当前 obs 的 3 个相机图 + state.
2. state 经 dim_utils.pad_to_16 补到 16 维(3-DOF 任务在 base index 2 补 Z=0),与训练数据 state 同维同义.
3. POST 到 VLA 服务的 /process_frame,带 language prompt、16 维 state、3 张原始分辨率 PNG 图,取回 16 维 action chunk.
4. 从 chunk 取 replan_step 个 action 逐个执行;每个 action 先经 dim_utils.drop_z 去掉 3-DOF 任务 pad 的 Z(还原成 15 维),再 clip 到 action_space 边界,然后 env.step.
5. 检测成功(reward > 0.5 或 task_success),记录.

queue 空时重新请求(action chunk 周期 replan).

## 与训练数据的一致性

| 项 | 训练数据(生成端) | 评测端 |
|---|---|---|
| language prompt | TASKS[task]["prompt"] | 同源 TASKS[task]["prompt"] |
| state | pad_to_16 后 16 维 | pad_to_16 后 16 维 |
| action | pad_to_16 后 16 维 | VLA 出 16 维,drop_z 还原后 step |
| 相机 | head D455 + 双腕 D405,原始分辨率 | 同(发原始分辨率 PNG,不降采样) |

## 用法

先起好 VLA HTTP 服务(监听 /process_frame),再在 eval/ 目录下运行:

```
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python eval_bigym.py \
    --task ReachTargetDual \
    --base-url http://localhost:7891 \
    --n-episodes 50 \
    --output results/reach_dual.json \
    --video-dir videos/reach_dual \
    --tag lora
```

主要参数:

| 参数 | 作用 |
|---|---|
| --task <名> | 任务名(PascalCase,40 任务之一) |
| --base-url <url> | VLA HTTP 服务地址 |
| --n-episodes N | 评测 episode 数(默认 50) |
| --max-steps N | 覆盖 max_step_limit.yaml 的 per-task 步数上限 |
| --replan-step N | action chunk 执行多少步后 replan(默认 10) |
| --seed0 N | 起始随机种子(默认 0,每 episode 加 1) |
| --output <文件> | 结果 JSON 输出路径 |
| --video-dir <目录> | 回灌视频目录(不传则不录像) |
| --tag <标记> | 视频文件名标记(默认 eval) |
| --no-video | 不录回灌视频 |

每个任务的最大步数在 replay_generation/max_step_limit.yaml(评测自动读取).

## 回灌视频

指定 --video-dir 时录制. 每 episode 一个 mp4,先用 opencv mp4v 录制,结束后转码成 H.264 + yuv420p + baseline(浏览器/移动端通用可播放). 文件名格式:{任务}_{tag}_ep{NN}_{ok|no}.mp4,ok/no 表示该 episode 是否成功.
