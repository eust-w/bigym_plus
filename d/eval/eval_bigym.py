"""bigym_plus 评测客户端 (16 维统一版, 与 data_collection 生成对齐).

与 data_collection/generate_dataset.py 共用 d/dim_utils.py 的维度转换, 保证
生成-评测一致:
  - state: env 原始维度 (3-DOF 15 / 4-DOF 16) 经 pad_to_16 补到 16 维喂 VLA
    (训练数据 state 也是 pad_to_16 后的 16 维).
  - action: VLA 返回 16 维 action, 经 drop_z 去掉 3-DOF 任务的 Z (index 2)
    还原成 15 维再 env.step (3-DOF env 只收 15 维); 4-DOF 原样 step.

其余与 d/eval_bigym_plus.py 一致: D405/D455 相机, per-task maxstep (yaml),
HTTP /process_frame 协议 (兼容 OpenDM DM05 等 VLA 服务), 回灌视频录制.

产物: 全部落进单一输出目录 {output-dir}/{snake_case 任务名}/, 内含
全部 episode 的回灌视频 ({env}_{tag}_ep{NN}_{ok|no}.mp4) + 一个汇总 JSON
(results.json).

用法:
  python eval_bigym.py --task ReachTargetDual --base-url http://localhost:7891 \
      --output-dir eval_videos --n-episodes 50 --tag lora
  # 产物: eval_videos/reach_target_dual/  (视频 + results.json)
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import requests

# tasks/env_utils/dim_utils 在 ../replay_generation/, rollout_video 在同目录 (eval/)
_D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # d/
sys.path.insert(0, os.path.join(_D, "replay_generation"))
from tasks import TASKS, FREQ, resolve_task, get_maxstep, task_to_snake  # noqa: E402
from env_utils import build_env, get_state, CAMERAS  # noqa: E402
from dim_utils import pad_to_16, drop_z  # noqa: E402
from rollout_video import RolloutRecorder, transcode_to_h264, video_path  # noqa: E402


def encode_png(chw):
    """CHW RGB uint8 -> PNG bytes (原始分辨率, 与生成端写入一致)."""
    arr = np.asarray(chw, dtype=np.uint8).transpose(1, 2, 0)  # CHW->HWC RGB
    bgr = arr[:, :, ::-1]
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    return buf.tobytes()


def check_success(env, reward, info):
    if reward is not None and reward > 0.5:
        return True
    if info.get("task_success"):
        return True
    s = getattr(env, "success", None)
    if callable(s):
        try:
            return bool(s())
        except Exception:
            pass
    elif s:
        return bool(s)
    return False


def run_eval(task, base_url, n_episodes, max_steps, replan_step, seed0,
             out_dir, tag):
    prompt = TASKS[task]["prompt"]
    env = build_env(task, render_mode="rgb_array")
    mojo = env.action_mode._mojo
    recorder = RolloutRecorder(fps=float(FREQ))
    successes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        tmp_path = os.path.join(out_dir, f"_tmp_ep{ep:02d}.mp4")
        recorder.add(env.render(), tmp_path)  # 初始帧
        queue = []
        success = False
        step = -1
        for step in range(max_steps):
            if not queue:
                imgs = [encode_png(obs[f"rgb_{c[0]}"]) for c in CAMERAS]
                state = get_state(env.robot, obs, mojo)
                state16 = pad_to_16(state, task)  # 补到 16 维喂 VLA (与训练一致)
                try:
                    r = requests.post(
                        base_url + "/process_frame",
                        data={"text": prompt, "states": json.dumps(state16.tolist())},
                        files=[("image", im) for im in imgs],
                        timeout=120,
                    )
                    chunk = np.array(r.json()["response"], dtype=np.float32)  # 16 维 action chunk
                except Exception as e:
                    print(f"  ep{ep} step{step} err: {e}")
                    break
                queue = chunk[:replan_step].tolist()
            action16 = np.array(queue.pop(0), dtype=np.float32)
            action = drop_z(action16, task)  # 3-DOF: 16->15 (去 Z); 4-DOF: 原样
            # 裁剪到 action_space 边界 (BiGym env.step 对超界严格抛错)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            try:
                obs, reward, term, trunc, info = env.step(action)
            except Exception:
                break
            recorder.add(env.render(), tmp_path)
            if check_success(env, reward, info):
                success = True
                break
            if term or trunc:
                break
        recorder.close()
        final_path = video_path(out_dir, task, tag, ep, success)
        if os.path.exists(tmp_path):
            os.replace(tmp_path, final_path)
            transcode_to_h264(final_path, fps=float(FREQ))
        successes.append(success)
        print(f"  ep{ep}: {success} (steps={step + 1})")
    env.close()
    return sum(successes) / len(successes) if successes else 0.0, successes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS.keys()),
                    help="PascalCase env_class (canonical 任务名)")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--output-dir", required=True,
                    help="输出根目录; 脚本自动在其下建 snake_case 任务子目录 "
                         "(如 output_dir/dishwasher_load_plates/), 内含全部 "
                         "episode 回灌视频 + 汇总 results.json")
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="覆盖 max_step_limit.yaml 的 per-task maxstep")
    ap.add_argument("--replan-step", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--tag", default="eval", help="视频文件名 tag")
    args = ap.parse_args()

    task = resolve_task(args.task)
    out_dir = os.path.join(args.output_dir, task_to_snake(task))
    os.makedirs(out_dir, exist_ok=True)
    max_steps = args.max_steps if args.max_steps is not None else get_maxstep(task)
    ms_src = "override" if args.max_steps is not None else "yaml"
    print(f"[{task}] eval {args.base_url} eps={args.n_episodes} "
          f"max_steps={max_steps} ({ms_src}) replan={args.replan_step} "
          f"out={out_dir}")
    rate, succ = run_eval(
        task, args.base_url, args.n_episodes, max_steps, args.replan_step,
        args.seed0, out_dir, args.tag,
    )
    print(f"[{task}] === success rate: {rate * 100:.1f}% "
          f"({sum(succ)}/{len(succ)}) ===")
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "task": task, "base_url": args.base_url, "success_rate": rate,
                "successes": succ, "n_episodes": args.n_episodes,
                "replan_step": args.replan_step, "max_steps": max_steps,
            }, f, indent=2,
        )
    print(f"saved {results_path}")


if __name__ == "__main__":
    main()
