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
import time
import uuid

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
from trajectory_log import EpisodeTrajectoryLog  # noqa: E402


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


def run_eval(task, base_url, n_episodes, max_steps, seed0,
             out_dir, tag, record_trajectories=False):
    prompt = TASKS[task]["prompt"]
    env = build_env(task, render_mode="rgb_array")
    mojo = env.action_mode._mojo
    recorder = RolloutRecorder(fps=float(FREQ))
    successes = []
    for ep in range(n_episodes):
        seed = seed0 + ep
        trajectory = None
        if record_trajectories:
            trajectory = EpisodeTrajectoryLog(
                os.path.join(out_dir, "trajectories", tag),
                episode_index=ep,
                seed=seed,
                task=task,
                metadata={"base_url": base_url, "prompt": prompt},
            )
        try:
            obs, reset_info = env.reset(seed=seed)
        except KeyboardInterrupt:
            if trajectory is not None:
                trajectory.append("interrupt", phase="reset")
                trajectory.finish(
                    status="interrupted",
                    success=False,
                    transitions=0,
                    failure_reason="keyboard_interrupt",
                )
            env.close()
            raise
        except Exception as e:
            if trajectory is not None:
                trajectory.append(
                    "reset_error",
                    error_type=type(e).__name__,
                    error=str(e),
                )
                trajectory.finish(
                    status="failed",
                    success=False,
                    transitions=0,
                    failure_reason="environment_reset_failed",
                )
            env.close()
            raise
        if trajectory is not None:
            reset_state = pad_to_16(get_state(env.robot, obs, mojo), task)
            trajectory.append(
                "reset",
                state16=reset_state,
                info=reset_info,
                camera_shapes={
                    camera[0]: list(np.asarray(obs[f"rgb_{camera[0]}"]).shape)
                    for camera in CAMERAS
                },
            )
        tmp_path = os.path.join(out_dir, f"_tmp_ep{ep:02d}.mp4")
        recorder.add(env.render(), tmp_path)  # 初始帧
        queue = []
        success = False
        step = -1
        transitions = 0
        failure_reason = None
        infrastructure_failed = False
        active_request_id = None
        active_policy_latency_ms = None
        active_server_request_id = None
        active_chunk_index = 0
        for step in range(max_steps):
            state = get_state(env.robot, obs, mojo)
            state16 = pad_to_16(state, task)
            if not queue:
                imgs = [encode_png(obs[f"rgb_{c[0]}"]) for c in CAMERAS]
                request_id = uuid.uuid4().hex
                request_started = time.perf_counter()
                try:
                    request_data = {
                        "text": prompt,
                        "states": json.dumps(state16.tolist()),
                    }
                    if trajectory is not None:
                        request_data["request_id"] = request_id
                    r = requests.post(
                        base_url + "/process_frame",
                        data=request_data,
                        files=[("image", im) for im in imgs],
                        timeout=120,
                    )
                    r.raise_for_status()
                    response_payload = r.json()
                    chunk = np.array(
                        response_payload["response"], dtype=np.float32
                    )  # 16 维 action chunk
                    if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != 16:
                        raise ValueError(
                            f"expected a non-empty Nx16 action chunk, got {chunk.shape}"
                        )
                except KeyboardInterrupt:
                    recorder.close()
                    if trajectory is not None:
                        trajectory.append(
                            "interrupt",
                            phase="policy_request",
                            environment_step=step,
                            request_id=request_id,
                        )
                        trajectory.finish(
                            status="interrupted",
                            success=False,
                            transitions=transitions,
                            failure_reason="keyboard_interrupt",
                        )
                    env.close()
                    raise
                except Exception as e:
                    print(f"  ep{ep} step{step} err: {e}")
                    failure_reason = "policy_request_failed"
                    infrastructure_failed = True
                    if trajectory is not None:
                        trajectory.append(
                            "policy_request_error",
                            environment_step=step,
                            state16=state16,
                            request_id=request_id,
                            client_latency_ms=(time.perf_counter() - request_started)
                            * 1000.0,
                            error_type=type(e).__name__,
                            error=str(e),
                        )
                    break
                active_request_id = request_id
                active_policy_latency_ms = (
                    time.perf_counter() - request_started
                ) * 1000.0
                active_server_request_id = response_payload.get("request_id")
                active_chunk_index = 0
                queue = chunk.tolist()  # server 侧已 slice; client 零 slice 逻辑, 执行返回的全部动作
            action16 = np.array(queue.pop(0), dtype=np.float32)
            action_unclipped = drop_z(action16, task)  # 3-DOF: 16->15 (去 Z); 4-DOF: 原样
            # 裁剪到 action_space 边界 (BiGym env.step 对超界严格抛错)
            action = np.clip(
                action_unclipped, env.action_space.low, env.action_space.high
            )
            clip_mask = np.not_equal(action, action_unclipped)
            try:
                obs, reward, term, trunc, info = env.step(action)
            except KeyboardInterrupt:
                recorder.close()
                if trajectory is not None:
                    trajectory.append(
                        "interrupt",
                        phase="environment_step",
                        environment_step=step,
                        state16_before=state16,
                        action16_model=action16,
                        action_env=action,
                    )
                    trajectory.finish(
                        status="interrupted",
                        success=False,
                        transitions=transitions,
                        failure_reason="keyboard_interrupt",
                    )
                env.close()
                raise
            except Exception as e:
                failure_reason = "environment_step_failed"
                infrastructure_failed = True
                if trajectory is not None:
                    trajectory.append(
                        "environment_step_error",
                        environment_step=step,
                        state16_before=state16,
                        action16_model=action16,
                        action_env=action,
                        request_id=active_request_id,
                        action_chunk_index=active_chunk_index,
                        error_type=type(e).__name__,
                        error=str(e),
                    )
                break
            transitions += 1
            state16_after = pad_to_16(get_state(env.robot, obs, mojo), task)
            recorder.add(env.render(), tmp_path)
            success = check_success(env, reward, info)
            if trajectory is not None:
                trajectory.append(
                    "transition",
                    environment_step=step,
                    state16_before=state16,
                    state16_after=state16_after,
                    action16_model=action16,
                    action_env_unclipped=action_unclipped,
                    action_env=action,
                    clip_mask=clip_mask,
                    reward=reward,
                    success=success,
                    terminated=term,
                    truncated=trunc,
                    info=info,
                    request_id=active_request_id,
                    server_request_id=active_server_request_id,
                    action_chunk_index=active_chunk_index,
                    client_policy_latency_ms=active_policy_latency_ms,
                )
            active_chunk_index += 1
            if success:
                break
            if term or trunc:
                failure_reason = "environment_terminated_without_success"
                break
        recorder.close()
        if not success and failure_reason is None:
            failure_reason = "max_steps_without_success"
        final_path = video_path(out_dir, task, tag, ep, success)
        if os.path.exists(tmp_path):
            os.replace(tmp_path, final_path)
            transcode_to_h264(final_path, fps=float(FREQ))
        if trajectory is not None:
            trajectory.finish(
                status="failed" if infrastructure_failed else "complete",
                success=success,
                transitions=transitions,
                failure_reason=failure_reason,
                rollout_video=os.path.basename(final_path),
            )
        successes.append(success)
        print(
            f"  ep{ep}: {success} (steps={step + 1}, "
            f"failure={failure_reason})"
        )
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
    # replan/slice 已移到推理 server (--inference-config.replan-step); bigym 零 slice,
    # server 返回几组动作就执行几组。
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--tag", default="eval", help="视频文件名 tag")
    ap.add_argument(
        "--record-trajectories",
        action="store_true",
        help="追加记录每一步的 state/action/reward/终止原因到 JSONL, "
             "并原子写入 episode manifest",
    )
    args = ap.parse_args()

    task = resolve_task(args.task)
    out_dir = os.path.join(args.output_dir, task_to_snake(task))
    os.makedirs(out_dir, exist_ok=True)
    max_steps = args.max_steps if args.max_steps is not None else get_maxstep(task)
    ms_src = "override" if args.max_steps is not None else "yaml"
    print(f"[{task}] eval {args.base_url} eps={args.n_episodes} "
          f"max_steps={max_steps} ({ms_src}) "
          f"out={out_dir}")
    rate, succ = run_eval(
        task, args.base_url, args.n_episodes, max_steps,
        args.seed0, out_dir, args.tag, args.record_trajectories,
    )
    print(f"[{task}] === success rate: {rate * 100:.1f}% "
          f"({sum(succ)}/{len(succ)}) ===")
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "task": task, "base_url": args.base_url, "success_rate": rate,
                "successes": succ, "n_episodes": args.n_episodes,
                "max_steps": max_steps,
                "trajectory_recording": {
                    "enabled": args.record_trajectories,
                    "schema_version": 1 if args.record_trajectories else None,
                    "path": (
                        os.path.join("trajectories", args.tag)
                        if args.record_trajectories else None
                    ),
                },
            }, f, indent=2,
        )
    print(f"saved {results_path}")


if __name__ == "__main__":
    main()
