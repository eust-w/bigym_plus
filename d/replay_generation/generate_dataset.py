"""bigym_plus demo -> LeRobot Dataset v3 (lerobot 0.6.0 / py3.12).

两种生成模式:
  --group <组名>  组合并模式: 每组一个 dataset (repo_id=bigym_plus/<组>), 组内多个
                  原始任务用各自的 language prompt 组织 (lerobot 自动建 tasks.parquet,
                  task_index -> prompt 区分). 4 组见 groups.py.
  --task <任务名> 单任务模式: 每个任务一个独立 dataset (repo_id=bigym_plus/<任务>),
                  40 个任务可单独生成.

维度全局统一 16 (base[X,Y,Z,RZ] + arm10 + gripper2). 3-DOF 任务 (ReachTarget /
ReachTargetSingle / ReachTargetDual / MovePlate / MoveTwoPlates, 共 15 维) 用
dim_utils.pad_to_16 在 base index 2 pad 一个 Z (值 0.0); 4-DOF (16 维) 原样.
state 与 action 都 pad. 转换逻辑与评测同源 (dim_utils.py), 保证生成-评测一致.

注意 env.step() 用原始维度 action (15 或 16, 匹配该任务 env); pad 只作用于写进
dataset 的 state/action.

断点续传: 目录已存在 (非 --force) 用 LeRobotDataset.resume() 追加, progress.json
记录已加的 (task, demo_idx), 重跑跳过, episode_index 由 lerobot 续编. --force 删
目录从头建.

用法:
  MUJOCO_GL=egl python generate_dataset.py --task ReachTargetDual --probe   # 单任务探针
  MUJOCO_GL=egl python generate_dataset.py --task ReachTargetDual           # 单任务全量
  MUJOCO_GL=egl python generate_dataset.py --group reach --probe            # 组探针
  MUJOCO_GL=egl python generate_dataset.py --group all                      # 4 组全量
  MUJOCO_GL=egl python generate_dataset.py --group all --amount 12          # 每任务限 12 条
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import TASKS, FREQ, resolve_task, action_dim, task_to_snake  # noqa: E402
from env_utils import build_env, get_state, CAMERAS  # noqa: E402
from dim_utils import pad_to_16, UNIFORM_DIM  # noqa: E402
from groups import GROUPS_DICT, repo_id, group_summary, all_group_keys  # noqa: E402


def build_features():
    """16 维统一 features: 3 个 video 相机 + state(16) + action(16)."""
    features = {}
    for _name, lr_key, res, _fovy in CAMERAS:
        features[f"observation.images.{lr_key}"] = {
            "dtype": "video",
            "shape": (3, res[0], res[1]),
            "names": ["channels", "height", "width"],
        }
    features["observation.state"] = {"dtype": "float32", "shape": (UNIFORM_DIM,), "names": None}
    features["action"] = {"dtype": "float32", "shape": (UNIFORM_DIM,), "names": None}
    return features


def _target(tasks, amount, probe):
    """目标 episode 数 (probe 每任务 1 条, 否则 min(amount, n_demos))."""
    total = 0
    for t in tasks:
        nd = TASKS[t]["n_demos"]
        total += 1 if probe else (nd if amount is None else min(amount, nd))
    return total


def _save_progress(path, done_set):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted([list(x) for x in done_set]), f)
    os.replace(tmp, path)


def _open_or_resume(rid, dataset_root, force, vcodec, image_writer_threads):
    """返回 (dataset, done_set, progress_path).
    force: 删目录重建 (create); 目录已存在: resume + 读 progress; 否则: create."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs import RGBEncoderConfig

    info_path = os.path.join(dataset_root, "meta", "info.json")
    progress_path = os.path.join(dataset_root, "progress.json")
    rgb_encoder = RGBEncoderConfig(vcodec=vcodec) if vcodec else None

    if force and os.path.exists(dataset_root):
        print(f"    --force: 删除已存在 {dataset_root}")
        shutil.rmtree(dataset_root)

    done_set = set()
    if os.path.exists(info_path) and not force:
        dataset = LeRobotDataset.resume(
            repo_id=rid, root=dataset_root, rgb_encoder=rgb_encoder,
            image_writer_threads=image_writer_threads,
        )
        if os.path.exists(progress_path):
            try:
                done_set = {tuple(x) for x in json.load(open(progress_path))}
            except Exception:
                done_set = set()
        print(f"    RESUME: 跳过 {len(done_set)} 条已加 demo")
    else:
        dataset = LeRobotDataset.create(
            repo_id=rid, fps=FREQ, features=build_features(), root=dataset_root,
            use_videos=True, image_writer_threads=image_writer_threads,
            rgb_encoder=rgb_encoder,
        )
    print(f"    dataset 就绪 (vcodec={vcodec or 'libsvtav1(default)'})")
    return dataset, done_set, progress_path


def _check_skip(dataset_root, target_total, probe):
    """resume 后若已达标且非 probe, 返回 True (SKIP)."""
    if probe:
        return False
    info_path = os.path.join(dataset_root, "meta", "info.json")
    if os.path.exists(info_path):
        try:
            done = json.load(open(info_path)).get("total_episodes", 0)
            if done >= target_total:
                print(f"    SKIP: 已有 {done} episodes (>= {target_total}), 用 --force 重跑")
                return True
        except Exception:
            pass
    return False


def _collect_tasks(dataset, tasks, amount, probe, done_set, progress_path, target_total, keep_failures=False):
    """把 tasks 列表里每个任务的 demo replay 进 dataset. 返回新增 episode 数."""
    from demonstrations.demo_store import DemoStore
    from demonstrations.utils import Metadata

    n_done = len(done_set)
    t_start = time.time()
    for ti, task in enumerate(tasks):
        info = TASKS[task]
        prompt = info["prompt"]
        amt = 1 if probe else (info["n_demos"] if amount is None else min(amount, info["n_demos"]))
        env = build_env(task)
        mojo = env.action_mode._mojo
        metadata = Metadata.from_env(env, is_lightweight=True)
        demos = DemoStore().get_demos(metadata, amount=amt, frequency=FREQ)
        print(f"  [{ti + 1}/{len(tasks)}] {task}: {len(demos)} demos, "
              f"dim={action_dim(task)}, prompt={prompt!r}")
        for di, demo in enumerate(demos):
            if (task, di) in done_set:
                continue
            t_ep = time.time()
            try:
                actions = [np.asarray(ts.executed_action, dtype=np.float32)
                           for ts in demo.timesteps]
            except Exception as e:
                print(f"    demo {di}: actions fail {e}")
                continue
            T = len(actions)
            if T < 2:
                continue
            try:
                obs, _ = env.reset(seed=getattr(demo, "seed", None))
            except Exception as e:
                print(f"    demo {di}: reset fail {e}")
                continue
            for t in range(T - 1):  # 丢最后一帧 (无 next obs)
                state = get_state(env.robot, obs, mojo)
                raw_action = actions[t]
                frame = {
                    "observation.images.cam_high": obs["rgb_head"],
                    "observation.images.cam_left_wrist": obs["rgb_left_wrist"],
                    "observation.images.cam_right_wrist": obs["rgb_right_wrist"],
                    "observation.state": pad_to_16(state, task),
                    "action": pad_to_16(raw_action, task),
                    "task": prompt,
                }
                dataset.add_frame(frame)
                obs, _r, _term, _trunc, _info = env.step(raw_action)  # 原始维度 (15/16)
            # replay success 过滤 (口径同 check_demo_success: env.reward>0, reward∈{0,1})
            if not keep_failures and env.reward <= 0:
                print(f"    demo {di}: replay 未达 success (reward={env.reward}), 丢弃")
                try:
                    dataset.clear_episode_buffer()  # 丢已 add 帧 + 删临时图片, 不留痕迹
                except Exception as e:
                    print(f"    demo {di}: clear_episode_buffer fail {e}")
                done_set.add((task, di))  # 标已判定, resume 不重判 (同环境重放确定性)
                _save_progress(progress_path, done_set)
                continue
            try:
                dataset.save_episode()
            except Exception as e:
                print(f"    demo {di}: save_episode fail {e}")
                try:
                    dataset.clear_episode_buffer()  # 防残留缓冲污染下一条 demo
                except Exception:
                    pass
                continue
            n_done += 1
            done_set.add((task, di))
            _save_progress(progress_path, done_set)
            dt = time.time() - t_ep
            eta = (time.time() - t_start) / max(n_done, 1) * max(target_total - n_done, 0)
            print(f"    demo {di}: {T - 1} frames {dt:.1f}s "
                  f"(累计 {n_done}/{target_total}, eta {eta / 60:.1f}min)")
            if probe:
                break
        env.close()
    return n_done


def _run(rid, dataset_root, tasks, label, n_demos_total, amount, probe, force,
         vcodec, image_writer_threads, keep_failures=False):
    """通用采集入口: 开 dataset -> (skip 检查) -> collect -> finalize."""
    target_total = _target(tasks, amount, probe)
    print(f"\n=== {label}: {len(tasks)} 任务 / {n_demos_total} demos ===")
    print(f"    repo_id={rid}  root={dataset_root}  target_episodes={target_total}")
    dataset, done_set, progress_path = _open_or_resume(
        rid, dataset_root, force, vcodec, image_writer_threads)
    if _check_skip(dataset_root, target_total, probe):
        dataset.finalize()
        return
    n = _collect_tasks(dataset, tasks, amount, probe, done_set, progress_path, target_total, keep_failures=keep_failures)
    dataset.finalize()
    print(f"    DONE {label}: {n} episodes -> {dataset_root}")


def gen_group(group_key, root, amount=None, probe=False, force=False,
              vcodec=None, image_writer_threads=4, keep_failures=False):
    name, tasks = GROUPS_DICT[group_key]
    _, _n_tasks, n_demos_total, dim_dist = group_summary(group_key)
    label = f"组 [{group_key}] {name} (维度分布 {dim_dist})"
    _run(repo_id(group_key), os.path.join(root, group_key), tasks,
         label, n_demos_total, amount, probe, force, vcodec, image_writer_threads,
         keep_failures=keep_failures)


def gen_task(task_key, root, amount=None, probe=False, force=False,
             vcodec=None, image_writer_threads=4, keep_failures=False):
    task = resolve_task(task_key)
    snake = task_to_snake(task)
    label = f"单任务 [{task}] (dim={action_dim(task)})"
    _run(f"bigym_plus/{snake}", os.path.join(root, snake), [task],
         label, TASKS[task]["n_demos"], amount, probe, force, vcodec, image_writer_threads,
         keep_failures=keep_failures)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--group", default=None, help=f"组名 {all_group_keys()} 或 all")
    ap.add_argument("--task", default=None, help="单任务名 (PascalCase 或 snake_case)")
    ap.add_argument("--amount", type=int, default=None, help="每个任务的 demo 上限 (默认全量)")
    ap.add_argument("--probe", action="store_true", help="每个任务只跑 1 条 demo (验证管线)")
    ap.add_argument("--root", default="data", help="数据集根目录 (相对 cwd)")
    ap.add_argument("--vcodec", default=None, help="视频编码 (默认 libsvtav1; 不可用改 h264)")
    ap.add_argument("--force", action="store_true", help="删除已存在目录, 从头重建")
    ap.add_argument("--image-writer-threads", type=int, default=4)
    ap.add_argument("--keep-failures", action="store_true",
                    help="不过滤 replay 失败的 demo (默认丢弃未达 success 的; 加此开关保留旧行为)")
    args = ap.parse_args()

    if args.task and args.group:
        ap.error("--task 与 --group 互斥, 只选其一")
    if not args.task and not args.group:
        ap.error("必须指定 --task <任务名> 或 --group <组名|all>")

    kw = dict(root=args.root, amount=args.amount, probe=args.probe, force=args.force,
              vcodec=args.vcodec, image_writer_threads=args.image_writer_threads,
              keep_failures=args.keep_failures)
    if args.task:
        gen_task(args.task, **kw)
    else:
        keys = all_group_keys() if args.group == "all" else [args.group]
        for k in keys:
            if k not in GROUPS_DICT:
                print(f"未知组: {k}, 可选 {all_group_keys()}")
                continue
            gen_group(k, **kw)


if __name__ == "__main__":
    main()
