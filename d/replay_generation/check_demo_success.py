"""bigym_plus 官方 demo 回灌 success 审计 (无渲染, fast=True 最快路径).

逐条 replay BiGym 官方 demo (demonstrations.zip v0.9.0, chernyadev VR 录制) 到
无相机 env, 用 env.reward > 0 判定 replay 后是否达到 task success, 统计 per-task
比例. 回答: 作者提供的轨迹里 success 到底多少.

为什么需要: generate_dataset.py replay demo 时**不 check success** (信任 demo 来源
success-gated), 但 mujoco EGL 下 seed+action 复现可能有微小数值漂移, 致某条 demo
replay 后实际未达 success 却被存进训练集. 本脚本量化这个比例.

判定口径 (三者等价, 已核查):
  env.reward > 0  ← _reward() = float(self.success)*1.0 (bigym_env.py:505-507), reward∈{0,1}
  ≡ eval check_success 主路径 reward>0.5 (eval_bigym.py:50, 因 reward∈{0,1})
  ≡ 官方 DemoPlayer.validate_in_env env.reward>0 (demo_player.py:60)

设计要点:
  - 无渲染: build_env(with_cameras=False, render_mode=None), 不调 env.render()
    (否则 BiGymRenderer 抛 AttributeError). 仅跑物理仿真.
  - 最快: env.step(action, fast=True) 不建 obs (返回 {},0,...); env.reward property
    从 _step_cache 读真实值 (bigym_env.py:445-450, _step_mujoco_simulation 在 fast
    判断前已执行).
  - env 复用: per-task 建一次, 循环该 task 全量 demo (不像 validate_in_env 每 demo
    close). 和 generate_dataset.py:138-144 同模式.
  - 取 demo 与 generate_dataset 同源: DemoStore().get_demos(Metadata.from_env(env,
    is_lightweight=True), amount, frequency=FREQ).

用法:
  MUJOCO_GL=egl python check_demo_success.py --task ReachTargetDual --probe  # 1 条验管线
  MUJOCO_GL=egl python check_demo_success.py --task ReachTargetDual          # 单任务全量
  MUJOCO_GL=egl python check_demo_success.py --group all                     # 全 40 任务
  MUJOCO_GL=egl python check_demo_success.py --group all --amount 12         # 每任务限 12
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import TASKS, FREQ, resolve_task, action_dim  # noqa: E402
from env_utils import build_env  # noqa: E402
from groups import GROUPS_DICT, all_group_keys  # noqa: E402

OUT_JSON = "/home/chao01.wu/BiGym_Dev/claude_analysis_notes/demo_success_audit.json"


def _select_tasks(args):
    """--task 单任务 / --group 组或 all / 默认全量 40."""
    if args.task:
        return [resolve_task(args.task)]
    keys = all_group_keys() if args.group == "all" else [args.group]
    tasks = []
    for k in keys:
        if k not in GROUPS_DICT:
            print(f"未知组: {k}, 可选 {all_group_keys()}")
            continue
        tasks.extend(GROUPS_DICT[k][1])
    return tasks


def _audit_task(task, amount, probe):
    """per-task: 建无相机 env, replay 全量 demo, 统计 success. 返回统计 dict."""
    from demonstrations.demo_store import DemoStore
    from demonstrations.utils import Metadata

    info = TASKS[task]
    dof = len(info["floating_dofs"])
    amt = 1 if probe else (info["n_demos"] if amount is None else min(amount, info["n_demos"]))
    env = build_env(task, with_cameras=False, render_mode=None)
    metadata = Metadata.from_env(env, is_lightweight=True)
    demos = DemoStore().get_demos(metadata, amount=amt, frequency=FREQ)
    n_total = len(demos)
    n_succ = n_err = n_no_succ = 0
    per_demo = []
    for di, demo in enumerate(demos):
        try:
            env.reset(seed=demo.seed)                      # 复现录制 seed
        except Exception as e:
            n_err += 1
            per_demo.append({"idx": di, "ok": False, "err": f"reset: {e}"})
            continue
        ok = False
        step_err = None
        for ts in demo.timesteps:
            try:
                env.step(ts.executed_action, fast=True)     # 无 obs, 最快; reward 仍可读
            except Exception as e:
                step_err = f"step: {e}"                    # action 越界/仿真崩
                break
            if env.reward > 0:                              # property 读 cache 真实值
                ok = True
                break
        if ok:
            n_succ += 1
        elif step_err:
            n_err += 1
        else:
            n_no_succ += 1                                 # replay 跑通但未 success
        per_demo.append({"idx": di, "ok": ok, "err": step_err or ("" if ok else "no_success")})
    env.close()
    rate = n_succ / n_total * 100 if n_total else 0.0
    return {
        "task": task, "dof": dof, "dim": action_dim(task),
        "n_total": n_total, "n_success": n_succ,
        "n_replay_errors": n_err, "n_no_success": n_no_succ,
        "rate": round(rate, 1), "per_demo": per_demo if probe else None,
    }


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--task", default=None, help="单任务名 (PascalCase 或 snake_case)")
    ap.add_argument("--group", default=None, help=f"组名 {all_group_keys()} 或 all")
    ap.add_argument("--amount", type=int, default=None, help="每任务 demo 上限 (默认全量)")
    ap.add_argument("--probe", action="store_true", help="每任务只跑 1 条 demo (验管线)")
    args = ap.parse_args()

    if args.task and args.group:
        ap.error("--task 与 --group 互斥, 只选其一")
    if not args.task and not args.group:
        args.group = "all"                                  # 默认全量 40 任务

    tasks = _select_tasks(args)
    if not tasks:
        ap.error("无任务可选 (检查 --task / --group)")

    print(f"\n=== demo success 审计: {len(tasks)} 任务 "
          f"(probe={args.probe}, amount={args.amount}, freq={FREQ}, 无渲染) ===")
    print(f"{'task':<30} {'dof':<4} {'succ/total':<14} {'rate%':<8} {'err':<5} {'no_succ':<8}")
    print("-" * 76)
    results = []
    t0 = time.time()
    for task in tasks:
        t_task = time.time()
        r = _audit_task(task, args.amount, args.probe)
        results.append(r)
        print(f"{r['task']:<30} {r['dof']:<4} {r['n_success']}/{r['n_total']:<11} "
              f"{r['rate']:<8} {r['n_replay_errors']:<5} {r['n_no_success']:<8}"
              f"  ({time.time() - t_task:.1f}s)")
    print("-" * 76)
    tot_succ = sum(r["n_success"] for r in results)
    tot = sum(r["n_total"] for r in results)
    tot_err = sum(r["n_replay_errors"] for r in results)
    tot_no = sum(r["n_no_success"] for r in results)
    print(f"合计 {len(results)} 任务 / {tot} demos: succ {tot_succ} "
          f"({tot_succ / tot * 100:.1f}%) err {tot_err} no_succ {tot_no} "
          f"({time.time() - t0:.1f}s)")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    payload = {
        "meta": {
            "conda": "amd_bigym_plus312", "freq": FREQ,
            "criterion": "env.reward > 0 (replay, fast=True, 无渲染)",
            "equiv_to": "eval check_success reward>0.5 / 官方 DemoPlayer.validate_in_env",
            "probe": args.probe, "amount": args.amount,
        },
        "tasks": results,
        "summary": {
            "n_tasks": len(results), "n_demos": tot, "n_success": tot_succ,
            "n_replay_errors": tot_err, "n_no_success": tot_no,
            "overall_rate": round(tot_succ / tot * 100, 1) if tot else 0.0,
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"saved {OUT_JSON}")


if __name__ == "__main__":
    main()
