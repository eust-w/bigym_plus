"""下载 BiGym 官方轨迹数据 (demonstrations.zip v0.9.0) 到 ~/.bigym 缓存.

用 bigym 内置 DemoStore.pull_demos (wget + 解压 + .lock), 与 DemoStore.get_demos
缓存路径完全对齐. 原始数据 500Hz 录制, get_demos(frequency=20) 时由 DemoConverter.decimate
降到 20Hz.

下载地址 (bigym/const.py):
  https://github.com/chernyadev/bigym_data/releases/download/v0.9.0/demonstrations.zip

用法:
  python download_demos.py            # 下载 (已缓存则跳过)
  python download_demos.py --force    # 删 .lock 强制重下
  python download_demos.py --verify   # 仅统计每任务 demo 数, 不下载
"""
import argparse
import glob
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify(cache_dir):
    from tasks import TASKS
    print(f"{'env_class':<30} {'dof':<5} {'n_demos'}")
    print("-" * 50)
    total = 0
    for k in sorted(TASKS):
        info = TASKS[k]
        pattern = os.path.join(
            cache_dir, info["env_class"],
            "JointPositionActionMode_*_absolute", "lightweight", "*.safetensors",
        )
        n = len(glob.glob(pattern))
        total += n
        dof = len(info["floating_dofs"])
        print(f"{k:<30} {dof:<5} {n}")
    print("-" * 50)
    print(f"total: {total} demos across {len(TASKS)} tasks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="删 .lock 强制重下")
    ap.add_argument("--verify", action="store_true", help="仅统计 demo 数, 不下载")
    args = ap.parse_args()

    from bigym.const import CACHE_PATH, DEMO_VERSION
    cache = CACHE_PATH / "demonstrations" / DEMO_VERSION
    lock = cache / ".lock"

    if args.verify:
        verify(str(cache))
        return

    from demonstrations.demo_store import DemoStore
    if args.force and lock.exists():
        lock.unlink()
        print(f"removed {lock} → 将重新下载")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ds = DemoStore()
    print(f"cache dir: {cache}")
    print(f"already cached: {ds.cached}")
    if ds.cached and not args.force:
        print("已缓存, 跳过下载 (用 --force 强制重下, --verify 统计)")
    ds.pull_demos()  # cached 时 no-op
    print(f"done. cached={ds.cached}")
    verify(str(cache))


if __name__ == "__main__":
    main()
