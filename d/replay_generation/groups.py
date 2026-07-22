"""bigym_plus 数据采集 · 任务分组 (4 组, 分组方案来自 select.md).

每组生成一个 LeRobot Dataset v3:
  repo_id    = bigym_plus/<group_key>
  数据目录   = <root>/<group_key>
组内多个原始任务用各自的 language prompt (frame["task"]) 组织, lerobot 自动建
meta/tasks.parquet (task_index -> prompt) 区分 —— 这就是 lerobot 自带的多 prompt
机制, 一个 dataset 容纳多种任务/指令.

40 任务全覆盖 (无遗漏无重复). 维度全局统一 16: 3-DOF 的 5 个任务
(ReachTarget / ReachTargetSingle / ReachTargetDual / MovePlate / MoveTwoPlates)
在 base 的 index 2 位 pad 常数 Z, pad 逻辑见 generate_dataset.py.

分组 (与 select.md 一致):
  group_key     中文名      任务数   demo 数    维度
  reach         简单到达     3        140        全 3-DOF (15 -> pad 16)
  long_horizon  长程复合     3        120        4-DOF (16)
  dishwasher    洗碗机       10       504        4-DOF (16)
  tabletop      桌面操作     24       1104       22 个 4-DOF + MovePlate×2 共 2 个 3-DOF
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks import TASKS, action_dim  # noqa: E402  单一数据源 (d/tasks.py)

# (group_key, 中文名, [PascalCase 任务])
GROUPS = [
    ("reach", "简单到达", [
        "ReachTarget", "ReachTargetSingle", "ReachTargetDual",
    ]),
    ("long_horizon", "长程复合", [
        "DishwasherUnloadCupsLong",
        "DishwasherUnloadPlatesLong",
        "DishwasherUnloadCutleryLong",
    ]),
    ("dishwasher", "洗碗机", [
        "DishwasherOpen", "DishwasherClose",
        "DishwasherOpenTrays", "DishwasherCloseTrays",
        "DishwasherLoadCups", "DishwasherUnloadCups",
        "DishwasherLoadPlates", "DishwasherUnloadPlates",
        "DishwasherLoadCutlery", "DishwasherUnloadCutlery",
    ]),
    ("tabletop", "桌面操作", [
        # 抽屉与柜门开关 (8)
        "DrawerTopOpen", "DrawerTopClose",
        "DrawersAllOpen", "DrawersAllClose",
        "WallCupboardOpen", "WallCupboardClose",
        "CupboardsOpenAll", "CupboardsCloseAll",
        # 取放 (8)
        "PutCups", "TakeCups", "StoreBox", "PickBox",
        "SaucepanToHob", "StoreKitchenware",
        "GroceriesStoreLower", "GroceriesStoreUpper",
        # 移盘 (2, 3-DOF)
        "MovePlate", "MoveTwoPlates",
        # 翻转 (5)
        "FlipCup", "FlipCutlery", "FlipSandwich",
        "ToastSandwich", "RemoveSandwich",
        # 堆叠 (1)
        "StackBlocks",
    ]),
]

GROUPS_DICT = {k: (name, tasks) for k, name, tasks in GROUPS}


def _validate():
    """校验 4 组恰好覆盖 tasks.py 的 40 个任务, 无遗漏无重复."""
    all_tasks = [t for _, _, ts in GROUPS for t in ts]
    assert len(all_tasks) == 40, f"分组任务总数 {len(all_tasks)}, 应为 40"
    assert len(set(all_tasks)) == 40, f"分组内有重复任务"
    missing = set(TASKS) - set(all_tasks)
    extra = set(all_tasks) - set(TASKS)
    assert not missing, f"分组遗漏任务: {sorted(missing)}"
    assert not extra, f"分组多余任务(拼写错?): {sorted(extra)}"


_validate()


def group_summary(key):
    """返回 (中文名, 任务数, demo 总数, 维度分布 {dim: 任务数})."""
    name, tasks = GROUPS_DICT[key]
    n_demos = sum(TASKS[t]["n_demos"] for t in tasks)
    dim_dist = dict(Counter(action_dim(t) for t in tasks))
    return name, len(tasks), n_demos, dim_dist


def repo_id(key):
    """lerobot repo_id = bigym_plus/<group_key>."""
    return f"bigym_plus/{key}"


def all_group_keys():
    return [k for k, _, _ in GROUPS]


if __name__ == "__main__":
    print(f"{'group_key':<14}{'中文名':<12}{'任务':<6}{'demos':<8}维度分布")
    print("-" * 60)
    total_demos = 0
    for key, name, _tasks in GROUPS:
        _, n, nd, dim_dist = group_summary(key)
        total_demos += nd
        print(f"{key:<14}{name:<12}{n:<6}{nd:<8}{dim_dist}")
    print("-" * 60)
    print(f"合计 {len(GROUPS)} 组 / 40 任务 / {total_demos} demos")
