"""bigym_plus 40 任务单一数据源 (canonical 字符串名称 = env_class, PascalCase).

清洗自 bigym 0.9.0 全部 env_class:
- env_module / env_class: 直接来自 bigym.envs.* 源码 (importlib 实测)
- prompt: 取自 env class 的 docstring 首行 (权威语言指令来源, 可追溯)
- floating_dofs: 从 demo cache action mode 目录名实测
  (JointPositionActionMode_floating_pelvis_x_pelvis_y[_pelvis_z]_pelvis_rz_absolute,
  含 pelvis_z → 4-DOF, 否则 3-DOF). 非几何猜测.
- n_demos: ~/.bigym/demonstrations/0.9.0/<EnvClass>/.../lightweight/*.safetensors 计数

覆盖 eval_rules.md 全 40 env_class, 修正旧 bigym_def.py 只覆盖 25 任务且漏标
ReachTarget/MovePlate 为 3-DOF 的问题. 3-DOF 共 5 个 (reach×3 + move_plate×2),
其余 35 个 4-DOF.

maxstep 不在此重复 —— 单源在 max_step_limit.yaml (eval 程序读取), 见 load_maxstep().
维度结构 (state 与 action 同构, 顺序一致):
  [0:base_dof]   floating base (X/Y/[Z]/RZ)         base     JOINT
  [base:+10]     双臂关节角 (左肩3+左肘+左腕 + 右同)  arm      JOINT
  [-2:]          left/right gripper 开合 [0,1]      gripper  GRIPPER
action_dim = len(floating_dofs) + ARM_DIM(10) + GRIPPER_DIM(2), 随任务浮动 (15 或 16).
"""
import os
import re

from bigym.action_modes import PelvisDof as _D

DOFS_3 = [_D.X, _D.Y, _D.RZ]            # 3-DOF: reach×3 + move_plate×2 → action 15 维
DOFS_4 = [_D.X, _D.Y, _D.Z, _D.RZ]      # 4-DOF: 其余 35 任务 → action 16 维

ARM_DIM = 10
GRIPPER_DIM = 2

FREQ = 20

# canonical key = env_class (PascalCase). 与 demo cache 目录名 / Python 类名 / eval_rules 三处对齐.
TASKS = {
    "CupboardsCloseAll": dict(
        env_module="bigym.envs.cupboards", env_class="CupboardsCloseAll",
        prompt="Close all doors/drawers of the kitchen counter task.",
        floating_dofs=DOFS_4, n_demos=58),
    "CupboardsOpenAll": dict(
        env_module="bigym.envs.cupboards", env_class="CupboardsOpenAll",
        prompt="Open all doors/drawers of the kitchen counter task.",
        floating_dofs=DOFS_4, n_demos=46),
    "DishwasherClose": dict(
        env_module="bigym.envs.dishwasher", env_class="DishwasherClose",
        prompt="Push back all trays and close the door of the dishwasher.",
        floating_dofs=DOFS_4, n_demos=69),
    "DishwasherCloseTrays": dict(
        env_module="bigym.envs.dishwasher", env_class="DishwasherCloseTrays",
        prompt="Push the dishwasher's trays back with the door initially open.",
        floating_dofs=DOFS_4, n_demos=60),
    "DishwasherLoadCups": dict(
        env_module="bigym.envs.dishwasher_cups", env_class="DishwasherLoadCups",
        prompt="Load cups to dishwasher task.",
        floating_dofs=DOFS_4, n_demos=60),
    "DishwasherLoadCutlery": dict(
        env_module="bigym.envs.dishwasher_cutlery", env_class="DishwasherLoadCutlery",
        prompt="Load cutlery to dishwasher task.",
        floating_dofs=DOFS_4, n_demos=34),
    "DishwasherLoadPlates": dict(
        env_module="bigym.envs.dishwasher_plates", env_class="DishwasherLoadPlates",
        prompt="Load plates to dishwasher task.",
        floating_dofs=DOFS_4, n_demos=34),
    "DishwasherOpen": dict(
        env_module="bigym.envs.dishwasher", env_class="DishwasherOpen",
        prompt="Open the dishwasher door and pull out all trays.",
        floating_dofs=DOFS_4, n_demos=56),
    "DishwasherOpenTrays": dict(
        env_module="bigym.envs.dishwasher", env_class="DishwasherOpenTrays",
        prompt="Pull the dishwasher's trays with the door initially open.",
        floating_dofs=DOFS_4, n_demos=60),
    "DishwasherUnloadCups": dict(
        env_module="bigym.envs.dishwasher_cups", env_class="DishwasherUnloadCups",
        prompt="Unload cups from dishwasher task.",
        floating_dofs=DOFS_4, n_demos=57),
    "DishwasherUnloadCupsLong": dict(
        env_module="bigym.envs.dishwasher_cups", env_class="DishwasherUnloadCupsLong",
        prompt="Unload cup from dishwasher in wall cabinet task.",
        floating_dofs=DOFS_4, n_demos=51),
    "DishwasherUnloadCutlery": dict(
        env_module="bigym.envs.dishwasher_cutlery", env_class="DishwasherUnloadCutlery",
        prompt="Unload cutlery from dishwasher task.",
        floating_dofs=DOFS_4, n_demos=39),
    "DishwasherUnloadCutleryLong": dict(
        env_module="bigym.envs.dishwasher_cutlery", env_class="DishwasherUnloadCutleryLong",
        prompt="Unload cutlery from dishwasher to drawer task.",
        floating_dofs=DOFS_4, n_demos=32),
    "DishwasherUnloadPlates": dict(
        env_module="bigym.envs.dishwasher_plates", env_class="DishwasherUnloadPlates",
        prompt="Unload plates from dishwasher task.",
        floating_dofs=DOFS_4, n_demos=35),
    "DishwasherUnloadPlatesLong": dict(
        env_module="bigym.envs.dishwasher_plates", env_class="DishwasherUnloadPlatesLong",
        prompt="Unload plate from dishwasher in wall cabinet task.",
        floating_dofs=DOFS_4, n_demos=37),
    "DrawerTopClose": dict(
        env_module="bigym.envs.cupboards", env_class="DrawerTopClose",
        prompt="Close top drawer of the cupboard task.",
        floating_dofs=DOFS_4, n_demos=51),
    "DrawerTopOpen": dict(
        env_module="bigym.envs.cupboards", env_class="DrawerTopOpen",
        prompt="Open top drawer of the cupboard task.",
        floating_dofs=DOFS_4, n_demos=47),
    "DrawersAllClose": dict(
        env_module="bigym.envs.cupboards", env_class="DrawersAllClose",
        prompt="Close all drawers of the cupboard task.",
        floating_dofs=DOFS_4, n_demos=60),
    "DrawersAllOpen": dict(
        env_module="bigym.envs.cupboards", env_class="DrawersAllOpen",
        prompt="Open all drawers of the cupboard task.",
        floating_dofs=DOFS_4, n_demos=54),
    "FlipCup": dict(
        env_module="bigym.envs.manipulation", env_class="FlipCup",
        prompt="Flip cup upside-up task.",
        floating_dofs=DOFS_4, n_demos=60),
    "FlipCutlery": dict(
        env_module="bigym.envs.manipulation", env_class="FlipCutlery",
        prompt="Flip cutlery item task.",
        floating_dofs=DOFS_4, n_demos=47),
    "FlipSandwich": dict(
        env_module="bigym.envs.pick_and_place", env_class="FlipSandwich",
        prompt="Flip sandwich using spatula.",
        floating_dofs=DOFS_4, n_demos=51),
    "GroceriesStoreLower": dict(
        env_module="bigym.envs.groceries", env_class="GroceriesStoreLower",
        prompt="Put groceries to lower cabinets tasks.",
        floating_dofs=DOFS_4, n_demos=32),
    "GroceriesStoreUpper": dict(
        env_module="bigym.envs.groceries", env_class="GroceriesStoreUpper",
        prompt="Put groceries to upper cabinets tasks.",
        floating_dofs=DOFS_4, n_demos=35),
    "MovePlate": dict(
        env_module="bigym.envs.move_plates", env_class="MovePlate",
        prompt="Move one plate from one rack to another.",
        floating_dofs=DOFS_3, n_demos=60),
    "MoveTwoPlates": dict(
        env_module="bigym.envs.move_plates", env_class="MoveTwoPlates",
        prompt="Move two plates from one rack to another.",
        floating_dofs=DOFS_3, n_demos=51),
    "PickBox": dict(
        env_module="bigym.envs.pick_and_place", env_class="PickBox",
        prompt="Pick up box from and place it on the counter task.",
        floating_dofs=DOFS_4, n_demos=35),
    "PutCups": dict(
        env_module="bigym.envs.pick_and_place", env_class="PutCups",
        prompt="Put cups in the wall cabinet.",
        floating_dofs=DOFS_4, n_demos=56),
    "ReachTarget": dict(
        env_module="bigym.envs.reach_target", env_class="ReachTarget",
        prompt="Reach the target with either left or right wrist.",
        floating_dofs=DOFS_3, n_demos=60),
    "ReachTargetDual": dict(
        env_module="bigym.envs.reach_target", env_class="ReachTargetDual",
        prompt="Reach 2 targets, one with each arm.",
        floating_dofs=DOFS_3, n_demos=50),
    "ReachTargetSingle": dict(
        env_module="bigym.envs.reach_target", env_class="ReachTargetSingle",
        prompt="Reach the target with specific wrist.",
        floating_dofs=DOFS_3, n_demos=30),
    "RemoveSandwich": dict(
        env_module="bigym.envs.pick_and_place", env_class="RemoveSandwich",
        prompt="Remove sandwich from the frying pan.",
        floating_dofs=DOFS_4, n_demos=36),
    "SaucepanToHob": dict(
        env_module="bigym.envs.pick_and_place", env_class="SaucepanToHob",
        prompt="Take saucepan from cabinet and place it to hob.",
        floating_dofs=DOFS_4, n_demos=36),
    "StackBlocks": dict(
        env_module="bigym.envs.manipulation", env_class="StackBlocks",
        prompt="Stack blocks in the correct area of the table.",
        floating_dofs=DOFS_4, n_demos=35),
    "StoreBox": dict(
        env_module="bigym.envs.pick_and_place", env_class="StoreBox",
        prompt="Put box in the cupboard task.",
        floating_dofs=DOFS_4, n_demos=39),
    "StoreKitchenware": dict(
        env_module="bigym.envs.pick_and_place", env_class="StoreKitchenware",
        prompt="Put all kitchenware to cupboard.",
        floating_dofs=DOFS_4, n_demos=33),
    "TakeCups": dict(
        env_module="bigym.envs.pick_and_place", env_class="TakeCups",
        prompt="Take cups from the wall cupboard.",
        floating_dofs=DOFS_4, n_demos=36),
    "ToastSandwich": dict(
        env_module="bigym.envs.pick_and_place", env_class="ToastSandwich",
        prompt="Move sandwich on the frying pan.",
        floating_dofs=DOFS_4, n_demos=35),
    "WallCupboardClose": dict(
        env_module="bigym.envs.cupboards", env_class="WallCupboardClose",
        prompt="Close doors of the wall cupboard task.",
        floating_dofs=DOFS_4, n_demos=60),
    "WallCupboardOpen": dict(
        env_module="bigym.envs.cupboards", env_class="WallCupboardOpen",
        prompt="Open doors of the wall cupboard task.",
        floating_dofs=DOFS_4, n_demos=51),
}

_D_PLUS_DIR = os.path.dirname(os.path.abspath(__file__))
MAXSTEP_YAML = os.path.join(_D_PLUS_DIR, "max_step_limit.yaml")


def task_to_snake(name: str) -> str:
    """PascalCase env_class → snake_case (用于 dataset 目录名 / 文件名)."""
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower()


def action_dim(task_key: str) -> int:
    """state/action 维度 = base_dof + arm(10) + gripper(2)."""
    return len(TASKS[task_key]["floating_dofs"]) + ARM_DIM + GRIPPER_DIM


def resolve_task(task_key: str) -> str:
    """接受 PascalCase (canonical) 或 snake_case, 返回 canonical env_class."""
    if task_key in TASKS:
        return task_key
    snake_map = {task_to_snake(k): k for k in TASKS}
    if task_key in snake_map:
        return snake_map[task_key]
    raise KeyError(f"Unknown task: {task_key}. Valid: {sorted(TASKS)}")


def load_maxstep(path: str = MAXSTEP_YAML) -> dict:
    """读取 max_step_limit.yaml → {env_class: maxstep}. 单源, eval 调用."""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return {k: int(v) for k, v in data.items()}


def get_maxstep(task_key: str, path: str = MAXSTEP_YAML) -> int:
    """取单任务 maxstep (从 yaml 读)."""
    return load_maxstep(path)[resolve_task(task_key)]


if __name__ == "__main__":
    ms = load_maxstep()
    print(f"{'env_class':<30} {'dof':<6} {'dim':<4} {'n':<4} {'maxstep':<8} snake")
    print("-" * 90)
    for k in sorted(TASKS):
        info = TASKS[k]
        dof = len(info["floating_dofs"])
        print(f"{k:<30} {dof:<6} {action_dim(k):<4} {info['n_demos']:<4} "
              f"{ms[k]:<8} {task_to_snake(k)}")
    print(f"\ntotal: {len(TASKS)} tasks, 3-DOF: "
          f"{sum(1 for v in TASKS.values() if len(v['floating_dofs'])==3)}, "
          f"4-DOF: {sum(1 for v in TASKS.values() if len(v['floating_dofs'])==4)}")
