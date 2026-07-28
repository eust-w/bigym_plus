# demo success 审计(无渲染回灌)

检查 BiGym 官方演示轨迹在回灌(replay)后是否真正达到 task success。生成脚本 generate_dataset.py 回放 demo 时不检查 success,无条件保存;本脚本用无渲染的快速回灌逐条判定,统计成功率,定位受影响的任务。

## 为什么需要

demo 文件录制时的依赖版本(mujoco 3.1.5, bigym 4.0.0)可能与当前回放环境版本不同。回灌理论上是确定性的(reset 同 seed + 逐帧 step 原动作),但 mujoco 物理引擎跨版本会改求解器、积分器、接触约束的数值细节,bigym 跨版本可能改 action_mode 或子步数,导致同 seed + 同动作走出不同数值轨迹,部分演示回灌后不再达到 success。本脚本量化这个比例,供决定是否在生成管线加 success 过滤,或回退依赖版本对齐 demo 录制版本。

## 原理

无渲染最快路径:不建相机、不渲染图像,只跑 mujoco 物理仿真。

- env 构建:build_env(with_cameras=False, render_mode=None),不挂观测相机,不调 env.render()。obs 仍含 proprioception,但本脚本不读 obs。
- 回灌:env.reset(seed=demo.seed) 复现录制 seed,逐帧 env.step(demo 的 executed_action, fast=True)。fast=True 跳过观测构建,返回默认值,但 env.reward 属性仍从内部缓存读出真实 reward。
- 判定:env.reward 大于 0 即该演示回灌后达到 success。reward 取值 0 或 1(由 env.success 派生)。

判定口径与以下两条等价:

- 评测端 check_success 的主路径 reward 大于 0.5(eval_bigym.py)
- 官方 DemoPlayer.validate_in_env 的 env.reward 大于 0(demo_player.py)

env 复用:每个任务建一次 env,循环该任务全部 demo(不像官方 validate_in_env 每条 demo 都 close env),提高批量效率。取 demo 的方式(DemoStore.get_demos + Metadata.from_env,is_lightweight)与 generate_dataset.py 完全一致,保证审计的就是生成时存入数据集的那批 demo。

## 用法

所有命令在 replay_generation/ 目录下执行,conda 环境 amd_bigym_plus312,设 MUJOCO_GL=egl。脚本纯 mujoco CPU 物理仿真,实测不占 GPU 显存,但仍需指定 CUDA_VISIBLE_DEVICES 给 EGL 初始化。

前置:首次运行前先下载演示数据(同 generate_dataset,BiGym 官方轨迹 v0.9.0,缓存到 ~/.bigym):

```
python download_demos.py
```

验证管线(每任务 1 条):

```
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python check_demo_success.py --group reach --probe
```

单任务全量:

```
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python check_demo_success.py --task ReachTargetDual
```

单组全量:

```
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python check_demo_success.py --group reach
```

4 组全量(40 任务):

```
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python check_demo_success.py --group all
```

限制每任务条数(调试):

```
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python check_demo_success.py --group all --amount 12
```

主要参数:

| 参数 | 作用 |
|---|---|
| --task <名> | 单任务模式,接受 PascalCase 或 snake_case(与 --group 互斥) |
| --group <组\|all> | 组模式,组名 reach / long_horizon / dishwasher / tabletop 或 all |
| --amount N | 每任务演示条数上限(默认全量) |
| --probe | 每任务只跑 1 条,验证管线 |

不带 --task 也不带 --group 时默认等价 --group all,审计全部 40 任务。

## 输出

stdout 打印每任务统计表,列含义:

- task:任务名(PascalCase)
- dof:浮动基座自由度(3 或 4)
- succ/total:回灌后达到 success 的条数 / 该任务总条数
- rate%:成功率
- err:回灌异常条数(reset 失败或 step 抛错,如 action 越界)
- no_succ:回灌跑通但未达到 success 的条数(版本偏离的主要表现)

末尾打印合计:总任务数、总演示数、总 success、总 err、总 no_succ、整体成功率、耗时。

JSON 结果落到项目根目录的 claude_analysis_notes/demo_success_audit.json,含三部分:

- meta:环境(conda、频率、判定口径、参数)
- tasks:每任务统计(probe 模式另附 per_demo 逐条记录,非 probe 模式为空省空间)
- summary:合计(任务数、演示数、success、err、no_succ、整体成功率)

## 性能

无渲染 fast step,纯 mujoco CPU 物理仿真,单条 demo 回灌毫秒级,耗时主要在每任务建一次 env(加载 mujoco 模型)。参考实测(amd_bigym_plus312, mujoco 3.10.0, bigym 4.1.0):

- probe(每任务 1 条):秒级
- 单任务全量(50 条左右):数秒
- 4 组全量(40 任务 / 1868 条):约 1 小时

全量审计耗时较长,建议后台运行(nohup),进程号记入 process.txt,轮询日志直到出现 saved 字样。

## 参考基线

当前环境(amd_bigym_plus312, mujoco 3.10.0, bigym 4.1.0)实测全量 40 任务 / 1868 条,整体 success 约 72%;单刚体 reach 与 drawer/wall_cupboard close 类任务接近 100%,多物体精细操作类(dishwasher unload cups/plates/cutlery、stack blocks、sandwich 系列、cupboards open all)普遍偏低。各任务明细见 JSON。审计结果随环境依赖版本变化,换 mujoco 或 bigym 版本后需重跑。

## 依赖

不依赖 lerobot,只需 bigym + numpy。复用本目录现有模块,不改任何生成或评测代码:

- env_utils.build_env:无相机 env 构建
- tasks.TASKS / FREQ / resolve_task:任务定义与频率(20 Hz)
- groups.GROUPS_DICT / all_group_keys:分组
- demonstrations.demo_store.DemoStore + demonstrations.utils.Metadata:取演示与元数据
- demonstrations.demo_player.validate_in_env:判定口径来源(本脚本不复用它,因它每条 demo close env,批量效率低)

## 注意

- no_succ 不等于坏 demo:它只表示当前环境版本下回灌后未达 success,根因多为录制版本与回放版本不一致(mujoco / bigym 升级),非演示本身错误。
- 审计结果随环境依赖版本变化:换 mujoco 或 bigym 版本后,各任务成功率会变,需重跑。
- 脚本只读 demo + 跑物理 + 统计,不写入任何数据集或模型。
