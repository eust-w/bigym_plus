# BiGym 评测 maxstep 规范

定义 BiGym 评测的 per-task maxstep(单 episode 最大步数):

maxstep = ceil(max(step_count_20hz) × 1.5)

取该任务人类最长演示的 1.5 倍作为模型评测的步数预算。同一 env_class 的所有 episode 共用同一 maxstep,不同 env_class 各用各值,不跨任务混用。本规范覆盖官方 BiGym 0.9.0 演示数据全部 40 个 env_class、1868 条 absolute 轨迹,评测任一 env_class 时按下表取其 maxstep 作为 env.step 循环上限。

## 一、统计口径

- 控制频率 20Hz,env.step 一次 = 一个 20Hz timestep。
- 原始 demo 500Hz 录制,`DemoStore.get_demos(frequency=20)` 经 decimate 降到 20Hz,decimation_rate=25,20Hz step 数 = ceil(500Hz 原始 / 25)。
- step 数 = safetensors 文件 `info_demo_action` 张量第一维长度(`demonstrations/demo.py`)。
- 统计 absolute 轨迹,与训练/评测一致(delta 是 absolute 的差分,step 数相同,不单列)。
- ceil(raw/25) 在 31 个有 20Hz 缓存的 env_class 上逐条验证,1868 条零误差;9 个无缓存的用公式估算,可靠。

## 二、40 任务 step 分布与 maxstep

按 maxstep 升序。step 统计均为 20Hz。method:20hz_cache=直接读 20Hz 缓存,formula=无缓存用 ceil(raw/25) 估算。

| env_class | n | min | max | mean | p50 | p90 | p95 | maxstep | method |
|---|---|---|---|---|---|---|---|---|---|
| ReachTargetSingle | 30 | 54 | 76 | 62.6 | 61 | 70 | 75 | 114 | 20hz_cache |
| ReachTarget | 60 | 56 | 78 | 65.5 | 64 | 74 | 75 | 117 | 20hz_cache |
| DrawerTopClose | 51 | 62 | 80 | 67.3 | 66 | 72 | 76 | 120 | formula |
| ReachTargetDual | 50 | 58 | 92 | 71.5 | 68 | 86 | 90 | 138 | 20hz_cache |
| WallCupboardClose | 60 | 66 | 110 | 80.8 | 80 | 93 | 96 | 165 | formula |
| MovePlate | 60 | 53 | 134 | 73.5 | 72 | 85 | 91 | 201 | 20hz_cache |
| DrawersAllClose | 60 | 115 | 172 | 134.3 | 131 | 157 | 161 | 258 | 20hz_cache |
| DrawerTopOpen | 47 | 117 | 174 | 140.9 | 142 | 157 | 160 | 261 | 20hz_cache |
| FlipCup | 60 | 133 | 202 | 159.7 | 158 | 178 | 188 | 303 | 20hz_cache |
| MoveTwoPlates | 51 | 130 | 207 | 152.5 | 152 | 172 | 179 | 311 | 20hz_cache |
| WallCupboardOpen | 51 | 112 | 236 | 151.1 | 146 | 177 | 209 | 354 | 20hz_cache |
| DishwasherClose | 69 | 139 | 276 | 171.2 | 168 | 187 | 210 | 414 | 20hz_cache |
| DishwasherLoadCups | 60 | 118 | 282 | 187.7 | 182 | 234 | 252 | 423 | 20hz_cache |
| DishwasherOpen | 56 | 165 | 292 | 216.9 | 210 | 261 | 276 | 438 | 20hz_cache |
| DishwasherCloseTrays | 60 | 225 | 298 | 254.2 | 253 | 278 | 290 | 447 | 20hz_cache |
| PutCups | 56 | 192 | 335 | 270.6 | 269 | 300 | 311 | 503 | 20hz_cache |
| DishwasherOpenTrays | 60 | 232 | 379 | 271.1 | 267 | 295 | 311 | 569 | 20hz_cache |
| DishwasherUnloadCups | 57 | 240 | 388 | 303.7 | 302 | 341 | 356 | 582 | 20hz_cache |
| TakeCups | 36 | 267 | 407 | 317.2 | 312 | 353 | 379 | 611 | 20hz_cache |
| SaucepanToHob | 36 | 329 | 427 | 382.0 | 379 | 422 | 426 | 641 | 20hz_cache |
| DrawersAllOpen | 54 | 318 | 466 | 390.5 | 390 | 432 | 447 | 699 | 20hz_cache |
| FlipCutlery | 47 | 187 | 481 | 233.1 | 224 | 264 | 270 | 722 | 20hz_cache |
| RemoveSandwich | 36 | 325 | 522 | 409.8 | 411 | 474 | 521 | 783 | formula |
| DishwasherLoadCutlery | 34 | 271 | 528 | 348.3 | 334 | 403 | 470 | 792 | 20hz_cache |
| PickBox | 35 | 286 | 531 | 369.8 | 365 | 430 | 466 | 797 | 20hz_cache |
| DishwasherLoadPlates | 34 | 359 | 547 | 442.8 | 444 | 512 | 533 | 821 | 20hz_cache |
| StoreBox | 39 | 391 | 601 | 485.8 | 478 | 564 | 578 | 902 | formula |
| DishwasherUnloadCutlery | 39 | 398 | 606 | 475.4 | 472 | 554 | 604 | 909 | 20hz_cache |
| CupboardsCloseAll | 58 | 419 | 624 | 492.6 | 482 | 548 | 596 | 936 | 20hz_cache |
| FlipSandwich | 51 | 242 | 626 | 404.5 | 373 | 557 | 569 | 939 | formula |
| DishwasherUnloadCupsLong | 51 | 420 | 692 | 572.3 | 575 | 655 | 686 | 1038 | formula |
| ToastSandwich | 35 | 320 | 695 | 431.8 | 407 | 555 | 659 | 1043 | formula |
| GroceriesStoreUpper | 35 | 488 | 760 | 619.7 | 614 | 737 | 743 | 1140 | 20hz_cache |
| StoreKitchenware | 33 | 551 | 785 | 648.9 | 640 | 726 | 763 | 1178 | 20hz_cache |
| DishwasherUnloadPlates | 35 | 474 | 799 | 654.1 | 641 | 770 | 788 | 1199 | 20hz_cache |
| DishwasherUnloadCutleryLong | 32 | 544 | 802 | 659.8 | 657 | 744 | 752 | 1203 | formula |
| CupboardsOpenAll | 46 | 577 | 899 | 742.8 | 733 | 851 | 869 | 1349 | 20hz_cache |
| DishwasherUnloadPlatesLong | 37 | 713 | 1023 | 816.7 | 797 | 919 | 1000 | 1535 | formula |
| StackBlocks | 35 | 833 | 1125 | 996.1 | 994 | 1089 | 1117 | 1688 | 20hz_cache |
| GroceriesStoreLower | 32 | 753 | 1265 | 1000.7 | 990 | 1153 | 1210 | 1898 | 20hz_cache |

maxstep 最小 114(ReachTargetSingle,短执行链路伸手任务),最大 1898(GroceriesStoreLower,多物体入柜)。Long 变体(后缀 Long)maxstep 显著大于基础变体,因人类演示更长。

## 三、局限

1. max×1.5 假设模型所需步数不超过人类最长演示的 1.5 倍。模型若远比人类冗余(反复试探)可能仍不够,但 1.5 倍是经验上较宽松的预算。
2. 同任务所有 episode 用同一 maxstep,用最大值 1.5 倍已覆盖最难实例。
3. Long 变体 maxstep 较大(1038-1535),评测耗时长,由任务本身执行链路长度决定。
4. BiGym 论文仅报告部分任务的 ACT 成绩,Long 变体及部分基础任务无公开基线,本规范仍给其 maxstep 以备评测扩展。
