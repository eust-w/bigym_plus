# BiGym Plus 数据采集 · 任务选择

在「选」列把 `[ ]` 改成 `[x]` 表示选用该任务;分组为初版建议,可直接调整任务归属或组名。任务描述取自 `bigym/README_cn.md` 原文,预览图为 BiGym 官方任务截图(相对本文件位于 `../../doc/images/tasks/`)。

总览:40 任务 / 1868 demos,统一到 16 维;组4 的 5 个任务原为 15 维(base 无上下浮动),统一时在 Z 位(index 2)pad 常数。

---

## 组 简单任务

| 选 | 任务 | 描述 | 预览 |
|---|---|---|---|
| [ ] | ReachTarget | 用左手或右手腕到达目标 | <img src="../../doc/images/tasks/task_preview_reach_target@720x720.png" width=320> |
| [ ] | ReachTargetSingle | 用指定的手腕到达目标 | <img src="../../doc/images/tasks/task_preview_reach_target_single@720x720.png" width=320> |
| [ ] | ReachTargetDual | 到达 2 个目标,每只手臂各到达一个 | <img src="../../doc/images/tasks/task_preview_reach_target_dual@720x720.png" width=320> |


## 组1 · 长程任务

多阶段任务:从洗碗机取物 → 放入柜或抽屉 → 关洗碗机和柜门。

| 选 | 任务 | 描述 | 预览 |
|---|---|---|---|
| [ ] | DishwasherUnloadCupsLong | 完整的卸载杯子任务:拿起杯子,将其放入关闭的壁橱内,然后关闭洗碗机和橱柜 | <img src="../../doc/images/tasks/task_preview_dishwasher_unload_cups_long@720x720.png" width=320> |
| [ ] | DishwasherUnloadPlatesLong | 完整的卸载盘子任务:从洗碗机中拿起盘子,将其放入位于关闭的壁橱中的架子,然后关闭洗碗机和橱柜 | <img src="../../doc/images/tasks/task_preview_dishwasher_unload_plates_long@720x720.png" width=320> |
| [ ] | DishwasherUnloadCutleryLong | 完整的卸载餐具任务:拿起餐具,将其放入关闭抽屉内的餐具托盘,然后关闭洗碗机和抽屉 | <img src="../../doc/images/tasks/task_preview_dishwasher_unload_cutlery_long@720x720.png" width=320> |

## 组2 · 洗碗机工作台

围绕洗碗机的开关门托盘与杯盘餐具装卸。

| 选 | 任务 | 描述 | 预览 |
|---|---|---|---|
| [ ] | DishwasherOpen | 打开洗碗机门并拉出所有托盘 | <img src="../../doc/images/tasks/task_preview_dishwasher_open@720x720.png" width=320> |
| [ ] | DishwasherClose | 推回所有托盘并关上洗碗机的门 | <img src="../../doc/images/tasks/task_preview_dishwasher_close@720x720.png" width=320> |
| [ ] | DishwasherOpenTrays | 在门已打开的状态下拉出洗碗机的托盘 | <img src="../../doc/images/tasks/task_preview_dishwasher_open_trays@720x720.png" width=320> |
| [ ] | DishwasherCloseTrays | 在门已打开的状态下推回洗碗机的托盘 | <img src="../../doc/images/tasks/task_preview_dishwasher_close_trays@720x720.png" width=320> |
| [ ] | DishwasherLoadCups | 将杯子从桌面移动到洗碗机的上层托盘 | <img src="../../doc/images/tasks/task_preview_dishwasher_load_cups@720x720.png" width=320> |
| [ ] | DishwasherUnloadCups | 将杯子从洗碗机的上层托盘移动到桌面 | <img src="../../doc/images/tasks/task_preview_dishwasher_unload_cups@720x720.png" width=320> |
| [ ] | DishwasherLoadPlates | 将盘子从架子移动到洗碗机的下层托盘 | <img src="../../doc/images/tasks/task_preview_dishwasher_load_plates@720x720.png" width=320> |
| [ ] | DishwasherUnloadPlates | 将盘子从洗碗机托盘移动到桌面架子 | <img src="../../doc/images/tasks/task_preview_dishwasher_unload_plates@720x720.png" width=320> |
| [ ] | DishwasherLoadCutlery | 将餐具从桌面支架移动到洗碗机的餐具篮 | <img src="../../doc/images/tasks/task_preview_dishwasher_load_cutlery@720x720.png" width=320> |
| [ ] | DishwasherUnloadCutlery | 将餐具从餐具篮移动到桌面上的托盘 | <img src="../../doc/images/tasks/task_preview_dishwasher_unload_cutlery@720x720.png" width=320> |

## 组3 桌面操作

厨房橱柜的抽屉柜门开关,以及往柜内/台面间取放物品。

| 选 | 任务 | 描述 | 预览 |
|---|---|---|---|
| [ ] | DrawerTopOpen | 打开厨房橱柜的顶层抽屉 | <img src="../../doc/images/tasks/task_preview_drawer_top_open@720x720.png" width=320> |
| [ ] | DrawerTopClose | 关闭厨房橱柜的顶层抽屉 | <img src="../../doc/images/tasks/task_preview_drawer_top_close@720x720.png" width=320> |
| [ ] | DrawersAllOpen | 打开厨房橱柜的所有滑出式抽屉 | <img src="../../doc/images/tasks/task_preview_drawers_all_open@720x720.png" width=320> |
| [ ] | DrawersAllClose | 关闭厨房橱柜的所有滑出式抽屉 | <img src="../../doc/images/tasks/task_preview_drawers_all_close@720x720.png" width=320> |
| [ ] | WallCupboardOpen | 打开壁橱的门 | <img src="../../doc/images/tasks/task_preview_wall_cupboard_open@720x720.png" width=320> |
| [ ] | WallCupboardClose | 关闭壁橱的门 | <img src="../../doc/images/tasks/task_preview_wall_cupboard_close@720x720.png" width=320> |
| [ ] | CupboardsOpenAll | 打开厨房整体橱柜的所有抽屉和门 | <img src="../../doc/images/tasks/task_preview_cupboards_open_all@720x720.png" width=320> |
| [ ] | CupboardsCloseAll | 关闭厨房整体橱柜的所有抽屉和门 | <img src="../../doc/images/tasks/task_preview_cupboards_close_all@720x720.png" width=320> |
| [ ] | PutCups | 从桌面拿起杯子并放入关闭的壁橱 | <img src="../../doc/images/tasks/task_preview_put_cups@720x720.png" width=320> |
| [ ] | TakeCups | 从关闭的壁橱中取出两个杯子并放到桌面上 | <img src="../../doc/images/tasks/task_preview_take_cups@720x720.png" width=320> |
| [ ] | StoreBox | 将一个大箱子从台面移动到下方橱柜的搁板上 | <img src="../../doc/images/tasks/task_preview_store_box@720x720.png" width=320> |
| [ ] | PickBox | 从地面拿起一个大箱子并放到台面上 | <img src="../../doc/images/tasks/task_preview_pick_box@720x720.png" width=320> |
| [ ] | SaucepanToHob | 从关闭的橱柜中取出平底锅并放到灶台上 | <img src="../../doc/images/tasks/task_preview_saucepan_to_hob@720x720.png" width=320> |
| [ ] | StoreKitchenware | 从灶台上取下所有物品并放入下方的橱柜 | <img src="../../doc/images/tasks/task_preview_store_kitchenware@720x720.png" width=320> |
| [ ] | GroceriesStoreLower | 将一组随机选取的食品杂货放入台面下方的橱柜中 | <img src="../../doc/images/tasks/task_preview_groceries_store_lower@720x720.png" width=320> |
| [ ] | GroceriesStoreUpper | 将一组随机选取的食品杂货放入墙上的橱柜和搁板中 | <img src="../../doc/images/tasks/task_preview_groceries_store_upper@720x720.png" width=320> |
| [ ] | MovePlate | 在两个沥水架之间移动盘子 | <img src="../../doc/images/tasks/task_preview_move_plate@720x720.png" width=320> |
| [ ] | MoveTwoPlates | 同时将两个盘子从一个沥水架移动到另一个 | <img src="../../doc/images/tasks/task_preview_move_two_plates@720x720.png" width=320> |
| [ ] | StackBlocks | 在桌面上移动方块,并将其堆叠到目标区域 | <img src="../../doc/images/tasks/task_preview_stack_blocks@720x720.png" width=320> |
| [ ] | FlipCup | 将初始倒扣在桌面上的杯子翻转至正立位置 | <img src="../../doc/images/tasks/task_preview_flip_cup@720x720.png" width=320> |
| [ ] | FlipCutlery | 从固定支架上取下餐具,将其翻转,再放回支架 | <img src="../../doc/images/tasks/task_preview_flip_cutlery@720x720.png" width=320> |
| [ ] | FlipSandwich | 用锅铲翻转煎锅里的三明治 | <img src="../../doc/images/tasks/task_preview_flip_sandwich@720x720.png" width=320> |
| [ ] | ToastSandwich | 用锅铲将三明治放到煎锅里 | <img src="../../doc/images/tasks/task_preview_toast_sandwich@720x720.png" width=320> |
| [ ] | RemoveSandwich | 将三明治从煎锅中取出 | <img src="../../doc/images/tasks/task_preview_remove_sandwich@720x720.png" width=320> |
