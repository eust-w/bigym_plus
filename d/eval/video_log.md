
# headless 评测环境下录制并保存可播放的 rollout（回灌）视频

## 1. 背景与目标

在 headless Linux GPU 服务器上做机器人或强化学习策略评测时，仅看成功率、回报曲线等数值指标往往不够。把每条 rollout 录成视频能带来三类直接价值：

- 可视化策略行为：观察末端轨迹是否平滑、抓取是否对齐、是否出现抖动或卡死等数值难以反映的现象。
- 定位失败 episode：从失败样本里快速判断是策略决策错误、观测异常，还是环境动力学问题。
- 结果存档与展示：评测产出可归档、可回看、可对外分享，便于结果核查与方案评审。

本文目标是用通用方法把 rollout 录成"双击就能播放"的视频。所谓"可播放"指：浏览器（Chrome / Firefox / Safari）、macOS Quick Look（空格预览）、Windows 照片应用、手机相册都能直接打开播放，无需安装额外播放器或编解码器。VLC 这类桌面全能播放器能放的视频，并不等于满足这一标准。

## 2. 环境渲染原理

### 2.1 rgb_array 渲染接口

gymnasium 与基于 MuJoCo 的环境统一通过 `render_mode` 控制渲染行为。录制视频应使用 `render_mode="rgb_array"`：

```python
import gymnasium as gym

env = gym.make("<env_id>", render_mode="rgb_array")
frame = env.render()  # 返回 HWC, RGB, uint8, 形如 (H, W, 3)
```

`env.render()` 返回的是 NumPy 数组，形状 `(H, W, 3)`、dtype `uint8`、颜色顺序 RGB。默认分辨率由 MuJoCo model 的 `vis/global/offwidth`、`vis/global/offheight` 决定，常见为 640×480，可在 model 里调整。

### 2.2 录制视角与观测视角相互独立

这是录制能安全叠加在评测流程上的根本原因。

机器人环境里通常有两类相机：

- 观测相机：其图像进入 `env.get_observation()` 或 `obs`，是策略的实际输入。
- 渲染相机：由 `env.render()` 调用，仅用于人眼观看的外部视角。

`env.render()` 走的是渲染层旁路，返回的帧不会进入 `env.get_observation()`，也不参与 `env.step(action)` 的动作执行。因此录制视频不影响策略输入，不影响评测数值结果。开关录制对确定性策略得到的成功率、回报等指标是相同的。

### 2.3 headless 下的 EGL 离屏渲染

headless 服务器没有显示设备，默认渲染后端会因找不到 display 而报错。必须通过环境变量切到 EGL 离屏渲染：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

两个变量都要设。`MUJOCO_GL=egl` 让 MuJoCo 选 EGL 后端，`PYOPENGL_PLATFORM=egl` 让 Python OpenGL 绑定也走 EGL。缺任一个都可能回退到 GLX 并因无 display 失败。这两个变量应在启动评测进程前设置好。

## 3. 录制方案对比

三种常见 Python 视频写入方案：

| 方案 | 主要依赖 | 系统级 ffmpeg | 能否直接写 H.264 | 获取难度 |
|---|---|---|---|---|
| cv2.VideoWriter | opencv-python | 不需要（自带 ffmpeg 后端） | 否（仅 mp4v 等） | 多数 Python 环境最易得 |
| imageio | imageio + imageio-ffmpeg | 自带 ffmpeg 二进制 | 是（依赖自带 ffmpeg） | 中等，需装 imageio-ffmpeg |
| PyAV | av（ffmpeg + libx264 的 Python 绑定） | 不需要（自带） | 是 | 中等，需装 av |

OpenCV 在多数 Python 环境里最易得，且 `opencv-python` 包自带 ffmpeg 后端，无需系统额外安装 ffmpeg，写入 mp4v 编码的 mp4 文件零配置可用。代价是它写不出 H.264，mp4v 编码在浏览器与系统预览侧不可播放，需要额外的转码步骤。本文录制环节采用 cv2.VideoWriter，转码环节采用 PyAV。

## 4. 编码兼容性：为什么"录到了却放不出来"

这是本指南的核心坑，与文件路径、权限、磁盘空间都无关。

### 4.1 mp4v 是什么

cv2.VideoWriter 用 `fourcc='mp4v'` 写出的视频，封装在 mp4 容器里，但视频流编码是 **MPEG-4 Part 2**（即早期 DivX / Xvid 那一系），不是 H.264 / AVC：

```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
```

文件扩展名是 `.mp4`，但"mp4 容器"和"H.264 编码"是两回事。容器只规定怎么打包，编码才决定播放器能不能解。

### 4.2 OpenCV 为什么不内置 H.264

H.264 编码器（libx264）受专利许可约束。OpenCV 为避免专利问题，不内置 H.264 encoder，VideoWriter 能用的 fourcc 受限于其自带的 ffmpeg 后端实际编译进来的编码器，mp4v 是其中最通用的一个。尝试 `fourcc='avc1'` 或 `'H264'` 在标准 opencv-python 包里通常会静默回退或写入失败。

### 4.3 表现与影响

mp4v 视频在以下场景普遍打不开：

- Chrome / Firefox / Safari 浏览器（含 `<video>` 标签与文件拖入）。
- macOS Quick Look（空格预览）与 Finder 缩略图。
- Windows 照片应用、电影和电视应用。
- iOS / Android 手机相册。

典型现象是"文件存在、体积正常，但双击后播放器报错或一片黑"。VLC、mpv 等桌面全能播放器可以播放，但评测视频的预览场景恰恰在浏览器与系统预览侧，因此 mp4v 不满足"可播放"要求。

## 5. 转码到通用可播放格式

### 5.1 目标编码组合

浏览器、系统预览、移动端通用的组合是：

- 编码：H.264 / AVC（libx264）。
- 像素格式：`yuv420p`。H.264 默认可能是 `yuv444p`，部分播放器不支持，必须显式指定 `yuv420p`。
- profile：`baseline`。兼容性最好，移动端与低端硬件解得动；若需更高压缩比且确认目标端支持，可改 `high`，但 baseline 是最稳妥的默认。

推荐参数 `preset=veryfast`（速度与体积均衡）、`crf=23`（质量与体积均衡，18~28 之间可调，越小越清晰越大）。逐帧 decode→encode 可精确保持帧数，转码后体积通常比 mp4v 还小约一半。

### 5.2 转码实现

做法是原位替换：先写到 `<src>.tmp.mp4`，成功后用 `os.replace` 覆盖原文件，保证中途失败不破坏原始文件。

```python
import os
import av

def transcode_to_h264(src_path: str, fps: float | None = None) -> None:
    """将 mp4v 视频转码为 H.264 + yuv420p + baseline，原位替换。"""
    tmp_path = src_path + ".tmp.mp4"

    input_container = av.open(src_path)
    in_stream = input_container.streams.video[0]
    src_fps = fps if fps is not None else float(in_stream.average_rate)
    width, height = in_stream.width, in_stream.height

    output_container = av.open(tmp_path, mode="w")
    out_stream = output_container.add_stream("libx264", rate=src_fps)
    out_stream.width = width
    out_stream.height = height
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {
        "preset": "veryfast",
        "crf": "23",
        "profile": "baseline",
    }

    for frame in input_container.decode(video=0):
        output_container.mux(out_stream.encode(frame))
    output_container.mux(out_stream.encode())  # flush 编码器

    input_container.close()
    output_container.close()
    os.replace(tmp_path, src_path)
```

`fps` 参数允许外部传入，因为某些 mp4v 文件写头时记录的帧率不可靠，用评测时已知的控制频率覆盖更稳妥。

## 6. 录制实现要点

### 6.1 懒打开与逐帧写入

VideoWriter 在创建时就要确定分辨率。提前猜分辨率容易出错，正确做法是懒打开：首帧到来时按该帧实际尺寸创建 writer，之后逐帧写入。

```python
import cv2
from typing import Optional

class RolloutRecorder:
    """每个 episode 一个 mp4 文件，懒打开 VideoWriter。"""

    def __init__(self, fps: float):
        self.fps = float(fps)
        self._writer: Optional[cv2.VideoWriter] = None

    def _open(self, frame, path: str) -> None:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        if not self._writer.isOpened():
            raise RuntimeError(f"VideoWriter 打开失败: {path}")

    def add(self, frame, path: str) -> None:
        """frame: HWC, RGB, uint8。path 在同一 episode 内应保持一致。"""
        if self._writer is None:
            self._open(frame, path)
        bgr = frame[:, :, ::-1]  # RGB -> BGR（OpenCV 约定）
        self._writer.write(bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
```

注意 OpenCV 要求 BGR，而 `env.render()` 返回 RGB，写入前要反转通道。

### 6.2 命名约定与成功失败标注

每个 episode 单独一个 mp4 文件，方便按 episode 定位。命名约定：

```
{env}_{tag}_ep{NN}_{ok|no}.mp4
```

- `env`：环境标识。
- `tag`：策略或实验标识。
- `ep{NN}`：episode 序号，两位补零，保证文件排序正确。
- `ok` / `no`：该 episode 成功或失败。失败样本用 `no` 后缀，便于在文件管理器里一眼挑出失败回放集中查看。

`ok` / `no` 要到 episode 结束才能确定，而录制必须从首帧开始。处理方式：录制时先写到临时路径，episode 结束后根据成功与否 `os.replace` 改名为带 `ok`/`no` 后缀的最终路径。

### 6.3 帧率对齐控制频率

VideoWriter 的 `fps` 应与环境的控制频率（control frequency）一致，回放才是实时速度。控制频率是 `env.step()` 的调用频率，例如 20 Hz 表示每秒 20 个控制步。若 `fps` 设得高于控制频率，回放会快放；低于则会慢放，动作看起来拖沓。从 env 获取控制频率后直接传给 recorder 与 transcoder，三者保持同一个值。

### 6.4 录制器与评测循环集成

`reset` 后先录一帧初始状态，再进入 `step` 循环每步录一帧。这样视频开头能看到环境初始构型，便于判断后续动作的起点是否正常。

```python
import os
import gymnasium as gym

env = gym.make("<env_id>", render_mode="rgb_array")
control_freq = env.metadata.get("render_fps", 30)  # 按实际控制频率取
recorder = RolloutRecorder(fps=control_freq)
video_dir = "<video_dir>"
os.makedirs(video_dir, exist_ok=True)

for ep in range(num_episodes):
    obs, info = env.reset(seed=ep)
    tmp_path = os.path.join(video_dir, f"_tmp_ep{ep:02d}.mp4")

    recorder.add(env.render(), tmp_path)

    done = False
    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        recorder.add(env.render(), tmp_path)
        done = terminated or truncated

    recorder.close()

    success = bool(info.get("success", False))  # 按环境的实际判定字段
    suffix = "ok" if success else "no"
    final_path = os.path.join(
        video_dir, f"<env>_<tag>_ep{ep:02d}_{suffix}.mp4"
    )
    os.replace(tmp_path, final_path)

    transcode_to_h264(final_path, fps=control_freq)
```

录制的 mp4v 文件在转码后即被原位替换为 H.264 版本，最终磁盘上留下的都是可播放文件。

## 7. 验证与排查

### 7.1 全量编码校验

转码后应校验每个文件的编码确实是 h264、像素格式是 yuv420p，避免个别文件因转码中断残留 mp4v：

```python
import av

def check_encoding(path: str) -> tuple[str, str]:
    with av.open(path) as container:
        stream = container.streams.video[0]
        return stream.codec_context.name, stream.pix_fmt
```

遍历所有 mp4，断言 `codec == "h264"` 且 `pix_fmt == "yuv420p"`。

### 7.2 隐藏目录坑

批量处理或校验时，不要用 `glob.glob("*/*.mp4")`。Python `glob` 的 `*` 默认不匹配以 `.` 开头的隐藏目录，例如 Jupyter 自动生成的 `.ipynb_checkpoints`、某些工具的 `.cache` 等。这些目录里可能缓存着 mp4v 副本，会被 glob 漏掉，导致校验"全通过"但实际仍有不可播放文件残留。

正确做法是用 `os.walk` 遍历，进入所有目录包括隐藏目录：

```python
import os

def iter_video_files(root: str):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".mp4"):
                yield os.path.join(dirpath, name)
```

### 7.3 损坏文件判定

转码或录制中途异常可能留下空文件或损坏文件。判定标准：

- 帧数为 0（用 PyAV 打开后 decode 出 0 帧）。
- 文件体积小于 1 KB。

命中任一条即视为损坏，需重录该 episode。可结合上面的 `iter_video_files` 做全量扫描。

## 8. 对评测流程的影响

录制位于渲染层旁路，独立于策略推理与 `env.step()` 的动作执行：

- 不改变 `obs`，因此不改变策略输入。
- 不改变 `reward`、`terminated`、`truncated`、`info`，因此不改变评测数值。
- 确定性策略下，开录制与关录制得到的成功率、回报等指标相同。

这一点可作为录制正确性的回归验证手段：同一组 seed 跑两遍评测，一遍开录制、一遍不开，比较关键数值是否一致；若不一致，说明录制意外侵入了观测或动作通路，需要排查。

唯一成本是每步多一次 `env.render()` 的渲染与磁盘 IO 开销。EGL 离屏渲染在 GPU 上完成，开销通常在 15% 以内，对长 episode 评测可接受。若评测对时延极度敏感（如实时控制评测），可改为降采样录制或仅在失败 episode 附近触发录制。
