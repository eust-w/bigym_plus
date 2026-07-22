"""headless 评测回灌视频录制 (严格按 video_log.md).

录: cv2.VideoWriter mp4v (opencv 自带 ffmpeg, 零配置), 懒打开, RGB→BGR.
转码: PyAV libx264 + yuv420p + baseline (浏览器/系统预览/移动端通用可播放), 原位替换.
校验: 全量扫描断言 h264 + yuv420p, 用 os.walk (含隐藏目录, 不用 glob).

命名: {env}_{tag}_ep{NN}_{ok|no}.mp4 — ok/no 到 episode 结束才定, 故先写临时路径,
结束后 os.replace 改名为带后缀的最终路径.
"""
import os
from typing import Optional

import cv2
import numpy as np


class RolloutRecorder:
    """每 episode 一个 mp4, 懒打开 VideoWriter (首帧到来时按实际尺寸建)."""

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
        """frame: HWC RGB uint8. path 同一 episode 内保持一致."""
        if self._writer is None:
            self._open(frame, path)
        bgr = frame[:, :, ::-1]  # RGB -> BGR (OpenCV 约定)
        self._writer.write(bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def transcode_to_h264(src_path: str, fps: Optional[float] = None) -> None:
    """mp4v → H.264 + yuv420p + baseline, 原位替换 (先 .tmp.mp4 后 os.replace)."""
    from fractions import Fraction
    import av
    tmp_path = src_path + ".tmp.mp4"

    input_container = av.open(src_path)
    in_stream = input_container.streams.video[0]
    src_fps = fps if fps is not None else float(in_stream.average_rate)
    width, height = in_stream.width, in_stream.height
    # pyav 15: rate 须 Fraction (lerobot dataset_tools.py 同款)
    if abs(src_fps - round(src_fps)) < 1e-6:
        rate = Fraction(int(round(src_fps)))
    else:
        rate = Fraction(src_fps).limit_denominator(1000)

    output_container = av.open(tmp_path, mode="w")
    out_stream = output_container.add_stream(
        "libx264", rate=rate,
        options={"preset": "veryfast", "crf": "23", "profile": "baseline"},
    )
    out_stream.width = width
    out_stream.height = height
    out_stream.pix_fmt = "yuv420p"

    for frame in input_container.decode(video=0):
        output_container.mux(out_stream.encode(frame))
    output_container.mux(out_stream.encode())  # flush

    input_container.close()
    output_container.close()
    os.replace(tmp_path, src_path)


def check_encoding(path: str) -> tuple:
    """返回 (codec_name, pix_fmt)."""
    import av
    with av.open(path) as container:
        stream = container.streams.video[0]
        return stream.codec_context.name, stream.pix_fmt


def iter_video_files(root: str):
    """os.walk 遍历 (含隐藏目录, 不用 glob — glob 的 * 漏 . 开头目录)."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".mp4"):
                yield os.path.join(dirpath, name)


def check_all_videos(root: str) -> dict:
    """全量校验: 断言每个 mp4 是 h264 + yuv420p. 返回 {path: (codec, pix_fmt, ok)}."""
    report = {}
    for path in iter_video_files(root):
        codec, pix_fmt = check_encoding(path)
        ok = (codec == "h264") and (pix_fmt == "yuv420p")
        report[path] = (codec, pix_fmt, ok)
    return report


def video_path(video_dir: str, env: str, tag: str, ep: int, success: bool) -> str:
    """{env}_{tag}_ep{NN}_{ok|no}.mp4."""
    suffix = "ok" if success else "no"
    return os.path.join(
        video_dir, f"{env}_{tag}_ep{ep:02d}_{suffix}.mp4"
    )


if __name__ == "__main__":
    import sys
    # 自检: 生成一个 mp4v 测试视频 → 转码 → 校验 h264/yuv420p
    test_dir = "/tmp/_rollout_video_test"
    os.makedirs(test_dir, exist_ok=True)
    src = os.path.join(test_dir, "test_ep00.mp4")
    rec = RolloutRecorder(fps=20.0)
    for i in range(10):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 25  # 渐变红
        rec.add(frame, src)
    rec.close()
    print(f"recorded (mp4v): {src}")
    transcode_to_h264(src, fps=20.0)
    codec, pix = check_encoding(src)
    print(f"transcoded: codec={codec} pix_fmt={pix} "
          f"OK={'YES' if codec=='h264' and pix=='yuv420p' else 'NO'}")
    if "--keep" not in sys.argv:
        os.remove(src)
        os.rmdir(test_dir)
