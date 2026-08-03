#!/usr/bin/env python3
"""Browser-based desktop teleoperation for BiGym demonstration collection."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import queue
import random
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from aiohttp import WSMsgType, web

from demonstrations.demo_recorder import DemoRecorder
from tools.desktop_demo_recorder.enhanced_randomization import (
    RandomizationRecord,
    apply_enhanced_randomization,
)
from tools.desktop_demo_recorder.env_utils import build_env, get_state
from tools.desktop_demo_recorder.privileged_policy import (
    DEFAULT_DEMO_UUID,
    SUPPORTED_TASK,
    PrivilegedExpertPolicy,
)


LOG = logging.getLogger("bigym-desktop-teleop")
CAMERA_KEYS = (
    "rgb_head",
    "rgb_external",
    "rgb_left_wrist",
    "rgb_right_wrist",
)
ARM_NAMES = (
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist",
)
ARM_KEY_PAIRS = (
    ("KeyT", "KeyG"),
    ("KeyY", "KeyH"),
    ("KeyU", "KeyJ"),
    ("KeyI", "KeyK"),
    ("KeyO", "KeyL"),
)


@dataclass
class PublicState:
    task: str
    seed: int
    episode_step: int = 0
    recording: bool = False
    success: bool = False
    success_step: int | None = None
    paused: bool = False
    terminated: bool = False
    truncated: bool = False
    reward: float = 0.0
    last_saved: str | None = None
    message: str = "ready"
    control_hz: int = 20
    stream_fps: float = 0.0
    policy_available: bool = False
    policy_active: bool = False
    policy_stage: str = "待命"
    policy_step: int = 0
    policy_total: int = 0
    policy_demo_uuid: str | None = None
    policy_reference_seed: int | None = None
    policy_pose_score: float | None = None
    policy_reference_count: int = 0
    collection_source: str = "manual"


class TeleopController:
    def __init__(
        self,
        task: str,
        output_dir: Path,
        seed: int = 200000,
        expert_demo_root: Path | None = None,
        expert_demo_uuid: str = DEFAULT_DEMO_UUID,
    ):
        self.task = task
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.commands: queue.Queue = queue.Queue()
        self.init_error: BaseException | None = None
        self.env = None
        self.mojo = None
        self.recorder = None
        self.keys: set[str] = set()
        self.obs: dict[str, Any] = {}
        self.images: dict[str, str] = {}
        self.arm_targets = np.zeros(10, dtype=np.float32)
        self.gripper_targets = np.zeros(2, dtype=np.float32)
        self.action_low = np.zeros(16, dtype=np.float32)
        self.action_high = np.zeros(16, dtype=np.float32)
        demo_root = expert_demo_root or (
            Path.home() / ".bigym" / "demonstrations" / "0.9.0"
        )
        self.policy = PrivilegedExpertPolicy(task, demo_root, expert_demo_uuid)
        self.policy.validation_manifest = (
            output_dir / "privileged_batch" / "manifest.jsonl"
        )
        self.policy_privileged: dict[str, Any] = {}
        self.randomization_record: RandomizationRecord | None = None
        self.state = PublicState(
            task=task,
            seed=seed,
            policy_available=self.policy.available,
            policy_reference_count=self.policy.reference_count,
        )
        self._last_frame_time = time.monotonic()
        self._frame_counter = 0
        self.worker = threading.Thread(target=self._run_loop, daemon=True)
        self.worker.start()
        if not self.ready_event.wait(timeout=60.0):
            raise TimeoutError("Timed out while creating the BiGym environment")
        if self.init_error is not None:
            raise RuntimeError("Failed to create the BiGym environment") from self.init_error

    def _reset_locked(self, seed: int) -> None:
        self.policy.reset_controller_state()
        self.keys.clear()
        self.recorder.stop()
        self.recorder = DemoRecorder(self.output_dir)
        self.env.reset(seed=int(seed))
        self.randomization_record = apply_enhanced_randomization(self.env, int(seed))
        obs = self.env.get_observation()
        self.obs = obs
        robot_state = get_state(self.env.robot, obs, self.mojo)
        self.arm_targets = robot_state[4:14].astype(np.float32, copy=True)
        self.gripper_targets = robot_state[14:16].astype(np.float32, copy=True)
        self.policy_privileged = self._privileged_status_locked()
        self.state = PublicState(
            task=self.task,
            seed=int(seed),
            message="new episode",
            policy_available=self.policy.available,
            policy_stage=self.policy_privileged.get("stage", "待命"),
            policy_reference_count=self.policy.reference_count,
        )
        self._encode_images_locked(force=True)

    def _privileged_status_locked(self) -> dict[str, Any]:
        status = self.policy.privileged_status(self.env)
        if self.randomization_record is not None:
            status["randomization"] = self.randomization_record.to_dict()
        return status

    def reset(self, seed: int | None = None) -> None:
        self._submit("reset", seed if seed is not None else random.randint(0, 2**31 - 1))

    def start_recording(self, seed: int | None = None) -> None:
        self._submit("record", seed if seed is not None else random.randint(0, 2**31 - 1))

    def discard(self) -> None:
        self._submit("discard")

    def save(self) -> str:
        return self._submit("save")

    def start_policy(self, seed: int | None = None) -> None:
        self._submit("policy_start", self.state.seed if seed is None else seed)

    def stop_policy(self) -> None:
        self._submit("policy_stop")

    def set_controls(self, payload: dict[str, Any]) -> None:
        with self.lock:
            if "keys" in payload:
                base_keys = {"KeyW", "KeyS", "KeyA", "KeyD", "KeyR", "KeyF", "KeyQ", "KeyE"}
                arm_keys = {code for pair in ARM_KEY_PAIRS for code in pair}
                allowed = base_keys | arm_keys | {f"Shift+{code}" for code in arm_keys}
                self.keys = set(payload["keys"]) & allowed
            if "arms" in payload:
                values = np.asarray(payload["arms"], dtype=np.float32)
                if values.shape != (10,) or not np.isfinite(values).all():
                    raise ValueError("arms must contain 10 finite values")
                self.arm_targets = np.clip(values, self.action_low[4:14], self.action_high[4:14])
            if "grippers" in payload:
                values = np.asarray(payload["grippers"], dtype=np.float32)
                if values.shape != (2,) or not np.isfinite(values).all():
                    raise ValueError("grippers must contain 2 finite values")
                self.gripper_targets = np.clip(
                    values, self.action_low[14:16], self.action_high[14:16]
                )

    def emergency_stop(self) -> None:
        self._submit("emergency")

    def toggle_pause(self) -> None:
        self._submit("pause")

    def _submit(self, name: str, value: Any = None) -> Any:
        done = threading.Event()
        result: dict[str, Any] = {}
        self.commands.put((name, value, done, result))
        if not done.wait(timeout=30.0):
            raise TimeoutError(f"Timed out handling command: {name}")
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def _handle_command_locked(self, name: str, value: Any) -> Any:
        if name == "reset":
            self._reset_locked(int(value))
            self.state.paused = True
            self.state.message = "new episode ready"
        elif name == "record":
            self._reset_locked(int(value))
            self.recorder.record(self.env, lightweight_demo=True)
            self.state.recording = True
            self.state.paused = False
            self.state.message = "recording"
        elif name == "discard":
            self.policy.stop()
            self.recorder.stop()
            self.recorder = DemoRecorder(self.output_dir)
            self.state.recording = False
            self.state.policy_active = False
            self.state.policy_step = 0
            self.state.policy_total = 0
            self.state.policy_demo_uuid = None
            self.state.policy_reference_seed = None
            self.state.policy_pose_score = None
            self.state.policy_stage = self.policy_privileged.get("stage", "待命")
            self.state.collection_source = "manual"
            self.state.paused = True
            self.state.message = "recording discarded"
        elif name == "emergency":
            self.policy.stop()
            self.keys.clear()
            self.state.policy_active = False
            self.state.paused = True
            self.state.message = "paused by emergency stop"
        elif name == "pause":
            self.keys.clear()
            self.state.paused = not self.state.paused
            self.state.message = "paused" if self.state.paused else "running"
        elif name == "policy_start":
            if not self.policy.available:
                raise ValueError(f"No validated expert is available for {self.task}")
            self._reset_locked(int(value))
            self.policy.start(self.env, self.env.action_space)
            self.policy_privileged = self._privileged_status_locked()
            self.recorder.record(self.env, lightweight_demo=True)
            self.state.recording = True
            self.state.paused = False
            self.state.policy_active = True
            self.state.policy_step = 0
            self.state.policy_total = self.policy.total_steps
            self.state.policy_demo_uuid = self.policy.demo_uuid
            self.state.policy_reference_seed = self.policy.reference_seed
            self.state.policy_pose_score = self.policy.pose_score
            self.state.policy_reference_count = self.policy.reference_count
            self.state.policy_stage = self.policy_privileged.get("stage", "自动示教")
            self.state.collection_source = "privileged_closed_loop"
            self.state.message = "privileged closed-loop policy running"
        elif name == "policy_stop":
            self.policy.stop()
            self.keys.clear()
            self.state.policy_active = False
            self.state.paused = True
            self.state.message = "privileged expert demonstration stopped"
        elif name == "save":
            if not self.state.recording:
                raise ValueError("No active recording")
            if not self.state.success:
                raise ValueError("Save rejected: the task has not succeeded")
            self.recorder.stop()
            self.policy.stop()
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            filename = f"{self.task}_seed{self.state.seed}_{stamp}.safetensors"
            path = self.recorder.save_demo(filename)
            if path is None:
                raise RuntimeError("Recorder returned no output path")
            manifest = {
                "file": str(path),
                "task": self.task,
                "seed": self.state.seed,
                "episode_steps": self.state.episode_step,
                "success": True,
                "success_step": self.state.success_step,
                "reward": self.state.reward,
                "control_hz": self.state.control_hz,
                "collection_source": self.state.collection_source,
                "policy_demo_uuid": self.state.policy_demo_uuid,
                "policy_reference_seed": self.state.policy_reference_seed,
                "policy_pose_score": self.state.policy_pose_score,
                "randomization": (
                    self.randomization_record.to_dict()
                    if self.randomization_record is not None
                    else None
                ),
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            with (self.output_dir / "manifest.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(manifest, ensure_ascii=False) + "\n")
            self.state.recording = False
            self.state.policy_active = False
            self.state.paused = True
            self.state.last_saved = str(path)
            self.state.message = "successful demo saved"
            return str(path)
        else:
            raise ValueError(f"Unknown controller command: {name}")
        return None

    def _build_action_locked(self) -> np.ndarray:
        if self.policy.active:
            action = self.policy.next_action(self.env)
            if action is None:
                self.state.policy_active = False
                self.state.paused = True
                self.state.message = (
                    self.policy.failure_reason
                    or "closed-loop controller stopped without task success"
                )
                return np.concatenate(
                    (np.zeros(4, dtype=np.float32), self.arm_targets, self.gripper_targets)
                )
            self.arm_targets = action[4:14].astype(np.float32, copy=True)
            self.gripper_targets = action[14:16].astype(np.float32, copy=True)
            self.state.policy_step = self.policy.control_steps
            return np.clip(action, self.action_low, self.action_high).astype(np.float32)

        base = np.zeros(4, dtype=np.float32)
        base[0] = 0.01 * (("KeyW" in self.keys) - ("KeyS" in self.keys))
        base[1] = 0.01 * (("KeyA" in self.keys) - ("KeyD" in self.keys))
        base[2] = 0.01 * (("KeyR" in self.keys) - ("KeyF" in self.keys))
        base[3] = 0.03 * (("KeyQ" in self.keys) - ("KeyE" in self.keys))
        for joint, (positive, negative) in enumerate(ARM_KEY_PAIRS):
            self.arm_targets[joint] += 0.01 * (
                (positive in self.keys) - (negative in self.keys)
            )
            self.arm_targets[joint + 5] += 0.01 * (
                (f"Shift+{positive}" in self.keys) - (f"Shift+{negative}" in self.keys)
            )
        self.arm_targets = np.clip(
            self.arm_targets, self.action_low[4:14], self.action_high[4:14]
        )
        action = np.concatenate((base, self.arm_targets, self.gripper_targets))
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    def _encode_images_locked(self, force: bool = False) -> None:
        if not force and self.state.episode_step % 3:
            return
        encoded: dict[str, str] = {}
        for key in CAMERA_KEYS:
            frame = np.asarray(self.obs[key])
            if frame.ndim == 3 and frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))
            frame = np.ascontiguousarray(frame[:, :, ::-1])
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 76])
            if ok:
                encoded[key.removeprefix("rgb_")] = base64.b64encode(jpeg).decode("ascii")
        self.images = encoded
        self._frame_counter += 1
        now = time.monotonic()
        elapsed = now - self._last_frame_time
        if elapsed >= 2.0:
            self.state.stream_fps = round(self._frame_counter / elapsed, 1)
            self._frame_counter = 0
            self._last_frame_time = now

    def _run_loop(self) -> None:
        try:
            self.env = build_env(self.task, control_frequency=20, with_cameras=True)
            self.mojo = self.env.action_mode._mojo
            self.policy.bind_env(self.env)
            self.recorder = DemoRecorder(self.output_dir)
            self.action_low = np.asarray(self.env.action_space.low, dtype=np.float32)
            self.action_high = np.asarray(self.env.action_space.high, dtype=np.float32)
            with self.lock:
                self._reset_locked(self.state.seed)
                self.state.paused = True
                self.state.message = "ready"
            self.ready_event.set()
            period = 1.0 / self.state.control_hz
            while not self.stop_event.is_set():
                started = time.monotonic()
                with self.lock:
                    while True:
                        try:
                            name, value, done, result = self.commands.get_nowait()
                        except queue.Empty:
                            break
                        try:
                            result["value"] = self._handle_command_locked(name, value)
                        except Exception as exc:
                            result["error"] = exc
                        finally:
                            done.set()
                    if (
                        not self.state.paused
                        and not self.state.terminated
                        and not self.state.truncated
                    ):
                        try:
                            action = self._build_action_locked()
                            timestep = self.env.step(action)
                            self.obs = timestep[0]
                            reward = float(timestep[1])
                            self.state.reward = reward
                            self.state.episode_step += 1
                            self.state.terminated = bool(timestep[2])
                            self.state.truncated = bool(timestep[3])
                            self.policy_privileged = self._privileged_status_locked()
                            self.state.policy_stage = self.policy_privileged.get(
                                "stage", self.state.policy_stage
                            )
                            if self.state.recording:
                                self.recorder.add_timestep(timestep, action)
                            if reward > 0 and not self.state.success:
                                self.state.success = True
                                self.state.success_step = self.state.episode_step
                                self.state.message = "task succeeded; save is enabled"
                                if self.policy.active:
                                    self.policy.stop()
                                    self.state.policy_active = False
                                    self.state.paused = True
                            if self.state.terminated or self.state.truncated:
                                self.policy.stop()
                                self.state.policy_active = False
                                self.keys.clear()
                                self.state.paused = True
                                if not self.state.success:
                                    self.state.message = "episode ended without success"
                            self._encode_images_locked()
                        except Exception as exc:
                            LOG.exception("Simulation step failed")
                            self.policy.stop()
                            self.keys.clear()
                            self.state.policy_active = False
                            self.state.paused = True
                            self.state.message = f"simulation error: {exc}"
                remaining = period - (time.monotonic() - started)
                if remaining > 0:
                    self.stop_event.wait(remaining)
        except BaseException as exc:
            self.init_error = exc
            LOG.exception("Failed to initialize simulation")
            self.ready_event.set()
        finally:
            if self.recorder is not None:
                self.recorder.stop()
            if self.env is not None:
                self.env.close()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "state": asdict(self.state),
                "images": self.images.copy(),
                "controls": {
                    "arms": self.arm_targets.tolist(),
                    "grippers": self.gripper_targets.tolist(),
                    "arm_names": ARM_NAMES,
                    "arm_low": self.action_low[4:14].tolist(),
                    "arm_high": self.action_high[4:14].tolist(),
                    "keys": sorted(self.keys),
                    "display_keys": sorted(
                        self.policy.virtual_keys if self.policy.active else self.keys
                    ),
                    "source": (
                        "policy"
                        if self.state.collection_source.startswith("privileged_")
                        else "manual"
                    ),
                },
                "privileged": self.policy_privileged.copy(),
            }

    def close(self) -> None:
        self.stop_event.set()
        self.worker.join(timeout=2.0)


def create_app(controller: TeleopController, static_dir: Path) -> web.Application:
    app = web.Application(client_max_size=4 * 1024 * 1024)

    async def index(_request):
        return web.FileResponse(static_dir / "index.html")

    async def state(_request):
        return web.json_response(controller.snapshot())

    async def websocket_endpoint(request):
        ws = web.WebSocketResponse(
            max_msg_size=4 * 1024 * 1024,
            heartbeat=20.0,
            compress=False,
        )
        await ws.prepare(request)

        async def sender():
            while True:
                await ws.send_json({"type": "snapshot", **controller.snapshot()})
                await asyncio.sleep(0.18)

        send_task = asyncio.create_task(sender())
        try:
            async for message in ws:
                if message.type == WSMsgType.ERROR:
                    LOG.warning("WebSocket closed with error: %s", ws.exception())
                    break
                if message.type != WSMsgType.TEXT:
                    continue
                payload = json.loads(message.data)
                command = payload.get("command")
                try:
                    if command == "control":
                        controller.set_controls(payload)
                    elif command == "reset":
                        await asyncio.to_thread(controller.reset, payload.get("seed"))
                    elif command == "record":
                        await asyncio.to_thread(controller.start_recording, payload.get("seed"))
                    elif command == "discard":
                        await asyncio.to_thread(controller.discard)
                    elif command == "save":
                        path = await asyncio.to_thread(controller.save)
                        await ws.send_json({"type": "result", "ok": True, "message": path})
                    elif command == "pause":
                        await asyncio.to_thread(controller.toggle_pause)
                    elif command == "emergency":
                        await asyncio.to_thread(controller.emergency_stop)
                    elif command == "policy_start":
                        await asyncio.to_thread(controller.start_policy, payload.get("seed"))
                    elif command == "policy_stop":
                        await asyncio.to_thread(controller.stop_policy)
                    else:
                        await ws.send_json(
                            {
                                "type": "result",
                                "ok": False,
                                "message": "unknown command",
                            }
                        )
                except ValueError as exc:
                    LOG.info("Command rejected: %s", exc)
                    await ws.send_json({"type": "result", "ok": False, "message": str(exc)})
                except Exception as exc:
                    LOG.exception("WebSocket command failed")
                    await ws.send_json({"type": "result", "ok": False, "message": str(exc)})
        finally:
            controller.set_controls({"keys": []})
            send_task.cancel()
        return ws

    async def shutdown(_app):
        controller.close()

    app.router.add_get("/", index)
    app.router.add_get("/api/state", state)
    app.router.add_get("/ws", websocket_endpoint)
    app.router.add_static("/static", static_dir)
    app.on_shutdown.append(shutdown)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--task", default=SUPPORTED_TASK, choices=[SUPPORTED_TASK])
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--expert-demo-root",
        type=Path,
        default=Path.home() / ".bigym" / "demonstrations" / "0.9.0",
    )
    parser.add_argument("--expert-demo-uuid", default=DEFAULT_DEMO_UUID)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    output_dir = args.output_dir or Path.cwd() / "demos" / args.task
    ctl = TeleopController(
        args.task,
        output_dir,
        args.seed,
        args.expert_demo_root,
        args.expert_demo_uuid,
    )
    app = create_app(ctl, Path(__file__).resolve().parent / "static")
    web.run_app(app, host=args.host, port=args.port, print=LOG.info)
