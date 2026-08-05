import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np


ROOT = Path(__file__).parents[1]
EVAL_DIR = ROOT / "d" / "eval"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"response": [[0.0] * 16], "request_id": "server-request-1"}


class _RolloutRecorder:
    def __init__(self, fps):
        self.fps = fps

    def add(self, _frame, path):
        Path(path).write_bytes(b"video")

    def close(self):
        return None


class _Environment:
    def __init__(self):
        self.action_mode = types.SimpleNamespace(
            _mojo=types.SimpleNamespace()
        )
        self.action_space = types.SimpleNamespace(
            low=np.full(16, -1.0), high=np.full(16, 1.0)
        )
        self.robot = object()
        self.closed = False

    @staticmethod
    def _observation():
        image = np.zeros((3, 2, 2), dtype=np.uint8)
        return {
            "rgb_high": image,
            "rgb_l_wrist": image,
            "rgb_r_wrist": image,
        }

    def reset(self, seed):
        return self._observation(), {"reset_seed": seed}

    def render(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def step(self, _action):
        return self._observation(), 1.0, False, False, {"task_success": True}

    def close(self):
        self.closed = True


def test_eval_writes_action_aligned_trajectory(monkeypatch, tmp_path):
    cv2 = types.ModuleType("cv2")
    cv2.imencode = lambda _extension, _image: (
        True,
        np.frombuffer(b"png", dtype=np.uint8),
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    requests = types.ModuleType("requests")
    requests.post = lambda *_args, **_kwargs: _Response()
    monkeypatch.setitem(sys.modules, "requests", requests)

    tasks = types.ModuleType("tasks")
    tasks.TASKS = {"FakeTask": {"prompt": "test prompt"}}
    tasks.FREQ = 20
    tasks.resolve_task = lambda task: task
    tasks.get_maxstep = lambda _task: 2
    tasks.task_to_snake = lambda task: task.lower()
    monkeypatch.setitem(sys.modules, "tasks", tasks)

    environment = _Environment()
    env_utils = types.ModuleType("env_utils")
    env_utils.build_env = lambda *_args, **_kwargs: environment
    env_utils.get_state = lambda *_args: np.arange(16, dtype=np.float32)
    env_utils.CAMERAS = [("high",), ("l_wrist",), ("r_wrist",)]
    monkeypatch.setitem(sys.modules, "env_utils", env_utils)

    dim_utils = types.ModuleType("dim_utils")
    dim_utils.pad_to_16 = lambda value, _task: np.asarray(value, dtype=np.float32)
    dim_utils.drop_z = lambda value, _task: np.asarray(value, dtype=np.float32)
    monkeypatch.setitem(sys.modules, "dim_utils", dim_utils)

    rollout_video = types.ModuleType("rollout_video")
    rollout_video.RolloutRecorder = _RolloutRecorder
    rollout_video.transcode_to_h264 = lambda *_args, **_kwargs: None
    rollout_video.video_path = lambda directory, task, tag, episode, success: str(
        Path(directory)
        / f"{task}_{tag}_ep{episode:02d}_{'ok' if success else 'no'}.mp4"
    )
    monkeypatch.setitem(sys.modules, "rollout_video", rollout_video)

    _load("trajectory_log", EVAL_DIR / "trajectory_log.py")
    evaluator = _load("eval_bigym_test", EVAL_DIR / "eval_bigym.py")

    rate, successes = evaluator.run_eval(
        "FakeTask",
        "http://policy.invalid",
        1,
        2,
        10,
        str(tmp_path),
        "audit",
        True,
    )

    episode = tmp_path / "trajectories" / "audit" / "episode-000000"
    records = [
        json.loads(line)
        for line in (episode / "steps.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((episode / "manifest.json").read_text())

    assert rate == 1.0
    assert successes == [True]
    assert environment.closed
    assert [record["record_type"] for record in records] == [
        "reset",
        "transition",
    ]
    transition = records[1]
    assert transition["state16_before"] == list(range(16))
    assert transition["state16_after"] == list(range(16))
    assert transition["request_id"]
    assert transition["server_request_id"] == "server-request-1"
    assert transition["action_chunk_index"] == 0
    assert manifest["status"] == "complete"
    assert manifest["success"] is True
    assert manifest["transitions"] == 1
