import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "d" / "eval" / "trajectory_log.py"
SPEC = importlib.util.spec_from_file_location("trajectory_log", MODULE_PATH)
assert SPEC and SPEC.loader
trajectory_log = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trajectory_log)


def test_episode_log_is_append_only_and_terminal_manifest_is_atomic():
    with tempfile.TemporaryDirectory() as directory:
        log = trajectory_log.EpisodeTrajectoryLog(
            directory,
            episode_index=2,
            seed=42,
            task="ReachTargetDual",
            metadata={"backend": "test"},
        )
        log.append(
            "reset",
            state16=np.zeros(16, dtype=np.float32),
            info={"nested": np.int64(3)},
        )
        log.append(
            "transition",
            state16_before=np.zeros(16),
            state16_after=np.ones(16),
            action16_model=np.full(16, 0.25),
            reward=np.float32(1.0),
            info={"nonfinite": np.inf},
        )
        log.finish(
            status="complete",
            success=True,
            transitions=1,
            failure_reason=None,
        )

        records = [
            json.loads(line)
            for line in log.steps_path.read_text(encoding="utf-8").splitlines()
        ]
        manifest = json.loads(log.manifest_path.read_text(encoding="utf-8"))

        assert [record["record_index"] for record in records] == [0, 1]
        assert records[1]["info"]["nonfinite"] == "Infinity"
        assert manifest["status"] == "complete"
        assert manifest["record_count"] == 2
        assert manifest["success"] is True
        assert manifest["transitions"] == 1
        assert len(manifest["steps_jsonl"]["sha256"]) == 64
        assert not list(log.directory.glob("*.tmp"))


def test_existing_episode_is_never_silently_overwritten():
    with tempfile.TemporaryDirectory() as directory:
        trajectory_log.EpisodeTrajectoryLog(
            directory,
            episode_index=0,
            seed=0,
            task="ReachTargetDual",
        )
        with pytest.raises(FileExistsError, match="choose a new --tag"):
            trajectory_log.EpisodeTrajectoryLog(
                directory,
                episode_index=0,
                seed=0,
                task="ReachTargetDual",
            )


def test_closed_log_rejects_more_records_and_invalid_terminal_status():
    with tempfile.TemporaryDirectory() as directory:
        log = trajectory_log.EpisodeTrajectoryLog(
            directory,
            episode_index=0,
            seed=0,
            task="ReachTargetDual",
        )
        with pytest.raises(ValueError, match="invalid terminal status"):
            log.finish(
                status="recording",
                success=False,
                transitions=0,
                failure_reason="test",
            )
        log.finish(
            status="interrupted",
            success=False,
            transitions=0,
            failure_reason="keyboard_interrupt",
        )
        with pytest.raises(RuntimeError, match="terminal"):
            log.append("transition")
