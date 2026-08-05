"""Tests for the DemoRecorder lifecycle."""
from pathlib import Path
from types import SimpleNamespace

import pytest

import demonstrations.demo_recorder as demo_recorder_module
from demonstrations.demo_recorder import DemoRecorder


class _FakeDemo:
    def __init__(self, *, fail_save=False):
        self.metadata = SimpleNamespace(filename="demo.safetensors")
        self.fail_save = fail_save

    def save(self, path):
        if self.fail_save:
            raise RuntimeError("simulated save failure")
        path = Path(path)
        path.write_bytes(b"demo")
        return path


def test_successful_save_stops_recording_and_allows_next_recording(
    monkeypatch, tmp_path
):
    demos = [_FakeDemo(), _FakeDemo()]
    monkeypatch.setattr(
        demo_recorder_module.Demo,
        "from_env",
        classmethod(lambda _cls, _env: demos.pop(0)),
    )
    recorder = DemoRecorder(tmp_path)
    environment = object()

    recorder.record(environment)
    assert recorder.is_recording

    path = recorder.save_demo()

    assert path == tmp_path / "demo.safetensors"
    assert not recorder.is_recording
    assert recorder.demo is None

    recorder.record(environment)
    assert recorder.is_recording
    assert recorder.demo is not None


def test_failed_save_keeps_active_recording_for_retry(monkeypatch, tmp_path):
    demo = _FakeDemo(fail_save=True)
    monkeypatch.setattr(
        demo_recorder_module.Demo,
        "from_env",
        classmethod(lambda _cls, _env: demo),
    )
    recorder = DemoRecorder(tmp_path)
    recorder.record(object())

    with pytest.raises(RuntimeError, match="simulated save failure"):
        recorder.save_demo()

    assert recorder.is_recording
    assert recorder.demo is demo
