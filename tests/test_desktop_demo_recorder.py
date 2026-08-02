import json
from pathlib import Path

import numpy as np

from tools.desktop_demo_recorder.enhanced_randomization import (
    apply_enhanced_randomization,
)


TOOL_DIR = Path(__file__).parents[1] / "tools" / "desktop_demo_recorder"


class _Site:
    def __init__(self, position):
        self._position = np.asarray(position, dtype=np.float64)

    def get_position(self):
        return self._position.copy()


class _Body:
    def __init__(self):
        self.position = None
        self.quaternion = None

    def set_position(self, position, _reset):
        self.position = np.asarray(position, dtype=np.float64)

    def set_quaternion(self, quaternion, _reset):
        self.quaternion = np.asarray(quaternion, dtype=np.float64)


class _Physics:
    def __init__(self):
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1


class _Dishwasher:
    def __init__(self):
        self.state = None
        self.basket = type(
            "Basket",
            (),
            {
                "site_sets": [
                    [
                        _Site([0.0, 0.0, 0.43]),
                        _Site([0.04, -0.20, 0.43]),
                        _Site([0.08, -0.20, 0.43]),
                    ]
                ]
            },
        )()

    def set_state(self, **state):
        self.state = state


class _Environment:
    def __init__(self):
        self.dishwasher = _Dishwasher()
        self.cutlery = [type("Cutlery", (), {"body": _Body()})()]
        physics = _Physics()
        self.action_mode = type(
            "ActionMode",
            (),
            {"_mojo": type("Mojo", (), {"physics": physics})()},
        )()


def test_enhanced_randomization_is_deterministic_and_task_valid():
    left = _Environment()
    right = _Environment()

    left_record = apply_enhanced_randomization(left, seed=510102)
    right_record = apply_enhanced_randomization(right, seed=510102)

    assert left_record == right_record
    assert left_record.dishwasher_door == 1.0
    assert 0.95 <= left_record.dishwasher_bottom_tray <= 1.0
    assert -40.0 <= left_record.fork_angle_deg <= 40.0
    assert left.dishwasher.state["middle_tray"] == 0.0
    assert left.action_mode._mojo.physics.forward_calls == 2
    assert left.cutlery[0].body.position.shape == (3,)
    assert left.cutlery[0].body.quaternion.shape == (4,)


def test_reference_milestones_are_complete():
    milestones = json.loads(
        (TOOL_DIR / "reference_milestones.json").read_text(encoding="utf-8")
    )

    assert len(milestones) == 30
    assert all(record["valid"] for record in milestones.values())
    assert all(record["success_step"] > 0 for record in milestones.values())
    assert milestones["199949bc66134ef8881c72928605bf2d"]["pinch_relative_xyz"]


def test_operator_page_contains_four_cameras_and_closed_loop_controls():
    page = (TOOL_DIR / "static" / "index.html").read_text(encoding="utf-8")

    for camera in ("head", "external", "left_wrist", "right_wrist"):
        assert camera in page
    assert "policy_start" in page
    assert "启动自动示教" in page
    assert "randomization" in page
