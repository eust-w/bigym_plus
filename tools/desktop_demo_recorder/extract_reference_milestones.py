#!/usr/bin/env python3
"""Extract physical phase boundaries from validated BiGym demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from demonstrations.demo import Demo
from tools.desktop_demo_recorder.env_utils import build_env
from tools.desktop_demo_recorder.privileged_policy import (
    ACTION_MODE_DIR,
    CONTROL_REFERENCE_UUID,
    SUPPORTED_TASK,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo-root",
        type=Path,
        default=Path.home() / ".bigym" / "demonstrations" / "0.9.0",
    )
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--control-reference-uuid",
        default=CONTROL_REFERENCE_UUID,
        help="Reference that needs a dense pinch-relative trajectory",
    )
    args = parser.parse_args()

    validated = {
        json.loads(line)["source_demo_uuid"]
        for line in args.validation_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("status") == "success"
    }
    source_dir = (
        args.demo_root / SUPPORTED_TASK / ACTION_MODE_DIR / "lightweight" / "20hz"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.is_file():
        output = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        output = {}
    env = build_env(SUPPORTED_TASK, with_cameras=False)
    try:
        for uuid in sorted(validated):
            needs_dense_trace = uuid == args.control_reference_uuid
            if (
                uuid in output
                and output[uuid].get("valid")
                and (
                    not needs_dense_trace
                    or output[uuid].get("pinch_relative_xyz")
                )
            ):
                continue
            demo = Demo.from_safetensors(source_dir / f"{uuid}.safetensors")
            env.reset(seed=int(demo.seed))
            drawer_open_step = None
            drawer_close_step = None
            bottom_close_step = None
            door_close_step = None
            tray_entry_step = None
            drawer_was_open = False
            bottom_was_open = float(env.dishwasher.get_state()[1]) > 0.1
            door_was_open = float(env.dishwasher.get_state()[0]) > 0.1
            success_step = None
            pinch_relative_xyz = []
            left_gripper = next(iter(env.robot.grippers.values()))
            for step, timestep in enumerate(demo.timesteps, start=1):
                _, reward, terminated, truncated, _ = env.step(
                    np.asarray(timestep.executed_action, dtype=np.float32)
                )
                drawer_open = not np.allclose(
                    np.asarray(env.cutlery_cabinet.get_state()), 0, atol=0.1
                )
                if drawer_open and not drawer_was_open and drawer_open_step is None:
                    drawer_open_step = step
                if drawer_was_open and not drawer_open:
                    drawer_close_step = step
                drawer_was_open = drawer_open

                dishwasher_state = np.asarray(env.dishwasher.get_state())
                bottom_open = float(dishwasher_state[1]) > 0.1
                door_open = float(dishwasher_state[0]) > 0.1
                if bottom_was_open and not bottom_open:
                    bottom_close_step = step
                if door_was_open and not door_open:
                    door_close_step = step
                bottom_was_open = bottom_open
                door_was_open = door_open

                if tray_entry_step is None and env.cutlery[0].is_colliding(env.tray):
                    tray_entry_step = step
                if needs_dense_trace:
                    fork_position = np.asarray(env.cutlery[0].body.get_position())
                    pinch_position = np.asarray(left_gripper.pinch_position)
                    pinch_relative_xyz.append(
                        np.round(pinch_position - fork_position, 5).tolist()
                    )
                if reward > 0 or terminated:
                    success_step = step
                    break
                if truncated:
                    break

            output[str(uuid)] = {
                "seed": int(demo.seed),
                "drawer_open_step": int(drawer_open_step or 0),
                "drawer_close_step": int(drawer_close_step or 0),
                "bottom_close_step": int(bottom_close_step or 0),
                "door_close_step": int(door_close_step or 0),
                "tray_entry_step": int(tray_entry_step or 0),
                "success_step": int(success_step or 0),
                "valid": bool(success_step),
                "pinch_relative_xyz": pinch_relative_xyz,
            }
            args.output.write_text(
                json.dumps(output, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps({"uuid": uuid, **output[str(uuid)]}), flush=True)
    finally:
        env.close()

    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
