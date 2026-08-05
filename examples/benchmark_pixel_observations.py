"""Smoke-test and benchmark headless BiGym pixel observations."""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np


def summarize_frame(frame: np.ndarray, resolution: int) -> dict:
    """Validate one CHW RGB frame and return JSON-compatible statistics."""
    frame = np.asarray(frame)
    expected_shape = (3, resolution, resolution)
    if frame.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected pixel observation shape: {frame.shape}; "
            f"expected {expected_shape}."
        )
    if not np.issubdtype(frame.dtype, np.number):
        raise RuntimeError(f"Pixel observation is not numeric: {frame.dtype}.")
    if not np.isfinite(frame).all():
        raise RuntimeError("Pixel observation contains non-finite values.")

    minimum = float(frame.min())
    maximum = float(frame.max())
    standard_deviation = float(frame.std())
    if maximum <= minimum or standard_deviation <= 0:
        raise RuntimeError(
            "Pixel observation is blank or constant; check the GL backend and "
            "device access."
        )
    return {
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "minimum": minimum,
        "maximum": maximum,
        "standard_deviation": standard_deviation,
    }


def run_benchmark(steps: int, warmup_steps: int, resolution: int, seed: int) -> dict:
    """Run a deterministic pixel-observation loop and return its report."""
    import mujoco

    from bigym.action_modes import JointPositionActionMode
    from bigym.envs.reach_target import ReachTarget
    from bigym.utils.observation_config import CameraConfig, ObservationConfig

    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=False),
        observation_config=ObservationConfig(
            cameras=[CameraConfig(name="head", resolution=(resolution, resolution))],
            proprioception=False,
        ),
        render_mode=None,
    )
    try:
        observation, _ = env.reset(seed=seed)
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        for _ in range(warmup_steps):
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset(seed=seed)

        started = time.perf_counter()
        for _ in range(steps):
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset(seed=seed)
        elapsed_seconds = time.perf_counter() - started
    finally:
        env.close()

    frame = observation["rgb_head"]
    return {
        "gate_passed": True,
        "platform": platform.platform(),
        "mujoco_version": mujoco.__version__,
        "render_environment": {
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
        },
        "steps": steps,
        "warmup_steps": warmup_steps,
        "resolution": resolution,
        "elapsed_seconds": elapsed_seconds,
        "steps_per_second": steps / elapsed_seconds,
        "frame": summarize_frame(frame, resolution),
    }


def main() -> None:
    """Parse arguments, run the benchmark, and print a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    if args.resolution <= 0:
        parser.error("--resolution must be positive")
    print(
        json.dumps(
            run_benchmark(
                args.steps,
                args.warmup_steps,
                args.resolution,
                args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
