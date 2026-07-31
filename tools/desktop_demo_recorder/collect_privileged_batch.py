#!/usr/bin/env python3
"""Regenerate validated DishwasherUnloadCutleryLong demos in the current simulator."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from demonstrations.demo import Demo
from demonstrations.demo_recorder import DemoRecorder
from tools.desktop_demo_recorder.env_utils import build_env
from tools.desktop_demo_recorder.privileged_policy import (
    ACTION_MODE_DIR,
    SUPPORTED_TASK,
    dishwasher_unload_cutlery_status,
)


LOG = logging.getLogger("privileged-batch-collector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay official expert actions through the current BiGym physics, "
            "supervise them with privileged task state, and save successes only."
        )
    )
    parser.add_argument(
        "--demo-root",
        type=Path,
        default=Path.home() / ".bigym" / "demonstrations" / "0.9.0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "demos" / SUPPORTED_TASK / "privileged_batch",
    )
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--demo-uuid", action="append", default=[])
    return parser.parse_args()


def source_directory(demo_root: Path) -> Path:
    return (
        demo_root
        / SUPPORTED_TASK
        / ACTION_MODE_DIR
        / "lightweight"
        / "20hz"
    )


def load_records(manifest_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not manifest_path.is_file():
        return records
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        records.append(record)
    return records


def completed_demo_uuids(records: list[dict[str, Any]]) -> set[str]:
    return {
        str(record.get("source_demo_uuid"))
        for record in records
        if record.get("status") in {"success", "failed", "skipped"}
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    counts = {
        status: sum(record["status"] == status for record in records)
        for status in ("success", "failed", "error", "skipped")
    }
    payload = {
        "task": SUPPORTED_TASK,
        "total_records": len(records),
        "counts": counts,
        "records": records,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def replay_one(env: Any, source_path: Path, output_dir: Path) -> dict[str, Any]:
    source = Demo.from_safetensors(source_path)
    source_uuid = source_path.stem
    output_name = f"{source_uuid}_seed{int(source.seed)}_privileged_replay.safetensors"
    output_path = output_dir / output_name
    started = time.monotonic()

    if output_path.is_file():
        regenerated = Demo.from_safetensors(output_path)
        return {
            "source_demo_uuid": source_uuid,
            "seed": int(source.seed),
            "status": "skipped",
            "reason": "validated output already exists",
            "output": str(output_path),
            "saved_timesteps": len(regenerated.timesteps),
        }

    env.reset(seed=int(source.seed))
    recorder = DemoRecorder(output_dir)
    recorder.record(env, lightweight_demo=True)
    last_stage = None
    final_status: dict[str, Any] = dishwasher_unload_cutlery_status(env)
    success_step = None
    end_reason = "expert actions exhausted"

    try:
        for step_index, source_step in enumerate(source.timesteps, start=1):
            action = np.asarray(source_step.executed_action, dtype=np.float32)
            timestep = env.step(action)
            recorder.add_timestep(timestep, action)
            final_status = dishwasher_unload_cutlery_status(env)
            stage = final_status["stage"]
            if stage != last_stage:
                LOG.info(
                    "demo=%s seed=%s step=%s/%s stage=%s",
                    source_uuid,
                    source.seed,
                    step_index,
                    len(source.timesteps),
                    stage,
                )
                last_stage = stage
            if final_status["success"] or float(timestep[1]) > 0:
                success_step = step_index
                end_reason = "task success"
                break
            if bool(timestep[2]):
                end_reason = "terminated without success"
                break
            if bool(timestep[3]):
                end_reason = "truncated without success"
                break
    finally:
        recorder.stop()

    if success_step is None:
        return {
            "source_demo_uuid": source_uuid,
            "seed": int(source.seed),
            "status": "failed",
            "reason": end_reason,
            "source_timesteps": len(source.timesteps),
            "final_privileged_state": final_status,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

    saved_path = recorder.save_demo(output_name)
    if saved_path is None:
        raise RuntimeError("DemoRecorder returned no output path")
    regenerated = Demo.from_safetensors(saved_path)
    if int(regenerated.seed) != int(source.seed):
        raise RuntimeError("Saved demo seed does not match its source seed")
    if len(regenerated.timesteps) != success_step:
        raise RuntimeError("Saved demo timestep count failed validation")

    return {
        "source_demo_uuid": source_uuid,
        "seed": int(source.seed),
        "status": "success",
        "source_timesteps": len(source.timesteps),
        "saved_timesteps": len(regenerated.timesteps),
        "success_step": success_step,
        "output": str(saved_path),
        "collection_source": "privileged_expert_replay",
        "final_privileged_state": final_status,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def main() -> None:
    args = parse_args()
    source_dir = source_directory(args.demo_root)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Official demo directory does not exist: {source_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    summary_path = args.output_dir / "run_summary.json"

    paths = sorted(source_dir.glob("*.safetensors"))
    if args.demo_uuid:
        selected = set(args.demo_uuid)
        paths = [path for path in paths if path.stem in selected]
    if args.max_demos is not None:
        paths = paths[: max(0, args.max_demos)]
    if not paths:
        raise ValueError("No source demonstrations matched the selection")

    all_records = load_records(manifest_path)
    completed = completed_demo_uuids(all_records)
    current_records: list[dict[str, Any]] = []
    env = build_env(SUPPORTED_TASK, control_frequency=20, with_cameras=True)
    try:
        for index, path in enumerate(paths, start=1):
            if path.stem in completed:
                LOG.info("[%s/%s] skip completed demo=%s", index, len(paths), path.stem)
                continue
            LOG.info("[%s/%s] replay demo=%s", index, len(paths), path.stem)
            try:
                record = replay_one(env, path, args.output_dir)
            except Exception as exc:
                LOG.exception("demo=%s failed with an exception", path.stem)
                record = {
                    "source_demo_uuid": path.stem,
                    "status": "error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            record["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
            append_jsonl(manifest_path, record)
            current_records.append(record)
            all_records.append(record)
            write_summary(summary_path, all_records)
            LOG.info(
                "[%s/%s] result=%s success_step=%s",
                index,
                len(paths),
                record["status"],
                record.get("success_step"),
            )
    finally:
        env.close()

    write_summary(summary_path, all_records)
    successes = sum(record["status"] == "success" for record in current_records)
    LOG.info("finished: new_successes=%s processed=%s", successes, len(current_records))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    main()
