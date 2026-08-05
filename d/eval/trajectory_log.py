"""Crash-aware structured logs for policy evaluation episodes.

The logger deliberately stores metadata only: camera pixels remain in the
existing rollout videos.  Every reset, policy error and environment transition
is appended to JSONL immediately, while the compact episode manifest is
atomically replaced when an episode reaches a terminal state.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np


SCHEMA_VERSION = 1
TERMINAL_STATUSES = frozenset({"complete", "failed", "interrupted"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Convert common simulator values to strict, loss-aware JSON values."""

    def convert(item: Any, seen: set[int]) -> Any:
        if item is None or isinstance(item, (bool, str)):
            return item
        if isinstance(item, np.generic):
            return convert(item.item(), seen)
        if isinstance(item, int) and not isinstance(item, bool):
            return int(item)
        if isinstance(item, float):
            number = float(item)
            if math.isfinite(number):
                return number
            if math.isnan(number):
                return "NaN"
            return "Infinity" if number > 0 else "-Infinity"
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, (bytes, bytearray, memoryview)):
            return {
                "__type__": "bytes",
                "base64": base64.b64encode(bytes(item)).decode("ascii"),
            }

        identity = id(item)
        if identity in seen:
            return {"__type__": "cycle", "python_type": type(item).__name__}
        if isinstance(item, np.ndarray):
            seen.add(identity)
            try:
                return convert(item.tolist(), seen)
            finally:
                seen.remove(identity)
        if isinstance(item, Mapping):
            seen.add(identity)
            try:
                return {
                    str(convert(key, seen)): convert(child, seen)
                    for key, child in item.items()
                }
            finally:
                seen.remove(identity)
        if isinstance(item, Sequence):
            seen.add(identity)
            try:
                return [convert(child, seen) for child in item]
            finally:
                seen.remove(identity)

        return {
            "__type__": "python_object",
            "python_type": f"{type(item).__module__}.{type(item).__qualname__}",
            "repr": repr(item),
        }

    return convert(value, set())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON object through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoded = (
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EpisodeTrajectoryLog:
    """Append-only transition log plus an atomic per-episode manifest."""

    def __init__(
        self,
        root: str | Path,
        *,
        episode_index: int,
        seed: int,
        task: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.episode_index = int(episode_index)
        self.seed = int(seed)
        self.task = task
        self.directory = Path(root) / f"episode-{self.episode_index:06d}"
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"trajectory directory already exists: {self.directory}; "
                "choose a new --tag to preserve the previous evaluation"
            ) from error
        self.steps_path = self.directory / "steps.jsonl"
        self.manifest_path = self.directory / "manifest.json"
        self.started_at = utc_now()
        self.record_count = 0
        self.closed = False
        self.metadata = dict(metadata or {})
        self._write_manifest(status="recording")

    def _write_manifest(self, *, status: str, **summary: Any) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "task": self.task,
            "episode_index": self.episode_index,
            "seed": self.seed,
            "started_at": self.started_at,
            "updated_at": utc_now(),
            "record_count": self.record_count,
            "metadata": self.metadata,
            **summary,
        }
        atomic_write_json(self.manifest_path, payload)

    def append(self, record_type: str, **payload: Any) -> int:
        if self.closed:
            raise RuntimeError("cannot append to a terminal trajectory log")
        record_index = self.record_count
        record = {
            "schema_version": SCHEMA_VERSION,
            "record_index": record_index,
            "record_type": record_type,
            "episode_index": self.episode_index,
            "seed": self.seed,
            "recorded_at": utc_now(),
            **payload,
        }
        encoded = (
            json.dumps(json_safe(record), separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            self.steps_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("append-only JSONL write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.record_count += 1
        return record_index

    def finish(
        self,
        *,
        status: str,
        success: bool,
        transitions: int,
        failure_reason: str | None,
        **summary: Any,
    ) -> None:
        if self.closed:
            raise RuntimeError("trajectory log is already terminal")
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"invalid terminal status: {status}")
        self.closed = True
        step_bytes = self.steps_path.stat().st_size if self.steps_path.exists() else 0
        step_sha256 = sha256_file(self.steps_path) if self.steps_path.exists() else None
        self._write_manifest(
            status=status,
            finished_at=utc_now(),
            success=bool(success),
            transitions=int(transitions),
            failure_reason=failure_reason,
            steps_jsonl={
                "path": self.steps_path.name,
                "bytes": step_bytes,
                "sha256": step_sha256,
            },
            **summary,
        )
