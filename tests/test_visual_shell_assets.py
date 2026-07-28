from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from bigym.utils.visual_shell_assets import (
    VisualShellAsset,
    install_visual_shell,
    resolve_visual_shell_profile,
    verify_visual_shell_directory,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_bundle(tmp_path: Path) -> tuple[Path, VisualShellAsset]:
    source = tmp_path / "source" / "tiny-shell"
    source.mkdir(parents=True)
    files = {
        "profile.json": b'{"schema_version": 1}\n',
        "alignment.json": b'{"schema_version": 1}\n',
        "walls.ply": b"ply\nend_header\n",
    }
    manifest_entries = []
    for relative, content in files.items():
        path = source / relative
        path.write_bytes(content)
        manifest_entries.append(
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": manifest_entries,
            }
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "tiny-shell.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(source, arcname="tiny-shell")
    asset = VisualShellAsset(
        bundle_url=bundle.as_uri(),
        bundle_sha256=_sha256(bundle),
        bundle_bytes=bundle.stat().st_size,
        bundle_root="tiny-shell",
        profile_path="profile.json",
    )
    return bundle, asset


def test_install_resolve_and_reuse_verified_shell(tmp_path: Path) -> None:
    _bundle, asset = _make_bundle(tmp_path)
    registry = {"tiny": asset}
    cache = tmp_path / "cache"
    profile = install_visual_shell(
        "tiny",
        cache_dir=cache,
        registry=registry,
    )
    assert profile == cache / "tiny-shell" / "profile.json"
    assert verify_visual_shell_directory(profile.parent)["schema_version"] == 1
    assert (
        resolve_visual_shell_profile(
            "tiny",
            cache_dir=cache,
            registry=registry,
        )
        == profile
    )


def test_resolve_existing_profile_does_not_download(tmp_path: Path) -> None:
    profile = tmp_path / "custom.json"
    profile.write_text("{}\n", encoding="utf-8")
    assert resolve_visual_shell_profile(profile) == profile.resolve()


def test_bundle_sha_mismatch_does_not_install(tmp_path: Path) -> None:
    _bundle, asset = _make_bundle(tmp_path)
    invalid = VisualShellAsset(
        bundle_url=asset.bundle_url,
        bundle_sha256="0" * 64,
        bundle_bytes=asset.bundle_bytes,
        bundle_root=asset.bundle_root,
        profile_path=asset.profile_path,
    )
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        install_visual_shell(
            "tiny",
            cache_dir=cache,
            registry={"tiny": invalid},
        )
    assert not (cache / "tiny-shell").exists()


def test_archive_path_traversal_is_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "unsafe.tar.gz"
    payload = b"escape"
    with tarfile.open(bundle, "w:gz") as archive:
        member = tarfile.TarInfo("tiny-shell/../../escape")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    asset = VisualShellAsset(
        bundle_url=bundle.as_uri(),
        bundle_sha256=_sha256(bundle),
        bundle_bytes=bundle.stat().st_size,
        bundle_root="tiny-shell",
        profile_path="profile.json",
    )
    with pytest.raises(RuntimeError, match="unsafe member"):
        install_visual_shell(
            "tiny",
            cache_dir=tmp_path / "cache",
            registry={"tiny": asset},
        )
    assert not (tmp_path / "escape").exists()


def test_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    root = tmp_path / "shell"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {
                        "path": "../escape",
                        "bytes": 0,
                        "sha256": hashlib.sha256(b"").hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="unsafe path"):
        verify_visual_shell_directory(root)
