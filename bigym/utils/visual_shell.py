"""Optional native-3DGS visual background for BiGym observations.

MuJoCo remains the only physics engine.  The Gaussian files are never attached
to a MuJoCo body and therefore cannot add bodies, geoms, contacts, or collision
behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


def metric_ground_projection(
    native_rgb: np.ndarray,
    camera_position_m: np.ndarray,
    camera_rotation_mujoco: np.ndarray,
    fovy_degrees: float,
    config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Render a metric floor and return its camera-Z intersection depth.

    The plane is visual only. Its world-space tile coordinates are derived
    from MuJoCo camera rays, so the floor remains fixed at ``z_m`` as the head
    and wrist cameras move.
    """

    result = np.asarray(native_rgb, dtype=np.uint8).copy()
    if result.ndim != 3 or result.shape[2] != 3:
        raise RuntimeError(f"invalid native RGB for metric ground: {result.shape}")
    position = np.asarray(camera_position_m, dtype=np.float64)
    rotation_mujoco = np.asarray(camera_rotation_mujoco, dtype=np.float64)
    if (
        position.shape != (3,)
        or rotation_mujoco.shape != (3, 3)
        or not np.isfinite(position).all()
        or not np.isfinite(rotation_mujoco).all()
        or not np.isfinite(fovy_degrees)
        or not (1.0 < fovy_degrees < 179.0)
    ):
        raise RuntimeError("invalid camera pose for metric ground")

    height, width = result.shape[:2]
    focal = 0.5 * height / np.tan(np.radians(fovy_degrees) / 2.0)
    pixel_x = (np.arange(width, dtype=np.float64) + 0.5 - width / 2.0) / focal
    pixel_y = (np.arange(height, dtype=np.float64) + 0.5 - height / 2.0) / focal
    x_grid, y_grid = np.meshgrid(pixel_x, pixel_y)
    camera_rays = np.stack(
        [x_grid, y_grid, np.ones_like(x_grid)],
        axis=-1,
    )
    conversion = np.diag([1.0, -1.0, -1.0])
    rotation_cv = rotation_mujoco @ conversion
    world_rays = camera_rays @ rotation_cv.T
    floor_z = float(config.get("z_m", 0.0))
    denominator = world_rays[..., 2]
    distance = np.full((height, width), np.nan, dtype=np.float64)
    downward = denominator < -1e-8
    distance[downward] = (
        floor_z - position[2]
    ) / denominator[downward]
    valid = downward & np.isfinite(distance) & (distance > 0.0)
    if not valid.any():
        return result, distance

    world_x = position[0] + distance * world_rays[..., 0]
    world_y = position[1] + distance * world_rays[..., 1]
    tile_size = float(config.get("tile_size_m", 0.60))
    grout_width = float(config.get("grout_width_m", 0.012))
    if not (
        np.isfinite(tile_size)
        and np.isfinite(grout_width)
        and tile_size > 0.0
        and 0.0 < grout_width < tile_size / 4.0
    ):
        raise RuntimeError("invalid metric-ground tile dimensions")
    base = np.asarray(
        config.get("base_rgb", [178, 177, 171]),
        dtype=np.float64,
    )
    grout = np.asarray(
        config.get("grout_rgb", [138, 138, 134]),
        dtype=np.float64,
    )
    if base.shape != (3,) or grout.shape != (3,):
        raise RuntimeError("metric-ground colors must contain three channels")

    safe_world_x = np.where(valid, world_x, 0.0)
    safe_world_y = np.where(valid, world_y, 0.0)
    tile_x = np.floor(safe_world_x / tile_size).astype(np.int64)
    tile_y = np.floor(safe_world_y / tile_size).astype(np.int64)
    checker = ((tile_x + tile_y) & 1).astype(np.float64)
    variation = (checker * 2.0 - 1.0) * float(
        config.get("tile_variation", 3.0)
    )
    color = np.broadcast_to(base, (height, width, 3)).copy()
    color += variation[..., None]
    local_x = np.mod(safe_world_x, tile_size)
    local_y = np.mod(safe_world_y, tile_size)
    on_grout = (
        (local_x < grout_width)
        | (local_x > tile_size - grout_width)
        | (local_y < grout_width)
        | (local_y > tile_size - grout_width)
    )
    color[on_grout] = grout
    result[valid] = np.clip(color[valid], 0.0, 255.0).astype(np.uint8)
    return result, distance


def metric_ground_background(
    native_rgb: np.ndarray,
    camera_position_m: np.ndarray,
    camera_rotation_mujoco: np.ndarray,
    fovy_degrees: float,
    config: dict[str, object],
) -> np.ndarray:
    """Return the visual-only metric floor background image."""

    rendered, _depth = metric_ground_projection(
        native_rgb,
        camera_position_m,
        camera_rotation_mujoco,
        fovy_degrees,
        config,
    )
    return rendered


def replace_native_floor_leakage(
    rendered_rgb: np.ndarray,
    native_rgb: np.ndarray,
    metric_ground_rgb: np.ndarray,
    native_floor_mask: np.ndarray,
    threshold_rgb: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace only weak Gaussian coverage over the native MuJoCo floor."""

    rendered = np.asarray(rendered_rgb, dtype=np.uint8)
    native = np.asarray(native_rgb, dtype=np.uint8)
    metric = np.asarray(metric_ground_rgb, dtype=np.uint8)
    floor_mask = np.asarray(native_floor_mask, dtype=bool)
    if (
        rendered.shape != native.shape
        or rendered.shape != metric.shape
        or rendered.ndim != 3
        or rendered.shape[2] != 3
        or floor_mask.shape != rendered.shape[:2]
    ):
        raise RuntimeError("invalid arrays for native-floor leakage replacement")
    if not np.isfinite(threshold_rgb) or threshold_rgb < 0.0:
        raise RuntimeError("native-floor leakage threshold must be non-negative")
    influence = np.mean(
        np.abs(rendered.astype(np.int16) - native.astype(np.int16)),
        axis=2,
    )
    replacement = floor_mask & (influence <= threshold_rgb)
    result = rendered.copy()
    result[replacement] = metric[replacement]
    return result, replacement


def replace_native_floor_with_metric_ground(
    rendered_rgb: np.ndarray,
    metric_ground_rgb: np.ndarray,
    native_floor_mask: np.ndarray,
    gaussian_depth_m: np.ndarray,
    metric_ground_depth_m: np.ndarray,
    depth_tolerance_m: float = 0.02,
) -> np.ndarray:
    """Insert metric ground unless a closer Gaussian surface occludes it."""

    rendered = np.asarray(rendered_rgb, dtype=np.uint8)
    metric = np.asarray(metric_ground_rgb, dtype=np.uint8)
    floor_mask = np.asarray(native_floor_mask, dtype=bool)
    gaussian_depth = np.asarray(gaussian_depth_m, dtype=np.float64)
    metric_depth = np.asarray(metric_ground_depth_m, dtype=np.float64)
    if (
        rendered.shape != metric.shape
        or rendered.ndim != 3
        or rendered.shape[2] != 3
        or floor_mask.shape != rendered.shape[:2]
        or gaussian_depth.shape != rendered.shape[:2]
        or metric_depth.shape != rendered.shape[:2]
    ):
        raise RuntimeError("invalid arrays for metric-ground floor replacement")
    if not np.isfinite(depth_tolerance_m) or depth_tolerance_m < 0.0:
        raise RuntimeError("invalid metric-ground depth tolerance")
    valid_metric = np.isfinite(metric_depth) & (metric_depth > 0.0)
    valid_gaussian = np.isfinite(gaussian_depth) & (gaussian_depth > 0.0)
    ground_visible = (
        floor_mask
        & valid_metric
        & (
            ~valid_gaussian
            | (gaussian_depth >= metric_depth - depth_tolerance_m)
        )
    )
    result = rendered.copy()
    result[ground_visible] = metric[ground_visible]
    return result


def neutralize_blue_native_floor(
    rendered_rgb: np.ndarray,
    native_floor_mask: np.ndarray,
    low_blue_dominance: float = 6.0,
    high_blue_dominance: float = 35.0,
) -> np.ndarray:
    """Turn the native blue checker floor into a neutral metric surface."""

    rendered = np.asarray(rendered_rgb, dtype=np.uint8)
    floor_mask = np.asarray(native_floor_mask, dtype=bool)
    if (
        rendered.ndim != 3
        or rendered.shape[2] != 3
        or floor_mask.shape != rendered.shape[:2]
    ):
        raise RuntimeError("invalid arrays for native-floor neutralization")
    if not (
        np.isfinite(low_blue_dominance)
        and np.isfinite(high_blue_dominance)
        and 0.0 <= low_blue_dominance < high_blue_dominance
    ):
        raise RuntimeError("invalid blue-dominance neutralization thresholds")
    source = rendered.astype(np.float64)
    blue_dominance = source[..., 2] - np.maximum(
        source[..., 0],
        source[..., 1],
    )
    weight = np.clip(
        (blue_dominance - low_blue_dominance)
        / (high_blue_dominance - low_blue_dominance),
        0.0,
        1.0,
    )
    weight = weight * weight * (3.0 - 2.0 * weight)
    weight *= floor_mask
    luma = (
        0.2126 * source[..., 0]
        + 0.7152 * source[..., 1]
        + 0.0722 * source[..., 2]
    )
    neutral = np.stack([1.02 * luma, luma, 0.98 * luma], axis=2)
    result = source * (1.0 - weight[..., None]) + neutral * weight[..., None]
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


class GaussianVisualShell:
    """Render a static Gaussian shell behind native MuJoCo geometry."""

    def __init__(
        self,
        profile_path: str | Path,
        model: mujoco.MjModel,
        *,
        strict: bool = True,
    ) -> None:
        self.profile_path = Path(profile_path).expanduser().resolve()
        self.strict = strict
        self.profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        asset_dir = self.profile_path.parent
        alignment_name = self.profile.get("alignment", {}).get("path", "alignment.json")
        alignment = json.loads(
            (asset_dir / alignment_name).read_text(encoding="utf-8")
        )
        self.mujoco_to_gaussian = np.asarray(
            alignment["mujoco_to_gaussian"], dtype=np.float64
        )
        if self.mujoco_to_gaussian.shape != (4, 4):
            raise RuntimeError("mujoco_to_gaussian must be a 4x4 matrix")

        layer_files = self.profile.get("layer_files", {})
        resolved = {
            "background": asset_dir
            / layer_files.get("walls_decor", "walls_decor.ply"),
            "ceiling": asset_dir / layer_files.get("ceiling", "ceiling.ply"),
        }
        self.ground_visual = self.profile.get("ground_visual", {})
        self.ground_mode = str(self.ground_visual.get("mode", "gaussian"))
        if self.ground_mode in {"gaussian", "gaussian_with_metric_fallback"}:
            resolved["floor"] = (
                asset_dir / layer_files.get("floor", "floor.ply")
            )
        elif self.ground_mode != "metric_procedural":
            raise RuntimeError(
                f"unsupported visual-shell ground mode: {self.ground_mode}"
            )
        missing = [str(path) for path in resolved.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Gaussian visual-shell layers are missing: {missing}")

        import torch
        from gaussian_renderer.gs_renderer_mujoco import GSRendererMuJoCo

        if not torch.cuda.is_available():
            raise RuntimeError("Gaussian visual shell requires CUDA")
        self.torch = torch
        self.renderer: Any = GSRendererMuJoCo(
            {
                f"__visual_shell_{name}__": str(path)
                for name, path in resolved.items()
            },
            model,
        )
        if len(self.renderer.gs_body_ids) != 0:
            raise RuntimeError("static Gaussian shell unexpectedly mapped to a body")
        self.native_background_geom_names = tuple(
            str(name)
            for name in self.profile.get("integration_contract", {}).get(
                "native_background_geom_names",
                ["floor"],
            )
        )
        self.native_background_geom_ids = {
            int(geom_id)
            for name in self.native_background_geom_names
            if (
                geom_id := mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    name,
                )
            )
            >= 0
        }
        self.rendered_frames = 0
        self.last_error: str | None = None

    def native_foreground_mask(
        self,
        segmentation: np.ndarray,
    ) -> np.ndarray:
        """Return pixels belonging to MuJoCo foreground geoms.

        MuJoCo segmentation stores ``(object id, object type)``.  Native floor
        geoms are intentionally treated as background so that the Gaussian
        room floor remains visible, while the robot, table, props, and other
        task geometry are composited in front of the Gaussian renderer.
        """

        segmentation = np.asarray(segmentation)
        if segmentation.ndim != 3 or segmentation.shape[-1] != 2:
            raise RuntimeError(
                f"invalid MuJoCo segmentation image: {segmentation.shape}"
            )
        object_id = segmentation[..., 0]
        object_type = segmentation[..., 1]
        foreground = (
            (object_id >= 0)
            & (object_type == int(mujoco.mjtObj.mjOBJ_GEOM))
        )
        for geom_id in self.native_background_geom_ids:
            foreground &= object_id != geom_id
        return foreground

    def native_floor_mask(self, segmentation: np.ndarray) -> np.ndarray:
        """Return pixels whose nearest native MuJoCo geom is a floor."""

        segmentation = np.asarray(segmentation)
        if segmentation.ndim != 3 or segmentation.shape[-1] != 2:
            raise RuntimeError(
                f"invalid MuJoCo segmentation image: {segmentation.shape}"
            )
        object_id = segmentation[..., 0]
        object_type = segmentation[..., 1]
        floor = np.zeros(object_id.shape, dtype=bool)
        for geom_id in self.native_background_geom_ids:
            floor |= (
                (object_id == geom_id)
                & (object_type == int(mujoco.mjtObj.mjOBJ_GEOM))
            )
        return floor

    def _camera_in_gaussian_frame(
        self,
        data: mujoco.MjData,
        camera_id: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        position_metric = np.asarray(data.cam_xpos[camera_id], dtype=np.float64)
        rotation_mujoco_metric = np.asarray(
            data.cam_xmat[camera_id], dtype=np.float64
        ).reshape(3, 3)
        conversion = np.diag([1.0, -1.0, -1.0])
        rotation_cv_metric = rotation_mujoco_metric @ conversion
        rigid_inverse = self.mujoco_to_gaussian[:3, :3]
        inverse_scale = float(np.mean(np.linalg.norm(rigid_inverse, axis=1)))
        if not np.isfinite(inverse_scale) or inverse_scale <= 0.0:
            raise RuntimeError("invalid Sim(3) inverse scale")
        inverse_rotation = rigid_inverse / inverse_scale
        position_gaussian = (
            rigid_inverse @ position_metric + self.mujoco_to_gaussian[:3, 3]
        )
        rotation_cv_gaussian = inverse_rotation @ rotation_cv_metric
        rotation_mujoco_gaussian = rotation_cv_gaussian @ conversion
        return (
            position_gaussian,
            rotation_mujoco_gaussian.reshape(9),
            inverse_scale,
        )

    def render_rgb(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_id: int,
        native_rgb: np.ndarray,
        native_foreground_mask: np.ndarray | None = None,
        native_floor_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Composite MuJoCo task geometry over the Gaussian room shell."""

        try:
            height, width = native_rgb.shape[:2]
            if native_foreground_mask is not None:
                native_foreground_mask = np.asarray(
                    native_foreground_mask,
                    dtype=bool,
                )
                if native_foreground_mask.shape != (height, width):
                    raise RuntimeError(
                        "native foreground mask has invalid shape: "
                        f"{native_foreground_mask.shape}"
                    )
            if native_floor_mask is not None:
                native_floor_mask = np.asarray(native_floor_mask, dtype=bool)
                if native_floor_mask.shape != (height, width):
                    raise RuntimeError(
                        "native floor mask has invalid shape: "
                        f"{native_floor_mask.shape}"
                    )
            position, xmat, gaussian_units_per_meter = (
                self._camera_in_gaussian_frame(data, camera_id)
            )
            background_rgb = native_rgb
            metric_ground_rgb: np.ndarray | None = None
            metric_ground_depth_m: np.ndarray | None = None
            if self.ground_mode in {
                "metric_procedural",
                "gaussian_with_metric_fallback",
            }:
                metric_ground_rgb, metric_ground_depth_m = (
                    metric_ground_projection(
                        native_rgb,
                        np.asarray(
                            data.cam_xpos[camera_id],
                            dtype=np.float64,
                        ),
                        np.asarray(
                            data.cam_xmat[camera_id],
                            dtype=np.float64,
                        ).reshape(3, 3),
                        float(model.cam_fovy[camera_id]),
                        self.ground_visual,
                    )
                )
            if self.ground_mode == "metric_procedural":
                assert metric_ground_rgb is not None
                background_rgb = metric_ground_rgb
            elif self.ground_mode == "gaussian_with_metric_fallback":
                hybrid_compositing = self.ground_visual.get(
                    "hybrid_compositing",
                    "native_floor_blue_neutralization",
                )
                if hybrid_compositing in {
                    "native_floor_leakage_replacement",
                    "native_floor_blue_neutralization",
                }:
                    background_rgb = native_rgb
                else:
                    assert metric_ground_rgb is not None
                    background_rgb = metric_ground_rgb
            background = self.torch.as_tensor(
                background_rgb,
                dtype=self.torch.float32,
                device=self.renderer.gaussians.device,
            ).div_(255.0)[None]
            with self.torch.no_grad():
                rgb_tensor, depth_tensor = self.renderer.render_batch(
                    position[None],
                    xmat[None],
                    height,
                    width,
                    np.asarray([model.cam_fovy[camera_id]], dtype=np.float64),
                    bg_imgs=background,
                )
            result = (
                rgb_tensor[0]
                .clamp(0.0, 1.0)
                .mul(255.0)
                .byte()
                .cpu()
                .numpy()
            )
            if result.shape != native_rgb.shape or result.dtype != np.uint8:
                raise RuntimeError(
                    f"invalid composed RGB: {result.shape} {result.dtype}"
                )
            if self.ground_mode == "gaussian_with_metric_fallback":
                hybrid_compositing = self.ground_visual.get(
                    "hybrid_compositing",
                    "native_floor_blue_neutralization",
                )
                if hybrid_compositing == "native_floor_leakage_replacement":
                    if native_floor_mask is None or metric_ground_rgb is None:
                        raise RuntimeError(
                            "hybrid leakage replacement requires a native "
                            "floor mask"
                        )
                    result, _replacement = replace_native_floor_leakage(
                        result,
                        native_rgb,
                        metric_ground_rgb,
                        native_floor_mask,
                        float(
                            self.ground_visual.get(
                                "native_floor_leakage_threshold_rgb",
                                30.0,
                            )
                        ),
                    )
                elif hybrid_compositing == "native_floor_blue_neutralization":
                    if native_floor_mask is None:
                        raise RuntimeError(
                            "hybrid floor neutralization requires a native "
                            "floor mask"
                        )
                    result = neutralize_blue_native_floor(
                        result,
                        native_floor_mask,
                        float(
                            self.ground_visual.get(
                                "low_blue_dominance",
                                6.0,
                            )
                        ),
                        float(
                            self.ground_visual.get(
                                "high_blue_dominance",
                                35.0,
                            )
                        ),
                    )
            elif self.ground_mode == "metric_procedural":
                if (
                    native_floor_mask is None
                    or metric_ground_rgb is None
                    or metric_ground_depth_m is None
                ):
                    raise RuntimeError(
                        "metric procedural ground requires a native floor mask"
                    )
                gaussian_depth_m = (
                    depth_tensor[0, ..., 0]
                    .div(gaussian_units_per_meter)
                    .cpu()
                    .numpy()
                )
                result = replace_native_floor_with_metric_ground(
                    result,
                    metric_ground_rgb,
                    native_floor_mask,
                    gaussian_depth_m,
                    metric_ground_depth_m,
                    float(
                        self.ground_visual.get(
                            "depth_tolerance_m",
                            0.02,
                        )
                    ),
                )
            if native_foreground_mask is not None:
                result[native_foreground_mask] = native_rgb[
                    native_foreground_mask
                ]
            self.rendered_frames += 1
            self.last_error = None
            return result
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            if self.strict:
                raise RuntimeError(
                    f"Gaussian visual-shell render failed: {self.last_error}"
                ) from exc
            return native_rgb

    def status(self) -> dict[str, object]:
        return {
            "enabled": True,
            "profile": str(self.profile_path),
            "strict": self.strict,
            "rendered_frames": self.rendered_frames,
            "last_error": self.last_error,
            "background_physics": {
                "body_count": 0,
                "geom_count": 0,
                "collision_count": 0,
            },
            "foreground_occlusion": {
                "method": "MuJoCo segmentation foreground overwrite",
                "native_background_geom_names": list(
                    self.native_background_geom_names
                ),
            },
            "ground_visual": {
                "mode": self.ground_mode,
                "z_m": float(self.ground_visual.get("z_m", 0.0)),
                "gaussian_floor_loaded": self.ground_mode
                in {"gaussian", "gaussian_with_metric_fallback"},
                "collision_geometry_added": False,
            },
        }
