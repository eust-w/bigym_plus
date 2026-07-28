from __future__ import annotations

import mujoco
import numpy as np

from bigym.utils.visual_shell import (
    GaussianVisualShell,
    metric_ground_background,
    neutralize_blue_native_floor,
    replace_native_floor_leakage,
    replace_native_floor_with_metric_ground,
)


def test_metric_ground_is_aligned_and_replaces_default_blue_floor() -> None:
    native = np.zeros((120, 160, 3), dtype=np.uint8)
    native[..., 2] = 200
    rendered = metric_ground_background(
        native,
        np.asarray([0.0, 0.0, 1.5]),
        np.eye(3),
        58.0,
        {
            "z_m": 0.0,
            "tile_size_m": 0.6,
            "grout_width_m": 0.012,
            "base_rgb": [178, 177, 171],
            "grout_rgb": [138, 138, 134],
            "tile_variation": 3.0,
        },
    )
    assert rendered.shape == native.shape
    assert rendered.dtype == np.uint8
    assert not np.array_equal(rendered, native)
    assert float(rendered[..., 2].mean()) < 190.0
    assert np.unique(rendered.reshape(-1, 3), axis=0).shape[0] >= 3


def test_hybrid_ground_replaces_only_weak_native_floor_leakage() -> None:
    native = np.full((2, 3, 3), [40, 80, 180], dtype=np.uint8)
    metric = np.full((2, 3, 3), [168, 166, 160], dtype=np.uint8)
    rendered = native.copy()
    rendered[0, 1] = [42, 82, 176]
    rendered[0, 2] = [180, 80, 30]
    floor = np.asarray([[True, True, True], [False, False, False]])
    cleaned, replacement = replace_native_floor_leakage(
        rendered,
        native,
        metric,
        floor,
        30.0,
    )
    assert replacement.tolist() == [
        [True, True, False],
        [False, False, False],
    ]
    assert np.array_equal(cleaned[0, 0], metric[0, 0])
    assert np.array_equal(cleaned[0, 2], rendered[0, 2])


def test_metric_ground_preserves_closer_gaussian_surface() -> None:
    rendered = np.full((2, 3, 3), 220, dtype=np.uint8)
    metric = np.full((2, 3, 3), [72, 82, 90], dtype=np.uint8)
    floor = np.asarray(
        [[False, True, False], [True, False, True]],
        dtype=bool,
    )
    gaussian_depth = np.asarray(
        [[1.0, 3.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    metric_depth = np.full((2, 3), 2.0, dtype=np.float64)
    replaced = replace_native_floor_with_metric_ground(
        rendered,
        metric,
        floor,
        gaussian_depth,
        metric_depth,
    )
    expected_ground = np.asarray(
        [[False, True, False], [True, False, False]],
        dtype=bool,
    )
    assert np.array_equal(replaced[expected_ground], metric[expected_ground])
    assert np.array_equal(
        replaced[floor & ~expected_ground],
        rendered[floor & ~expected_ground],
    )
    assert np.array_equal(replaced[~floor], rendered[~floor])


def test_native_floor_blue_is_neutralized_without_touching_wall_colors() -> None:
    rendered = np.asarray(
        [[[40, 80, 180], [180, 60, 30], [245, 245, 245]]],
        dtype=np.uint8,
    )
    floor = np.asarray([[True, True, False]])
    neutralized = neutralize_blue_native_floor(rendered, floor)
    assert int(neutralized[0, 0, 2]) - int(neutralized[0, 0, 0]) < 5
    assert np.array_equal(neutralized[0, 1], rendered[0, 1])
    assert np.array_equal(neutralized[0, 2], rendered[0, 2])


def test_segmentation_keeps_native_floor_behind_foreground() -> None:
    shell = object.__new__(GaussianVisualShell)
    shell.native_background_geom_ids = {7}
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    segmentation = np.asarray(
        [
            [[7, geom_type], [8, geom_type], [-1, -1]],
            [[9, geom_type], [7, geom_type], [3, 0]],
        ],
        dtype=np.int32,
    )
    assert shell.native_floor_mask(segmentation).tolist() == [
        [True, False, False],
        [False, True, False],
    ]
    assert shell.native_foreground_mask(segmentation).tolist() == [
        [False, True, False],
        [True, False, False],
    ]
