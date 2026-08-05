"""Tests for the headless pixel-observation benchmark."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


EXAMPLE = Path(__file__).parents[1] / "examples" / "benchmark_pixel_observations.py"
SPEC = importlib.util.spec_from_file_location("benchmark_pixel_observations", EXAMPLE)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_frame_summary_accepts_finite_nonconstant_chw_rgb():
    frame = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4)

    summary = benchmark.summarize_frame(frame, resolution=4)

    assert summary["shape"] == [3, 4, 4]
    assert summary["dtype"] == "uint8"
    assert summary["minimum"] == 0.0
    assert summary["maximum"] == 47.0
    assert summary["standard_deviation"] > 0


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (np.zeros((3, 4, 4), dtype=np.uint8), "blank or constant"),
        (np.zeros((4, 4, 3), dtype=np.uint8), "shape"),
        (np.full((3, 4, 4), np.nan, dtype=np.float32), "non-finite"),
    ],
)
def test_frame_summary_rejects_invalid_render_output(frame, message):
    with pytest.raises(RuntimeError, match=message):
        benchmark.summarize_frame(frame, resolution=4)
