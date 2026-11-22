"""Red regression tests for the auto white balance kernels.

``awb/estimate.py`` is intentionally incomplete.  These tests are the
evaluation baseline and must not be edited; implement the module instead.
"""

import pytest

from awb.estimate import (
    apply_gains,
    awb_batch,
    estimate_kelvin,
    gray_world_gains,
    kelvin_curve,
    mean_rgb,
    scene_correction,
    validate,
)


def _assert_rgb_close(actual, expected, abs_=1e-9):
    assert len(actual) == len(expected) == 3
    for got, want in zip(actual, expected):
        assert got == pytest.approx(want, abs=abs_)


def test_mean_rgb_ignores_clipped_highlights_and_dead_black():
    """Defect 1: clipped white must not pull the white point upward."""
    img = [
        (0.25, 0.25, 0.25),
        (0.50, 0.50, 0.50),
        (0.75, 0.75, 0.75),
        (1.00, 0.40, 0.40),
        (0.40, 1.00, 0.40),
        (0.00, 0.40, 0.40),
        (0.40, 0.00, 0.40),
    ]

    assert mean_rgb(img) == pytest.approx((0.5, 0.5, 0.5), abs=1e-9)


def test_gray_world_gains_keep_green_channel_fixed():
    """Defect 2: scaling green as well makes the whole frame drift."""
    mean = (0.40, 0.50, 0.60)

    gains = gray_world_gains(mean)

    assert gains[1] == pytest.approx(1.0, abs=1e-9)
    assert mean[0] * gains[0] == pytest.approx(mean[1], abs=1e-9)
    assert mean[2] * gains[2] == pytest.approx(mean[1], abs=1e-9)


def test_gray_world_gains_use_green_anchor_not_arithmetic_average():
    """Defect 3: normalising by the average gain breaks the green anchor."""
    gains = gray_world_gains((0.20, 0.40, 0.80))

    assert gains == pytest.approx((2.0, 1.0, 0.5), abs=1e-9)


def test_kelvin_curve_direction_is_warm_at_low_kelvin():
    """Defect 4: a reversed curve makes 3000 K bluish instead of warm."""
    low = kelvin_curve(3000.0)
    high = kelvin_curve(9000.0)

    assert low[1] == pytest.approx(1.0, abs=1e-9)
    assert high[1] == pytest.approx(1.0, abs=1e-9)
    assert low[0] > 1.0 > low[2]
    assert high[2] > 1.0 > high[0]


def test_estimate_kelvin_uses_log_ratio_for_wide_cct_range():
    """Defect 5: a raw R/B ratio compresses the CCT range."""
    warm = estimate_kelvin((0.72, 0.50, 0.40))  # R/B = 1.80
    warmer = estimate_kelvin((0.81, 0.50, 0.25))  # R/B = 3.24

    assert 2000.0 < warm < 6200.0
    assert 2000.0 < warmer < warm
    # In log-CCT space, squaring the R/B ratio must square the CCT ratio
    # around the 6200 K anchor.
    assert warmer == pytest.approx(warm * warm / 6200.0, rel=0.02)


def test_apply_gains_is_linear_and_monotonic():
    """Defect 6: applying gain in sRGB space makes scaling non-linear."""
    img = [
        (0.10, 0.20, 0.30),
        (0.20, 0.40, 0.60),
        (0.30, 0.60, 0.90),
    ]
    gains = (1.5, 1.0, 0.5)

    corrected = apply_gains(img, gains)

    assert isinstance(corrected, list)
    assert len(corrected) == 3
    _assert_rgb_close(corrected[0], (0.15, 0.20, 0.15))
    _assert_rgb_close(corrected[1], (0.30, 0.40, 0.30))
    _assert_rgb_close(corrected[2], (0.45, 0.60, 0.45))
    for channel in range(3):
        assert corrected[0][channel] < corrected[1][channel] < corrected[2][channel]


def test_scene_correction_estimates_before_applying_gains():
    """Defect 7: estimating after correction always reports neutral 6200 K."""
    warm = [(0.70, 0.50, 0.35)] * 4
    expected_kelvin = estimate_kelvin(mean_rgb(warm))

    result = scene_correction(warm, {})

    assert {"gains", "kelvin"} <= set(result)
    assert result["kelvin"] == pytest.approx(expected_kelvin, abs=1e-9)
    assert result["kelvin"] < 5500.0
    assert result["gains"][2] > result["gains"][0]


def test_validate_rejects_non_unity_gain_product():
    """Defect 8: without the product check, total exposure is unconstrained."""
    img = [(0.25, 0.50, 0.75)]

    assert "gain_product" in validate(img, (1.20, 1.00, 1.20))
    assert validate(img, (1.20, 1.00, 1.0 / 1.20)) == []


def test_awb_batch_uses_each_frame_statistics():
    """Defect 9: caching the first mean contaminates later frames."""
    warm = [(0.70, 0.50, 0.35)] * 3
    cool = [(0.40, 0.50, 0.70)] * 3

    batch = awb_batch([warm, cool], {})
    results = batch["results"]

    assert len(results) == 2
    assert results[0]["kelvin"] < 5000.0
    assert results[1]["kelvin"] > 8000.0
    assert results[0]["gains"][2] > results[0]["gains"][0]
    assert results[1]["gains"][0] > results[1]["gains"][2]


def test_kelvin_curve_is_continuous_across_knots():
    """Defect 10: discontinuous piecewise endpoints jump between segments."""
    for knot in (3000.0, 4500.0, 6000.0, 7500.0, 9000.0):
        before = kelvin_curve(knot - 0.5)
        after = kelvin_curve(knot + 0.5)
        for left, right in zip(before, after):
            assert left == pytest.approx(right, abs=5e-3)


def test_known_6200k_neutral_scene_estimates_neutral():
    """Known-CCT restoration: neutral grey at the reference is 6200 K."""
    img = [(0.25, 0.25, 0.25), (0.50, 0.50, 0.50), (0.75, 0.75, 0.75)]

    kelvin = estimate_kelvin(mean_rgb(img))

    assert kelvin == pytest.approx(6200.0, abs=100.0)
    assert kelvin_curve(6200.0) == pytest.approx((1.0, 1.0, 1.0), abs=1e-9)


def test_applied_gains_equalize_channel_means():
    """Gain normalisation: corrected R/G/B means agree to 1e-9."""
    img = [
        (0.60, 0.50, 0.30),
        (0.40, 0.30, 0.70),
        (0.70, 0.60, 0.40),
        (0.50, 0.40, 0.60),
    ]

    gains = gray_world_gains(mean_rgb(img))
    corrected_means = mean_rgb(apply_gains(img, gains))

    assert corrected_means[0] == pytest.approx(corrected_means[1], abs=1e-9)
    assert corrected_means[1] == pytest.approx(corrected_means[2], abs=1e-9)


def test_kelvin_curve_total_gain_product_is_one():
    """Exposure stability: the white-point weights conserve total gain."""
    for kelvin in (2000.0, 3200.0, 4500.0, 6200.0, 8000.0, 10000.0):
        gains = kelvin_curve(kelvin)
        product = gains[0] * gains[1] * gains[2]
        assert gains[0] > 0.0 and gains[1] > 0.0 and gains[2] > 0.0
        assert product == pytest.approx(1.0, abs=1e-2)


def test_estimate_kelvin_is_monotonic_across_scene_casts():
    """Stability: warmer scenes must estimate below cooler scenes."""
    warm = [(0.70, 0.50, 0.50)] * 2
    neutral = [(0.50, 0.50, 0.50)] * 2
    cool = [(0.50, 0.50, 0.70)] * 2

    warm_k = estimate_kelvin(mean_rgb(warm))
    neutral_k = estimate_kelvin(mean_rgb(neutral))
    cool_k = estimate_kelvin(mean_rgb(cool))

    assert warm_k < neutral_k - 200.0
    assert cool_k > neutral_k + 200.0


def test_batch_matches_independent_per_frame_awb():
    """Per-frame independence: batch output equals separate frame runs."""
    frames = [
        [(0.68, 0.50, 0.36)] * 2,
        [(0.38, 0.50, 0.66)] * 3,
    ]

    batch = awb_batch(frames, {})

    assert batch["count"] == 2
    assert len(batch["results"]) == 2
    for index, frame in enumerate(frames):
        single = scene_correction(frame, {})
        batched = batch["results"][index]
        assert batched["kelvin"] == pytest.approx(single["kelvin"], abs=1e-9)
        _assert_rgb_close(batched["gains"], single["gains"])
