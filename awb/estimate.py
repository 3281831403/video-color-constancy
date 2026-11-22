"""Auto white balance kernels for linear-light RGB frames.

Frames are non-empty sequences of ``(r, g, b)`` triples whose components are
linear-light values in ``[0.0, 1.0]``.  No sRGB transfer function is applied
inside this module: correction gains operate directly on linear RGB.

The reference illuminant is 6200 K.  :func:`kelvin_curve` returns the
green-normalised relative RGB weights of a Planckian white at a given CCT,
using the log-linear fit::

    u = log(K / 6200)
    curve(K) = (exp(-0.6 * u), 1.0, exp(0.6 * u))

Low Kelvin is therefore warm (R > B), high Kelvin is cool (B > R), the green
weight is always exactly 1, and the product of the three weights is exactly 1.
:func:`estimate_kelvin` inverts the R/B ratio of that model::

    K = 6200 * (B / R) ** (5 / 6)

:func:`scene_correction` uses the reciprocal curve as correction gains, so a
warm scene receives a blue-heavy correction while green stays anchored at 1.

Unless a docstring says otherwise, numerical tolerances are absolute ``1e-9``.
Only the Python standard library is used; execution is deterministic and
offline.
"""

from __future__ import annotations

import math

__all__ = [
    "apply_gains",
    "awb_batch",
    "estimate_kelvin",
    "gray_world_gains",
    "kelvin_curve",
    "mean_rgb",
    "scene_correction",
    "validate",
]

_KELVIN_MIN = 2000.0
_KELVIN_MAX = 10000.0
_REFERENCE_KELVIN = 6200.0
_CURVE_SLOPE = 0.6
_GAIN_FLOOR_DEFAULT = 0.25
_GAIN_CEILING_DEFAULT = 4.0


def _is_real(value) -> bool:
    """Return True for finite real numbers (booleans excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_rgb_triple(pixel) -> bool:
    return (
        isinstance(pixel, (tuple, list))
        and len(pixel) == 3
        and all(_is_real(channel) for channel in pixel)
    )


def mean_rgb(img: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Return the per-channel mean of the usable pixels in ``img``.

    A pixel is usable only when **all three** components are finite and
    strictly inside ``(0.0, 1.0)``.  A pixel with any component equal to 0.0
    is dead black, and a pixel with any component equal to 1.0 is clipped or
    overexposed; both are excluded from every channel's average.  This keeps a
    small cluster of clipped highlights from pulling the estimated white point
    toward white.

    The result is a three-element tuple ``(r_mean, g_mean, b_mean)``.  If no
    usable pixels remain, raise :class:`ValueError`.  The input is not mutated.
    Averages agree with the reference to an absolute tolerance of ``1e-9``.
    """
    r_sum = g_sum = b_sum = 0.0
    count = 0
    for pixel in img:
        if not _valid_rgb_triple(pixel):
            raise ValueError("each pixel must be a finite (r, g, b) triple")
        r, g, b = pixel
        # Dead black (any 0.0) and clipped highlights (any 1.0) are excluded,
        # along with anything outside the unit interval.
        if 0.0 < r < 1.0 and 0.0 < g < 1.0 and 0.0 < b < 1.0:
            r_sum += r
            g_sum += g
            b_sum += b
            count += 1
    if count == 0:
        raise ValueError("no usable pixels remain after clipping exclusion")
    return r_sum / count, g_sum / count, b_sum / count


def gray_world_gains(mean: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return green-anchored grey-world correction gains for channel means.

    ``mean`` contains positive, finite channel averages.  The returned gains
    are exactly::

        (mean_g / mean_r, 1.0, mean_g / mean_b)

    Green is the anchor and is returned as exactly ``1.0``; there is no
    secondary normalisation by the arithmetic or geometric mean of the gains.
    Applying these gains to the channel means equalises all three means at
    ``mean_g`` while preserving green.  Non-positive or non-finite means raise
    :class:`ValueError`.
    """
    if not _valid_rgb_triple(mean):
        raise ValueError("mean must be a finite (r, g, b) triple")
    mean_r, mean_g, mean_b = mean
    if mean_r <= 0.0 or mean_g <= 0.0 or mean_b <= 0.0:
        raise ValueError("channel means must be strictly positive")
    return mean_g / mean_r, 1.0, mean_g / mean_b


def kelvin_curve(kelvin: float) -> tuple[float, float, float]:
    """Map a colour temperature to green-normalised Planckian RGB weights.

    ``kelvin`` is a finite CCT in ``[2000.0, 10000.0]`` K.  The log-linear fit
    documented in the module docstring is used::

        u = log(kelvin / 6200.0)
        (exp(-0.6 * u), 1.0, exp(0.6 * u))

    Consequently 6200 K maps to ``(1.0, 1.0, 1.0)``, lower temperatures are
    warm (red weight greater than blue), higher temperatures are cool (blue
    weight greater than red), and the curve is continuous for every CCT in
    the supported interval, including interpolation knots.  The green weight
    is exactly ``1.0`` and the product of all three weights is ``1.0``.
    Values outside the supported interval or non-finite values raise
    :class:`ValueError`.
    """
    if not _is_real(kelvin) or not (_KELVIN_MIN <= kelvin <= _KELVIN_MAX):
        raise ValueError("kelvin must be a finite CCT in [2000.0, 10000.0] K")
    # A single smooth log-linear formula over the whole interval: there are no
    # piecewise segments, so the curve cannot jump at interpolation knots.
    u = math.log(kelvin / _REFERENCE_KELVIN)
    return math.exp(-_CURVE_SLOPE * u), 1.0, math.exp(_CURVE_SLOPE * u)


def estimate_kelvin(mean: tuple[float, float, float]) -> float:
    """Estimate the scene CCT from the R/B ratio of channel means.

    The estimate is the inverse of :func:`kelvin_curve`'s red/blue ratio::

        kelvin = 6200.0 * (mean_b / mean_r) ** (5.0 / 6.0)

    The logarithm of the ratio is used, so equal *multiplicative* changes in
    R/B produce equal changes in log temperature rather than a compressed
    linear CCT range.  Equal red and blue means return exactly ``6200.0``.
    The green mean does not affect the estimate.  Non-positive or non-finite
    means raise :class:`ValueError`.
    """
    if not _valid_rgb_triple(mean):
        raise ValueError("mean must be a finite (r, g, b) triple")
    mean_r, _mean_g, mean_b = mean
    if mean_r <= 0.0 or mean_b <= 0.0:
        raise ValueError("red and blue means must be strictly positive")
    # Log-space inversion of the log-linear curve: the R/B ratio maps
    # monotonically onto CCT instead of being compressed into a linear range.
    return _REFERENCE_KELVIN * (mean_b / mean_r) ** (5.0 / 6.0)


def apply_gains(
    img: list[tuple[float, float, float]],
    gains: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    """Apply per-channel gains in linear light and clamp to ``[0.0, 1.0]``.

    Each output channel is ``clamp(input[channel] * gains[channel], 0, 1)``.
    The multiplication happens in the linear-light domain represented by the
    input, not in an sRGB-encoded domain, so an unclipped gain change scales
    that channel linearly and never changes the ordering of input samples.
    No gamma decode/encode and no rounding are performed.

    Returns a new list of ``(r, g, b)`` tuples in the input order; the input
    list and its pixels are not mutated.  Malformed pixels, non-finite values
    and non-positive or non-finite gains raise :class:`ValueError`.  Values
    agree with the reference to an absolute tolerance of ``1e-9``.
    """
    if not _valid_rgb_triple(gains):
        raise ValueError("gains must be a finite (g_r, g_g, g_b) triple")
    if any(gain <= 0.0 for gain in gains):
        raise ValueError("gains must be strictly positive")
    g_r, g_g, g_b = gains
    corrected = []
    for pixel in img:
        if not _valid_rgb_triple(pixel):
            raise ValueError("each pixel must be a finite (r, g, b) triple")
        r, g, b = pixel
        corrected.append(
            (
                min(1.0, max(0.0, r * g_r)),
                min(1.0, max(0.0, g * g_g)),
                min(1.0, max(0.0, b * g_b)),
            )
        )
    return corrected


def scene_correction(
    img: list[tuple[float, float, float]],
    config: dict,
) -> dict:
    """Estimate the illuminant and return green-anchored correction gains.

    The order is fixed and is the whole point of the function:

    1. compute the usable channel means from the **original** frame,
    2. estimate Kelvin from those original means,
    3. evaluate :func:`kelvin_curve` at that Kelvin,
    4. invert only the red and blue weights, keeping green at ``1.0``.

    The frame is not re-measured after correction, so the returned ``kelvin``
    is the scene's original CCT rather than a self-fulfilling 6200 K.  The
    returned mapping is::

        {"gains": (g_r, 1.0, g_b), "kelvin": estimated_kelvin}

    ``config`` is a mapping.  The optional keys ``gain_floor`` (default
    ``0.25``) and ``gain_ceiling`` (default ``4.0``) clamp the returned
    correction gains.  Unknown keys are ignored.  Invalid bounds raise
    :class:`ValueError`.  The frame itself is never modified.
    """
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    gain_floor = config.get("gain_floor", _GAIN_FLOOR_DEFAULT)
    gain_ceiling = config.get("gain_ceiling", _GAIN_CEILING_DEFAULT)
    if (
        not _is_real(gain_floor)
        or not _is_real(gain_ceiling)
        or gain_floor <= 0.0
        or gain_ceiling <= 0.0
        or gain_floor > gain_ceiling
    ):
        raise ValueError("invalid gain_floor/gain_ceiling bounds")

    # 1. Statistics on the untouched original frame.
    mean = mean_rgb(img)
    # 2. Kelvin estimated from the original statistics, before any correction.
    kelvin = estimate_kelvin(mean)
    # 3. White-point weights at the estimated CCT.  Clamp the query into the
    # supported interval so extreme scenes still produce a finite curve.
    curve_kelvin = min(_KELVIN_MAX, max(_KELVIN_MIN, kelvin))
    w_r, _w_g, w_b = kelvin_curve(curve_kelvin)
    # 4. Reciprocal weights correct the cast; green stays anchored at exactly
    # 1.0, then the configured safety bounds clamp red and blue.
    g_r = min(gain_ceiling, max(gain_floor, 1.0 / w_r))
    g_b = min(gain_ceiling, max(gain_floor, 1.0 / w_b))
    return {"gains": (g_r, 1.0, g_b), "kelvin": kelvin}


def validate(
    img: list[tuple[float, float, float]],
    gains: tuple[float, float, float],
) -> list[str]:
    """Return stable error codes for an image/gain pair; empty means valid.

    Image checks accept RGB components in the inclusive range ``[0.0, 1.0]``
    (clipping is a valid stored value, although :func:`mean_rgb` excludes it
    from statistics).  Gain checks require three positive finite values in
    ``[0.25, 4.0]``.  In addition, the product ``g_r * g_g * g_b`` must be
    within ``0.01`` of ``1.0`` so an exposure-preserving correction does not
    brighten or darken the whole frame.

    Supported codes, in this order and without duplicates, are:
    ``"image_empty"``, ``"image_channels"``, ``"image_nan"``,
    ``"image_range"``, ``"gains_channels"``, ``"gains_nan"``,
    ``"gains_range"`` and ``"gain_product"``.  The function does not raise for
    invalid content and does not mutate its arguments.
    """
    errors: list[str] = []

    if len(img) == 0:
        errors.append("image_empty")
    elif not all(isinstance(pixel, (tuple, list)) and len(pixel) == 3 for pixel in img):
        errors.append("image_channels")
    elif not all(_is_real(channel) for pixel in img for channel in pixel):
        errors.append("image_nan")
    elif not all(0.0 <= channel <= 1.0 for pixel in img for channel in pixel):
        errors.append("image_range")

    if not isinstance(gains, (tuple, list)) or len(gains) != 3:
        errors.append("gains_channels")
    elif not all(_is_real(gain) for gain in gains):
        errors.append("gains_nan")
    else:
        if not all(0.25 <= gain <= 4.0 for gain in gains):
            errors.append("gains_range")
        # Product is only meaningful once three finite gains exist.
        if abs(gains[0] * gains[1] * gains[2] - 1.0) > 0.01:
            errors.append("gain_product")

    return errors


def awb_batch(
    frames: list[list[tuple[float, float, float]]],
    config: dict,
) -> dict:
    """White-balance every frame independently and preserve input order.

    Each frame is passed to :func:`scene_correction` with the same ``config``,
    but no mean, gain or Kelvin value is cached or shared between frames.  The
    second frame therefore uses only its own statistics.  The return value is::

        {"results": [scene_correction(frame, config), ...],
         "count": len(frames)}

    An empty batch returns ``{"results": [], "count": 0}``.  The input frames
    are not mutated and frame order is preserved.  Errors raised by
    :func:`scene_correction` propagate; no partial result is returned.
    """
    results = [scene_correction(frame, config) for frame in frames]
    return {"results": results, "count": len(results)}
