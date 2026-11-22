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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
