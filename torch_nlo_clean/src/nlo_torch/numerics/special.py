"""Special functions used by the BK and DIS calculations."""

from __future__ import annotations

import math

import torch


def bessel_J1(x: torch.Tensor) -> torch.Tensor:
    return torch.special.bessel_j1(x)


def bessel_I1(x: torch.Tensor) -> torch.Tensor:
    return torch.special.modified_bessel_i1(x)


def bessel_K0(x: torch.Tensor) -> torch.Tensor:
    return torch.special.modified_bessel_k0(x)


def bessel_K1(x: torch.Tensor) -> torch.Tensor:
    return torch.special.modified_bessel_k1(x)


def bessel_K2(x: torch.Tensor) -> torch.Tensor:
    """Evaluate ``K_2(x) = K_0(x) + 2 K_1(x) / x`` for nonnegative ``x``."""

    zero = x == 0
    safe_x = torch.where(zero, torch.ones_like(x), x)
    value = bessel_K0(x) + 2 * bessel_K1(x) / safe_x
    return torch.where(zero, torch.full_like(value, torch.inf), value)


def Li2(x: torch.Tensor) -> torch.Tensor:
    """Real dilogarithm on the source-required domain ``x <= 1``.

    Reflection and inversion identities reduce the argument to ``[0, 1/2]``.
    The remaining power series converges to floating-point precision in 64
    terms for both float32 and float64 inputs.
    """

    if not x.is_floating_point():
        raise TypeError("Li2 expects a floating-point tensor")

    one = torch.ones_like(x)
    pi_squared_over_six = x.new_tensor(math.pi**2 / 6)

    below_minus_one = x < -1
    below_zero = (x >= -1) & (x < 0)
    above_half = x > 0.5

    inverse_x = torch.where(below_minus_one, 1 / x, -one)
    negative_core = x / (x - 1)
    inverse_core = inverse_x / (inverse_x - 1)

    core = torch.where(
        below_minus_one,
        inverse_core,
        torch.where(below_zero, negative_core, torch.where(above_half, 1 - x, x)),
    )

    term = core
    series = core
    for k in range(2, 65):
        term = term * core
        series = series + term / (k * k)

    log_one_minus_inverse = torch.log1p(
        -torch.where(below_minus_one, inverse_x, torch.zeros_like(x))
    )
    log_minus_x = torch.log(torch.where(below_minus_one, -x, one))
    large_negative_value = (
        series
        + 0.5 * log_one_minus_inverse.square()
        - pi_squared_over_six
        - 0.5 * log_minus_x.square()
    )

    negative_value = (
        -series - 0.5 * torch.log1p(-torch.where(below_zero, x, torch.zeros_like(x))).square()
    )

    reflected_x = torch.where(above_half & (x < 1), x, one / 2)
    near_one_value = (
        pi_squared_over_six - torch.log(reflected_x) * torch.log1p(-reflected_x) - series
    )

    value = torch.where(
        below_minus_one,
        large_negative_value,
        torch.where(below_zero, negative_value, torch.where(above_half, near_one_value, series)),
    )
    return torch.where(x == 1, pi_squared_over_six, torch.where(x == 0, x, value))


__all__ = ["Li2", "bessel_I1", "bessel_J1", "bessel_K0", "bessel_K1", "bessel_K2"]
