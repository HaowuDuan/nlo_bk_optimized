"""Pointwise LO-like BK kernel K1."""

from __future__ import annotations

import math

import torch

from nlo_torch.bk.config import (
    BKConfig,
    BKOrder,
    ResummationCoupling,
    RunningCouplingLO,
    RunningCouplingNLO,
)
from nlo_torch.coupling import bk_alpha_s
from nlo_torch.numerics.special import bessel_I1, bessel_J1


def Kernel_lo(
    r: torch.Tensor,
    z: torch.Tensor,
    theta: torch.Tensor,
    config: BKConfig,
) -> torch.Tensor:
    """Evaluate the source operation ``Kernel_lo(r, z, theta)`` elementwise."""

    r, z, theta = torch.broadcast_tensors(r, z, theta)
    Y = z
    Xsqr = r.square() + z.square() - 2 * r * z * torch.cos(theta)
    invalid_distance = Xsqr <= 0
    X = torch.sqrt(torch.clamp_min(Xsqr, 0))
    invalid_distance = invalid_distance | (X < 1e-20) | (Y < 1e-20)
    safe_X = torch.where(invalid_distance, torch.ones_like(X), X)
    safe_Y = torch.where(invalid_distance, torch.ones_like(Y), Y)

    min = torch.minimum(torch.minimum(safe_X, safe_Y), r)
    alpha_s_r = _alpha_s(r, config)

    if config.RC_LO in {RunningCouplingLO.BALITSKY_LO, RunningCouplingLO.FIXED_LO}:
        alpha_s_Y = _alpha_s(safe_Y, config)
        alpha_s_X = _alpha_s(safe_X, config)
        lo_kernel = (
            config.NC
            / (2 * math.pi**2)
            * alpha_s_r
            * (
                r.square() / (safe_X.square() * safe_Y.square())
                + (alpha_s_Y / alpha_s_X - 1) / safe_Y.square()
                + (alpha_s_X / alpha_s_Y - 1) / safe_X.square()
            )
        )
        alphas_scale = r
    elif config.RC_LO is RunningCouplingLO.SMALLEST_LO:
        lo_kernel = (
            config.NC * _alpha_s(min, config) / (2 * math.pi**2) * (r / (safe_X * safe_Y)).square()
        )
        alphas_scale = min
    elif config.RC_LO is RunningCouplingLO.PARENT_LO:
        lo_kernel = config.NC * alpha_s_r / (2 * math.pi**2) * (r / (safe_X * safe_Y)).square()
        alphas_scale = r
    elif config.RC_LO is RunningCouplingLO.FAC_LO:
        alphabar_r = alpha_s_r * config.NC / math.pi
        alphabar_X = _alpha_s(safe_X, config) * config.NC / math.pi
        alphabar_Y = _alpha_s(safe_Y, config) * config.NC / math.pi
        lo_kernel = (
            1
            / (2 * math.pi)
            / (
                1 / alphabar_r
                + (safe_X.square() - safe_Y.square())
                / r.square()
                * (alphabar_X - alphabar_Y)
                / (alphabar_X * alphabar_Y)
            )
            * (r / (safe_X * safe_Y)).square()
        )
        alphas_scale = r
    elif config.RC_LO is RunningCouplingLO.BEUF_LO:
        r_eff_sqr = r.square() * torch.pow(
            safe_Y.square() / safe_X.square(),
            (safe_X.square() - safe_Y.square()) / r.square(),
        )
        alphas_scale = torch.sqrt(r_eff_sqr)
        lo_kernel = (
            config.NC
            * _alpha_s(alphas_scale, config)
            / (2 * math.pi**2)
            * (r / (safe_X * safe_Y + 1e-40)).square()
        )
    else:
        raise ValueError(f"unsupported RC_LO choice: {config.RC_LO}")

    lo_kernel = torch.where(torch.isfinite(lo_kernel), lo_kernel, torch.zeros_like(lo_kernel))
    lo_kernel = torch.where(invalid_distance, torch.zeros_like(lo_kernel), lo_kernel)

    if config.Order in {BKOrder.LO, BKOrder.NLO}:
        return lo_kernel

    if config.RESUM_RC is ResummationCoupling.RESUM_RC_PARENT:
        resummation_alpha_s = alpha_s_r
    elif config.RESUM_RC is ResummationCoupling.RESUM_RC_SMALLEST:
        resummation_alpha_s = _alpha_s(min, config)
    elif config.RESUM_RC is ResummationCoupling.RESUM_RC_FIXED:
        resummation_alpha_s = torch.full_like(r, config.FIXED_AS)
    else:
        raise ValueError(f"unsupported RESUM_RC choice: {config.RESUM_RC}")

    x = 4 * torch.log(safe_X / r) * torch.log(safe_Y / r)
    as_x = torch.sqrt(resummation_alpha_s * config.NC / math.pi * torch.abs(x))
    safe_as_x = torch.where(as_x != 0, as_x, torch.ones_like(as_x))
    dlog_J1 = bessel_J1(2 * as_x) / safe_as_x
    dlog_I1 = bessel_I1(2 * as_x) / safe_as_x
    doublelog_resum = torch.where(x >= 0, dlog_J1, dlog_I1)
    doublelog_resum = torch.where(as_x == 0, torch.ones_like(as_x), doublelog_resum)
    doublelog_resum = torch.where(
        r > 1.01 * config.MINR, doublelog_resum, torch.ones_like(doublelog_resum)
    )
    resummation_valid = torch.isfinite(doublelog_resum)

    singlelog_resum = torch.ones_like(r)
    singlelog_resum_expansion = torch.zeros_like(r)
    minxy = torch.minimum(safe_X, safe_Y)
    if config.Order.resum_slog:
        alphabar = resummation_alpha_s * config.NC / math.pi
        A1 = 11 / 12
        singlelog_resum = torch.exp(
            -alphabar * A1 * torch.abs(torch.log(config.KSUB * (r / minxy).square()))
        )
        singlelog_resum_expansion = (
            -alphabar * A1 * torch.abs(2 * torch.log(math.sqrt(config.KSUB) * r / minxy))
        )

    lo_kernel_single_as = (
        _alpha_s(alphas_scale, config)
        * config.NC
        / (2 * math.pi**2)
        * (r / (safe_X * safe_Y)).square()
    )
    subtract = lo_kernel_single_as * singlelog_resum_expansion
    dlog = 0 if config.Order.resum_dlog else 1
    k1fin = (
        lo_kernel
        * _alpha_s(alphas_scale, config)
        * config.NC
        / (4 * math.pi)
        * (
            67 / 9
            - math.pi**2 / 3
            - 10 / 9 * config.NF / config.NC
            - dlog * 2 * 2 * torch.log(safe_X / r) * 2 * torch.log(safe_Y / r)
        )
    )
    result = doublelog_resum * singlelog_resum * lo_kernel - subtract + k1fin
    return torch.where(invalid_distance | ~resummation_valid, torch.zeros_like(result), result)


def _alpha_s(r: torch.Tensor, config: BKConfig) -> torch.Tensor:
    fixed = (
        config.RC_LO is RunningCouplingLO.FIXED_LO or config.RC_NLO is RunningCouplingNLO.FIXED_NLO
    )
    return bk_alpha_s(
        r,
        NC=config.NC,
        NF=config.NF,
        LambdaQCD=config.LambdaQCD,
        C2=config.C2,
        fixed=fixed,
        fixed_alpha_s=config.FIXED_AS,
    )


__all__ = ["Kernel_lo"]
