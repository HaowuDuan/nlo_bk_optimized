"""Local build and Python boundary for the custom CUDA NLO BK producer."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from nlo_torch.bk.config import BKConfig, RunningCouplingNLO
from nlo_torch.custom_kernels.extension import load_cuda_extension
from nlo_torch.numerics.interpolation import LogLogSpline


def _load_bk_extension():
    return load_cuda_extension(
        "nlo_torch_bk_nlo_cuda",
        Path(__file__).with_name("kernel.cu"),
    )


def bk_nlo_mixed_vegas_summaries_cuda(
    r: torch.Tensor,
    edges: torch.Tensor,
    bounds_lower: torch.Tensor,
    bounds_width: torch.Tensor,
    bin_index: torch.Tensor,
    random: torch.Tensor,
    volume: torch.Tensor,
    interpolator_n: LogLogSpline,
    sensitive_interpolator: LogLogSpline,
    r_grid: torch.Tensor,
    config: BKConfig,
    *,
    sensitive_ratio: float = 1e-2,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return histogram weights and disjoint mixed-precision block moments."""

    if config.RC_NLO is not RunningCouplingNLO.SMALLEST_NLO or config.NF > 3:
        raise ValueError("mixed Vegas CUDA fusion requires smallest-distance running with NF <= 3")
    if not 0 < sensitive_ratio < 1:
        raise ValueError("sensitive_ratio must lie between zero and one")
    regular_spline = interpolator_n._spline
    sensitive_spline = sensitive_interpolator._spline
    sensitive_r_grid = getattr(sensitive_interpolator, "_r_grid", None)
    if sensitive_r_grid is None:
        sensitive_r_grid = r_grid.double()
    outputs = _load_bk_extension().bk_nlo_mixed_vegas_summaries(
        r,
        edges,
        bounds_lower,
        bounds_width,
        bin_index,
        random,
        volume,
        r_grid,
        regular_spline.x,
        regular_spline.a,
        regular_spline.b,
        regular_spline.c,
        regular_spline.d,
        sensitive_r_grid,
        sensitive_spline.x,
        sensitive_spline.a,
        sensitive_spline.b,
        sensitive_spline.c,
        sensitive_spline.d,
        sensitive_ratio,
        *_bk_physics_arguments(config),
    )
    return tuple(outputs)


def _bk_physics_arguments(config: BKConfig) -> tuple:
    return (
        config.NF,
        config.NC,
        config.SYMMETRIZE_Z_Z2_INTEGRATION,
        8 * torch.pi**4,
        (11 * config.NC - 2 * config.NF) / (12 * math.pi),
        config.LambdaQCD**2,
        4 * config.C2,
        10 * math.log(2.5),
    )


__all__ = [
    "bk_nlo_mixed_vegas_summaries_cuda",
]
