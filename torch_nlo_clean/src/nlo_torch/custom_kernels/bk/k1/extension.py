"""Local build and Python boundary for the custom CUDA K1 integration kernel."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from nlo_torch.bk.config import BKConfig
from nlo_torch.custom_kernels.extension import load_cuda_extension
from nlo_torch.numerics.interpolation import LogLogSpline


def _load_k1_extension():
    return load_cuda_extension(
        "nlo_torch_k1_cuda",
        Path(__file__).with_name("kernel.cu"),
    )


def k1_cuda_bessel_values(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Development check for the device Bessel implementations used by fused K1."""

    if not x.is_cuda or x.dtype is not torch.float32 or not x.is_contiguous():
        raise ValueError("the custom K1 Bessel check requires contiguous CUDA float32 input")
    outputs = _load_k1_extension().bessel_values(x)
    return outputs[0], outputs[1]


def k1_theta_integrals_cuda(
    r: torch.Tensor,
    z: torch.Tensor,
    interpolator_n: LogLogSpline,
    r_grid: torch.Tensor,
    config: BKConfig,
) -> torch.Tensor:
    """Evaluate the unchanged adaptive K1 theta rule in one CUDA launch."""

    if not z.is_cuda or z.dtype is not torch.float32 or z.ndim != 1 or not z.is_contiguous():
        raise ValueError(
            "the custom K1 theta kernel requires contiguous one-dimensional CUDA input"
        )
    if (
        not r.is_cuda
        or r.dtype is not torch.float32
        or not r.is_contiguous()
        or r.numel() not in {1, z.numel()}
    ):
        raise ValueError("r must be contiguous CUDA float32 and scalar or match z")
    spline = interpolator_n._spline
    return _load_k1_extension().theta_integrals(
        r,
        z,
        r_grid,
        spline.x,
        spline.a,
        spline.b,
        spline.c,
        spline.d,
        config.THETAINTPOINTS,
        config.INTACCURACY,
        config.NC,
        config.NF,
        config.MINR,
        config.KSUB,
        (11 * config.NC - 2 * config.NF) / (12 * math.pi),
        config.LambdaQCD**2,
        4 * config.C2,
        10 * math.log(2.5),
    )


def k1_radial_integrals_cuda(
    r: torch.Tensor,
    interpolator_n: LogLogSpline,
    r_grid: torch.Tensor,
    config: BKConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate complete independent K1 integrals with staged custom CUDA kernels."""

    if not r.is_cuda or r.dtype is not torch.float32 or r.ndim != 1 or not r.is_contiguous():
        raise ValueError("the custom K1 radial kernel requires contiguous one-dimensional CUDA r")
    spline = interpolator_n._spline
    outputs = _load_k1_extension().radial_integrals(
        r,
        r_grid,
        spline.x,
        spline.a,
        spline.b,
        spline.c,
        spline.d,
        config.RINTPOINTS,
        config.THETAINTPOINTS,
        config.INTACCURACY,
        config.NC,
        config.NF,
        config.MINR,
        config.KSUB,
        (11 * config.NC - 2 * config.NF) / (12 * math.pi),
        config.LambdaQCD**2,
        4 * config.C2,
        10 * math.log(2.5),
    )
    return outputs[0], outputs[1], outputs[2], outputs[3]


def k1_fixed_grid_integrals_cuda(
    r: torch.Tensor,
    interpolator_n: LogLogSpline,
    r_grid: torch.Tensor,
    config: BKConfig,
    *,
    refine: bool = True,
    exclude_singular_panels: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate fixed-work K1, optionally refining the two singular-neighbor panels."""

    if not r.is_cuda or r.dtype is not torch.float32 or r.ndim != 1 or not r.is_contiguous():
        raise ValueError("fixed-grid K1 requires contiguous one-dimensional CUDA float32 r")
    if r.numel() == 0:
        raise ValueError("fixed-grid K1 requires at least one parent")
    spline = interpolator_n._spline
    outputs = _load_k1_extension().fixed_grid_integrals(
        r,
        r_grid,
        spline.x,
        spline.a,
        spline.b,
        spline.c,
        spline.d,
        config.NC,
        config.NF,
        config.MINR,
        config.KSUB,
        (11 * config.NC - 2 * config.NF) / (12 * math.pi),
        config.LambdaQCD**2,
        4 * config.C2,
        10 * math.log(2.5),
        refine,
        exclude_singular_panels,
    )
    return outputs[0], outputs[1]


def k1_mixed_fixed_grid_integrals_cuda(
    r: torch.Tensor,
    parent_index: torch.Tensor,
    interpolator_n: LogLogSpline,
    sensitive_interpolator_n: LogLogSpline,
    r_grid: torch.Tensor,
    config: BKConfig,
) -> torch.Tensor:
    """Evaluate the production fixed-grid K1 integral in float64."""

    if not r.is_cuda or r.dtype is not torch.float32 or r.ndim != 1 or not r.is_contiguous():
        raise ValueError("mixed fixed-grid K1 requires contiguous one-dimensional CUDA float32 r")
    if (
        not parent_index.is_cuda
        or parent_index.dtype is not torch.int64
        or parent_index.shape != r.shape
        or not parent_index.is_contiguous()
    ):
        raise ValueError("parent_index must be a matching contiguous CUDA int64 vector")
    regular = interpolator_n._spline
    sensitive = sensitive_interpolator_n._spline
    sensitive_r_grid = getattr(sensitive_interpolator_n, "_r_grid", None)
    if sensitive_r_grid is None:
        sensitive_r_grid = r_grid.double()
    return _load_k1_extension().mixed_fixed_grid_integrals(
        r,
        parent_index,
        r_grid,
        regular.x,
        regular.a,
        regular.b,
        regular.c,
        regular.d,
        sensitive_r_grid,
        sensitive.x,
        sensitive.a,
        sensitive.b,
        sensitive.c,
        sensitive.d,
        config.NC,
        config.NF,
        config.MINR,
        config.KSUB,
        (11 * config.NC - 2 * config.NF) / (12 * math.pi),
        config.LambdaQCD**2,
        4 * config.C2,
        10 * math.log(2.5),
    )


__all__ = [
    "k1_cuda_bessel_values",
    "k1_fixed_grid_integrals_cuda",
    "k1_mixed_fixed_grid_integrals_cuda",
    "k1_radial_integrals_cuda",
    "k1_theta_integrals_cuda",
]
