"""Local build and Python boundary for fused CUDA log-log spline construction."""

from __future__ import annotations

from pathlib import Path

import torch

from nlo_torch.custom_kernels.extension import load_cuda_extension
from nlo_torch.numerics.interpolation import LogLogSpline


def _load_spline_extension():
    return load_cuda_extension(
        "nlo_torch_spline_cuda",
        Path(__file__).with_name("interpolation.cu"),
    )


def evolution_loglog_splines_cuda(
    grid: torch.Tensor,
    amplitude: torch.Tensor,
    *,
    force_positive: bool,
) -> tuple[LogLogSpline, LogLogSpline]:
    """Construct the evolution N and S splines in one CUDA launch."""

    outputs = _load_spline_extension().evolution_loglog_splines(grid, amplitude, force_positive)
    log_grid, coefficient_a, coefficient_b, coefficient_c, coefficient_d = outputs
    splines = []
    for row in range(2):
        splines.append(
            LogLogSpline.from_coefficients(
                log_grid,
                coefficient_a[row],
                coefficient_b[row],
                coefficient_c[row],
                coefficient_d[row],
            )
        )
    return splines[0], splines[1]


def evolution_loglog_splines_mixed_cuda(
    grid: torch.Tensor,
    amplitude: torch.Tensor,
    *,
    force_positive: bool,
) -> tuple[LogLogSpline, LogLogSpline, LogLogSpline, LogLogSpline]:
    """Construct regular float32 and sensitive float64 N/S splines."""

    outputs = _load_spline_extension().evolution_loglog_splines_mixed(
        grid, amplitude, force_positive
    )
    splines = []
    for offset in (0, 5):
        log_grid = outputs[offset]
        for row in range(2):
            splines.append(
                LogLogSpline.from_coefficients(
                    log_grid,
                    outputs[offset + 1][row],
                    outputs[offset + 2][row],
                    outputs[offset + 3][row],
                    outputs[offset + 4][row],
                )
            )
    sensitive_grid = grid.double()
    splines[2]._r_grid = sensitive_grid
    splines[3]._r_grid = sensitive_grid
    return splines[0], splines[1], splines[2], splines[3]


__all__ = ["evolution_loglog_splines_cuda", "evolution_loglog_splines_mixed_cuda"]
