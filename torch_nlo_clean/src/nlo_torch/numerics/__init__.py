"""Shared interpolation, integration, and special-function mathematics."""

from nlo_torch.numerics.integration import (
    IntegralResult,
    adaptive_gauss_kronrod_1d,
    miser,
    tensor_gauss_legendre,
    vegas,
)
from nlo_torch.numerics.interpolation import LogLogSpline, NaturalCubicSpline

__all__ = [
    "IntegralResult",
    "LogLogSpline",
    "NaturalCubicSpline",
    "adaptive_gauss_kronrod_1d",
    "miser",
    "tensor_gauss_legendre",
    "vegas",
]
