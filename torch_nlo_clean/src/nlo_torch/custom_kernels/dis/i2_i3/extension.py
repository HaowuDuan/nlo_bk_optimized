"""Python boundary for the fused DIS I2 and I3 CUDA kernels."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from nlo_torch.custom_kernels.extension import load_cuda_extension


def _load_i2_i3_extension():
    return load_cuda_extension(
        "nlo_torch_dis_i2_i3_cuda",
        Path(__file__).with_name("kernel.cu"),
    )


@lru_cache(maxsize=8)
def _mapped_gauss_legendre_rule(
    points: int,
    device_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.legendre.leggauss(points)
    t = (nodes + 1.0) / 2.0
    y = t**3
    mapped_weights = weights / 2.0 * 3.0 * t**2
    with torch.cuda.device(device_index):
        device = torch.device("cuda", device_index)
        return (
            torch.tensor(y, dtype=torch.float32, device=device),
            torch.tensor(mapped_weights, dtype=torch.float32, device=device),
        )


def dis_nested_i2_i3_cuda(
    Q2: torch.Tensor,
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x01sq: torch.Tensor,
    x02sq: torch.Tensor,
    x21sq: torch.Tensor,
    *,
    transverse: bool,
    points: int = 48,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate the inner G functions and return factorized I2 and I3."""

    if not Q2.is_cuda or Q2.dtype is not torch.float32 or Q2.numel() != 1:
        raise ValueError("nested DIS requires a CUDA float32 scalar Q2")
    geometry = (z1, z2, x01sq, x02sq, x21sq)
    if any(
        tensor.device != Q2.device
        or tensor.dtype is not torch.float32
        or tensor.ndim != 1
        or not tensor.is_contiguous()
        for tensor in geometry
    ):
        raise ValueError("nested DIS geometry must use contiguous CUDA float32 vectors")
    if len({tensor.numel() for tensor in geometry}) != 1:
        raise ValueError("nested DIS geometry vectors must have equal lengths")
    if points < 8 or points > 128:
        raise ValueError("nested DIS supports between 8 and 128 inner points")
    device_index = Q2.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    nodes, weights = _mapped_gauss_legendre_rule(points, device_index)
    outputs = _load_i2_i3_extension().dis_nested_i2_i3(
        Q2,
        mf,
        z1,
        z2,
        x01sq,
        x02sq,
        x21sq,
        nodes,
        weights,
        transverse,
    )
    return outputs[0], outputs[1]


def dis_nested_gbw_integrand_cuda(
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    mf: float,
    x: torch.Tensor,
    *,
    transverse: bool,
    contribution: str,
    points: int,
    maxr: float,
    Qs0sqr: float,
    lambda_: float,
    gamma: float,
    x0: float,
    finite_nc: bool,
    parent_coupling: bool,
    smooth_coupling: bool,
    coupling_C2: float,
    active_flavors: int,
    maximum_alpha: float,
) -> torch.Tensor:
    """Evaluate one complete fixed-inner DIS contribution for a GBW dipole."""

    if contribution not in {"I2", "I3"}:
        raise ValueError("nested GBW fusion supports only I2 and I3")
    nodes, weights = _validate_gbw_inputs(Q2, xbj, x, points)
    return _load_i2_i3_extension().dis_nested_gbw_integrand(
        Q2,
        xbj,
        mf,
        x,
        nodes,
        weights,
        transverse,
        contribution == "I3",
        maxr,
        Qs0sqr,
        lambda_,
        gamma,
        x0,
        finite_nc,
        parent_coupling,
        smooth_coupling,
        coupling_C2,
        active_flavors,
        maximum_alpha,
    )


def dis_nested_gbw_i2_i3_integrand_cuda(
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    mf: float,
    x: torch.Tensor,
    *,
    transverse: bool,
    points: int,
    maxr: float,
    Qs0sqr: float,
    lambda_: float,
    gamma: float,
    x0: float,
    finite_nc: bool,
    parent_coupling: bool,
    smooth_coupling: bool,
    coupling_C2: float,
    active_flavors: int,
    maximum_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate complete GBW I2 and I3 contributions in one CUDA pass."""

    nodes, weights = _validate_gbw_inputs(Q2, xbj, x, points)
    outputs = _load_i2_i3_extension().dis_nested_gbw_i2_i3_integrand(
        Q2,
        xbj,
        mf,
        x,
        nodes,
        weights,
        transverse,
        maxr,
        Qs0sqr,
        lambda_,
        gamma,
        x0,
        finite_nc,
        parent_coupling,
        smooth_coupling,
        coupling_C2,
        active_flavors,
        maximum_alpha,
    )
    return outputs[0], outputs[1]


def _validate_gbw_inputs(
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    x: torch.Tensor,
    points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_gbw_sample_inputs(Q2, xbj, x)
    if points < 8 or points > 128:
        raise ValueError("nested DIS supports between 8 and 128 inner points")
    device_index = Q2.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _mapped_gauss_legendre_rule(points, device_index)


def _validate_gbw_sample_inputs(
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    x: torch.Tensor,
) -> None:
    if not Q2.is_cuda or Q2.dtype is not torch.float32 or Q2.numel() != 1:
        raise ValueError("nested DIS requires a CUDA float32 scalar Q2")
    if xbj.device != Q2.device or xbj.dtype is not torch.float32 or xbj.numel() != 1:
        raise ValueError("nested DIS requires a matching CUDA float32 scalar xbj")
    if x.device != Q2.device or x.dtype is not torch.float32 or x.ndim != 2 or x.shape[1] != 5:
        raise ValueError("nested DIS samples must use CUDA float32 shape [samples, 5]")
    if x.stride(0) <= 0 or x.stride(1) <= 0:
        raise ValueError("nested DIS samples must have positive strides")


__all__ = [
    "dis_nested_gbw_i2_i3_integrand_cuda",
    "dis_nested_gbw_integrand_cuda",
    "dis_nested_i2_i3_cuda",
]
