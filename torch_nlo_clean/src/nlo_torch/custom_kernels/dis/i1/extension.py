"""Python boundary for the DIS I1 CUDA kernels."""

from __future__ import annotations

from pathlib import Path

import torch

from nlo_torch.custom_kernels.extension import load_cuda_extension


def _load_i1_extension():
    return load_cuda_extension(
        "nlo_torch_dis_i1_cuda",
        Path(__file__).with_name("kernel.cu"),
    )


def dis_gbw_i1_integrand_regions_cuda(
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    mf: float,
    x: torch.Tensor,
    *,
    transverse: bool,
    sensitive_ratio: float = 1e-4,
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
    """Return disjoint regular-float32 and singular-float64 I1 samples."""

    _validate_gbw_sample_inputs(Q2, xbj, x)
    if not 0 < sensitive_ratio < 1:
        raise ValueError("the DIS I1 sensitive-region ratio must be between zero and one")
    outputs = _load_i1_extension().dis_gbw_i1_regions(
        Q2,
        xbj,
        mf,
        x,
        transverse,
        sensitive_ratio,
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


def dis_gbw_i1_region_sums_cuda(
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    mf: float,
    x: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    transverse: bool,
    sensitive_ratio: float = 1e-4,
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
    """Return weighted block sums for the two disjoint I1 precision regions."""

    _validate_gbw_sample_inputs(Q2, xbj, x)
    if (
        sample_weights.device != Q2.device
        or sample_weights.dtype is not torch.float32
        or sample_weights.ndim != 1
        or sample_weights.numel() != x.shape[0]
        or sample_weights.stride(0) <= 0
    ):
        raise ValueError("DIS I1 weights must be a matching CUDA float32 vector")
    if not 0 < sensitive_ratio < 1:
        raise ValueError("the DIS I1 sensitive-region ratio must be between zero and one")
    outputs = _load_i1_extension().dis_gbw_i1_region_sums(
        Q2,
        xbj,
        mf,
        x,
        sample_weights,
        transverse,
        sensitive_ratio,
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


def _validate_gbw_sample_inputs(
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    x: torch.Tensor,
) -> None:
    if not Q2.is_cuda or Q2.dtype is not torch.float32 or Q2.numel() != 1:
        raise ValueError("DIS I1 requires a CUDA float32 scalar Q2")
    if xbj.device != Q2.device or xbj.dtype is not torch.float32 or xbj.numel() != 1:
        raise ValueError("DIS I1 requires a matching CUDA float32 scalar xbj")
    if x.device != Q2.device or x.dtype is not torch.float32 or x.ndim != 2 or x.shape[1] != 5:
        raise ValueError("DIS I1 samples must use CUDA float32 shape [samples, 5]")
    if x.stride(0) <= 0 or x.stride(1) <= 0:
        raise ValueError("DIS I1 samples must have positive strides")


__all__ = ["dis_gbw_i1_integrand_regions_cuda", "dis_gbw_i1_region_sums_cuda"]
