"""Python boundary for the DIS LO and NLO dipole custom kernels."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import torch

from nlo_torch.custom_kernels.extension import load_cuda_extension
from nlo_torch.dipole.amplitude import GBW, BKDipole
from nlo_torch.dis.config import (
    DISConfig,
    HeavyQuarkX,
    Polarization,
    RunningCouplingIRScheme,
)


def _load_lo_dipole_extension():
    return load_cuda_extension(
        "nlo_torch_dis_lo_dipole_cuda",
        Path(__file__).with_name("kernel.cu"),
    )


@lru_cache(maxsize=16)
def _dummy(device_index: int) -> torch.Tensor:
    with torch.cuda.device(device_index):
        return torch.zeros(1, dtype=torch.float32, device=torch.device("cuda", device_index))


@lru_cache(maxsize=16)
def _quark_data(
    masses: tuple[float, ...],
    charge_squares: tuple[float, ...],
    device_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device_index):
        return (
            torch.tensor(masses, dtype=torch.float32, device=device),
            torch.tensor(charge_squares, dtype=torch.float32, device=device),
        )


def dis_lo_integrand_cuda(
    samples: torch.Tensor,
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    polarization: Polarization,
    dipole: GBW | BKDipole,
    config: DISConfig,
) -> torch.Tensor:
    """Evaluate the complete mapped LO integrand in one CUDA kernel."""

    _validate_samples(samples, Q2, xbj, 2)
    device_index = _device_index(samples)
    masses = tuple(quark.mass for quark in config.quarks)
    charge_squares = tuple(quark.charge**2 for quark in config.quarks)
    mass_tensor, charge_tensor = _quark_data(masses, charge_squares, device_index)
    rapidity = max(math.log(dipole.X0() / float(xbj.item())), 0.0)
    amplitude_arguments = _amplitude_arguments(dipole, rapidity, samples)
    lo_amplitude_arguments = amplitude_arguments[:7] + amplitude_arguments[9:]
    return _load_lo_dipole_extension().dis_lo_integrand(
        samples,
        float(Q2.item()),
        float(xbj.item()),
        polarization is Polarization.T,
        mass_tensor,
        charge_tensor,
        config.maxr,
        *lo_amplitude_arguments,
    )


def dis_dipole_integrand_cuda(
    samples: torch.Tensor,
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    polarization: Polarization,
    mf: float,
    contribution: str,
    dipole: GBW | BKDipole,
    config: DISConfig,
) -> torch.Tensor:
    """Evaluate one complete mapped NLO dipole contribution in one CUDA kernel."""

    contribution_codes = {
        (Polarization.L, "Omega_L_const"): (0, 2),
        (Polarization.L, "ab"): (1, 3),
        (Polarization.L, "cd"): (2, 4),
        (Polarization.T, "T0"): (3, 2),
        (Polarization.T, "T1"): (4, 3),
        (Polarization.T, "T2"): (5, 4),
    }
    try:
        contribution_code, dimensions = contribution_codes[(polarization, contribution)]
    except KeyError as error:
        raise ValueError(f"unknown dipole contribution: {polarization}, {contribution}") from error
    _validate_samples(samples, Q2, xbj, dimensions)
    Q2_value = float(Q2.item())
    xbj_value = float(xbj.item())
    W2 = (1.0 - xbj_value) * Q2_value / xbj_value
    X = (
        (Q2_value + 4.0 * mf**2) / (Q2_value + W2)
        if config.heavy_quark_x_scheme is HeavyQuarkX.MassDependentX
        else Q2_value / (Q2_value + W2)
    )
    rapidity = math.log(1.0 / X)
    return _load_lo_dipole_extension().dis_dipole_integrand(
        samples,
        Q2_value,
        xbj_value,
        mf,
        contribution_code,
        config.maxr,
        config.heavy_quark_x_scheme is HeavyQuarkX.MassDependentX,
        config.rc_ir_scheme is RunningCouplingIRScheme.SMOOTH,
        config.C2_alpha,
        config.active_flavors,
        config.max_alpha_s_freeze,
        *_amplitude_arguments(dipole, rapidity, samples),
    )


def _amplitude_arguments(
    dipole: GBW | BKDipole,
    rapidity: float,
    reference: torch.Tensor,
) -> tuple[object, ...]:
    if isinstance(dipole, GBW):
        dummy = _dummy(_device_index(reference))
        return (
            0,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            -1.0,
            dipole.min_r,
            dipole.max_r,
            dipole.Qs0sqr,
            dipole.lambda_,
            dipole.gamma,
            dipole.x0,
        )

    if dipole.table.r.device != reference.device or dipole.table.r.dtype is not torch.float32:
        raise ValueError("the BK table must be CUDA float32 for custom DIS kernels")
    dipole.initialize_interpolation(rapidity)
    spline = dipole._interpolator
    if spline is None:
        raise RuntimeError("BK interpolation initialization failed")
    return (
        1,
        spline.x.contiguous(),
        spline.a.contiguous(),
        spline.b.contiguous(),
        spline.c.contiguous(),
        spline.d.contiguous(),
        dipole.maxr_interpolate,
        dipole.min_r,
        dipole.max_r,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _validate_samples(
    samples: torch.Tensor,
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    dimensions: int,
) -> None:
    if (
        not samples.is_cuda
        or samples.dtype is not torch.float32
        or samples.ndim != 2
        or samples.shape[1] != dimensions
        or samples.stride(0) <= 0
        or samples.stride(1) <= 0
    ):
        raise ValueError(f"custom DIS samples require CUDA float32 shape [samples, {dimensions}]")
    if (
        Q2.device != samples.device
        or xbj.device != samples.device
        or Q2.dtype is not torch.float32
        or xbj.dtype is not torch.float32
        or Q2.numel() != 1
        or xbj.numel() != 1
    ):
        raise ValueError("custom DIS kinematics must be matching CUDA float32 scalars")


def _device_index(reference: torch.Tensor) -> int:
    index = reference.device.index
    return torch.cuda.current_device() if index is None else index


__all__ = ["dis_dipole_integrand_cuda", "dis_lo_integrand_cuda"]
