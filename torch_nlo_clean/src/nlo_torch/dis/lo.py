"""Leading-order photon wave functions and dipole integrand."""

from __future__ import annotations

import torch

from nlo_torch.dipole.amplitude import GBW, BKDipole
from nlo_torch.dis.config import Polarization, Quark
from nlo_torch.numerics.special import bessel_K0, bessel_K1


def integrand_photon_target_LO(
    r: torch.Tensor,
    z: torch.Tensor,
    xbj: torch.Tensor,
    Q2: torch.Tensor,
    polarization: Polarization,
    quarks: tuple[Quark, ...],
    dipole: GBW | BKDipole,
) -> torch.Tensor:
    """Evaluate the source's ``|Psi|^2 N(r)`` expression without radial Jacobian."""

    r, z, xbj, Q2 = torch.broadcast_tensors(r, z, xbj, Q2)
    result = torch.zeros_like(r)
    for quark in quarks:
        mf = quark.mass
        eps = torch.sqrt(Q2 * z * (1 - z) + mf**2)
        bessel_argument = r * eps
        active = (bessel_argument >= 1e-7) & (bessel_argument <= 5e2)
        safe_argument = torch.where(active, bessel_argument, torch.ones_like(bessel_argument))

        if polarization is Polarization.T:
            contribution = quark.charge**2 * (
                (1 - 2 * z + 2 * z.square()) * eps.square() * bessel_K1(safe_argument).square()
                + (mf * bessel_K0(safe_argument)).square()
            )
        elif polarization is Polarization.L:
            contribution = (
                quark.charge**2
                * 4
                * Q2
                * z.square()
                * (1 - z).square()
                * bessel_K0(safe_argument).square()
            )
        else:
            raise ValueError(f"unknown polarization: {polarization}")
        result = result + torch.where(active, contribution, torch.zeros_like(contribution))

    evolution_rapidity = torch.clamp_min(torch.log(dipole.X0() / xbj), 0)
    return result * dipole.dipole_amplitude(r, evolution_rapidity)


__all__ = ["integrand_photon_target_LO"]
