"""Massive transverse NLO DIS impact factors."""

from __future__ import annotations

import math

import torch

from nlo_torch.dis.longitudinal import G_integrand_simplified, L_dip
from nlo_torch.numerics.special import bessel_K0, bessel_K1


def _weighted_k1_difference_stable(
    r: torch.Tensor, kappa_base: torch.Tensor, delta_kappa_sq: torch.Tensor
) -> torch.Tensor:
    base_sq = kappa_base.square()
    shifted_sq = base_sq + delta_kappa_sq
    valid = (shifted_sq > 0) & (kappa_base > 0)
    rel_delta = torch.abs(delta_kappa_sq) / (base_sq + torch.abs(delta_kappa_sq) + 1e-300)
    linear = -0.5 * r * delta_kappa_sq * bessel_K0(r * kappa_base)
    shifted = torch.sqrt(torch.clamp_min(shifted_sq, 0))
    direct = shifted * bessel_K1(r * shifted) - kappa_base * bessel_K1(r * kappa_base)
    value = torch.where(rel_delta < 1e-5, linear, direct)
    return torch.where(valid, value, torch.zeros_like(value))


def ITdip_massive_0(
    Q2: torch.Tensor, z1: torch.Tensor, x01sq: torch.Tensor, mf: float
) -> torch.Tensor:
    x01 = torch.sqrt(x01sq)
    Q = torch.sqrt(Q2)
    kappa_z = torch.sqrt(z1 * (1 - z1) * Q2 + mf**2)
    common = (
        -(math.pi**2) / 3
        + math.pi**2 / 6
        + 0.5 * torch.log(z1 / (1 - z1)).square()
        + OmegaT_V(Q, z1, mf)
        + L_dip(Q2, z1, mf)
    )
    term1 = (kappa_z * bessel_K1(x01 * kappa_z)).square() * (
        (z1.square() + (1 - z1).square()) * (5 / 2 + common)
        + (2 * z1 - 1) / 2 * OmegaT_N(Q, z1, mf)
    )
    term2 = (mf * bessel_K0(x01 * kappa_z)).square() * (3 + common)
    return term1 + term2


def OmegaT_V(Q: torch.Tensor, z: torch.Tensor, mf: float) -> torch.Tensor:
    return OmegaT_V_unsymmetric(Q, z, mf) + OmegaT_V_unsymmetric(Q, 1 - z, mf)


def OmegaT_V_unsymmetric(Q: torch.Tensor, z: torch.Tensor, mf: float) -> torch.Tensor:
    gamma = torch.sqrt(1 + 4 * (mf / Q) ** 2)
    return (1 + 1 / (2 * z)) * (
        torch.log(1 - z) + gamma * torch.log((1 + gamma) / (1 + gamma - 2 * z))
    ) - 1 / (2 * z) * ((z + 0.5) * (1 - gamma) + mf**2 / Q.square()) * torch.log(
        (z * (1 - z) * Q.square() + mf**2) / mf**2
    )


def OmegaT_N(Q: torch.Tensor, z: torch.Tensor, mf: float) -> torch.Tensor:
    return OmegaT_N_unsymmetric(Q, z, mf) - OmegaT_N_unsymmetric(Q, 1 - z, mf)


def OmegaT_N_unsymmetric(Q: torch.Tensor, z: torch.Tensor, mf: float) -> torch.Tensor:
    gamma = torch.sqrt(1 + 4 * (mf / Q) ** 2)
    return (1 + z - 2 * z.square()) / z * (
        torch.log(1 - z) + gamma * torch.log((1 + gamma) / (1 + gamma - 2 * z))
    ) + (1 - z) / z * ((0.5 + z) * (gamma - 1) - mf**2 / Q.square()) * torch.log(
        (z * (1 - z) * Q.square() + mf**2) / mf**2
    )


def ITdip_massive_1(
    Q2: torch.Tensor,
    z1: torch.Tensor,
    x01sq: torch.Tensor,
    mf: float,
    xi: torch.Tensor,
) -> torch.Tensor:
    x01 = torch.sqrt(x01sq)
    Q = torch.sqrt(Q2)
    kappa_z = torch.sqrt(z1 * (1 - z1) * Q2 + mf**2)
    term1 = (
        kappa_z
        * bessel_K1(x01 * kappa_z)
        * (z1.square() + (1 - z1).square())
        * IT_V1(Q, z1, mf, x01, xi)
    )
    term2 = mf**2 * bessel_K0(x01 * kappa_z) * IT_VMS1(Q, z1, mf, x01, xi)
    return term1 + term2


def IT_V1(
    Q: torch.Tensor, z: torch.Tensor, mf: float, r: torch.Tensor, xi: torch.Tensor
) -> torch.Tensor:
    return IT_V1_unsymmetric(Q, z, mf, r, xi) + IT_V1_unsymmetric(Q, 1 - z, mf, r, xi)


def IT_V1_unsymmetric(
    Q: torch.Tensor, z: torch.Tensor, mf: float, r: torch.Tensor, xi: torch.Tensor
) -> torch.Tensor:
    kappa_z = torch.sqrt(z * (1 - z) * Q.square() + mf**2)
    delta_kappa_sq = xi / (1 - xi) * (1 - z) * mf**2
    shifted = torch.sqrt(kappa_z.square() + delta_kappa_sq)
    difference = shifted * bessel_K1(r * shifted) - kappa_z * bessel_K1(r * kappa_z)
    term1 = 1 / xi * (2 * torch.log(xi) / (1 - xi) - (1 + xi) / 2) * difference
    term2 = (
        -(torch.log(xi) / (1 - xi).square() + z / (1 - xi) + z / 2)
        * (1 - z)
        * mf**2
        / shifted
        * bessel_K1(r * shifted)
    )
    return term1 + term2


def IT_VMS1(
    Q: torch.Tensor, z: torch.Tensor, mf: float, r: torch.Tensor, xi: torch.Tensor
) -> torch.Tensor:
    return IT_VMS1_unsymmetric(Q, z, mf, r, xi) + IT_VMS1_unsymmetric(Q, 1 - z, mf, r, xi)


def IT_VMS1_unsymmetric(
    Q: torch.Tensor, z: torch.Tensor, mf: float, r: torch.Tensor, xi: torch.Tensor
) -> torch.Tensor:
    kappa_z = torch.sqrt(z * (1 - z) * Q.square() + mf**2)
    shifted = torch.sqrt(kappa_z.square() + xi / (1 - xi) * (1 - z) * mf**2)
    term1 = (
        1
        / xi
        * (2 * torch.log(xi) / (1 - xi) - (1 + xi) / 2)
        * (bessel_K0(r * shifted) - bessel_K0(r * kappa_z))
    )
    term2 = (-1.5 * (1 - z) / (1 - xi) + (1 - z) / 2) * bessel_K0(r * shifted)
    return term1 + term2


def ITdip_massive_2(
    Q2: torch.Tensor,
    z1: torch.Tensor,
    x01sq: torch.Tensor,
    mf: float,
    y_chi: torch.Tensor,
    y_u: torch.Tensor,
) -> torch.Tensor:
    x01 = torch.sqrt(x01sq)
    Q = torch.sqrt(Q2)
    kappa_z = torch.sqrt(z1 * (1 - z1) * Q.square() + mf**2)
    term1 = (
        kappa_z
        * bessel_K1(x01 * kappa_z)
        * (
            (z1.square() + (1 - z1).square()) * IT_V2(Q, z1, mf, x01, y_chi, y_u)
            + (2 * z1 - 1) / 2 * IT_N(Q, z1, mf, x01, y_chi, y_u)
        )
    )
    term2 = mf**2 * bessel_K0(x01 * kappa_z) * IT_VMS2(Q, z1, mf, x01, y_chi, y_u)
    return term1 + term2


def IT_V2(
    Q: torch.Tensor,
    z: torch.Tensor,
    mf: float,
    r: torch.Tensor,
    y_chi: torch.Tensor,
    y_u: torch.Tensor,
) -> torch.Tensor:
    return IT_V2_unsymmetric(Q, z, mf, r, y_chi, y_u) + IT_V2_unsymmetric(
        Q, 1 - z, mf, r, y_chi, y_u
    )


def IT_V2_unsymmetric(
    Q: torch.Tensor,
    z: torch.Tensor,
    mf: float,
    r: torch.Tensor,
    y_chi: torch.Tensor,
    y_u: torch.Tensor,
) -> torch.Tensor:
    chi = z * y_chi
    u = (1 - y_u) / y_u
    kappa_z = torch.sqrt(z * (1 - z) * Q.square() + mf**2)
    kappa_chi = torch.sqrt(chi * (1 - chi) * Q.square() + mf**2)
    delta_kappa_sq = u * (1 - z) / (1 - chi) * kappa_chi.square()
    k1_difference = _weighted_k1_difference_stable(r, kappa_z, delta_kappa_sq)
    shifted = torch.sqrt(kappa_z.square() + delta_kappa_sq)
    term1 = (
        -1
        / (1 - chi)
        / (u * (u + 1))
        * mf**2
        / kappa_chi.square()
        * (2 * chi + (u / (u + 1)).square() / z * (z - chi) * (1 - 2 * chi))
        * k1_difference
    )
    term2 = (
        -1
        / (1 - chi).square()
        / (u + 1)
        * (z - chi)
        * (1 - 2 * u / (1 + u) * (z - chi) + (u / (u + 1)).square() / z * (z - chi).square())
        * mf**2
        / shifted
        * bessel_K1(r * shifted)
    )
    return z / y_u.square() * (term1 + term2)


def IT_VMS2(
    Q: torch.Tensor,
    z: torch.Tensor,
    mf: float,
    r: torch.Tensor,
    y_chi: torch.Tensor,
    y_u: torch.Tensor,
) -> torch.Tensor:
    return IT_VMS2_unsymmetric(Q, z, mf, r, y_chi, y_u) + IT_VMS2_unsymmetric(
        Q, 1 - z, mf, r, y_chi, y_u
    )


def IT_VMS2_unsymmetric(
    Q: torch.Tensor,
    z: torch.Tensor,
    mf: float,
    r: torch.Tensor,
    y_chi: torch.Tensor,
    y_u: torch.Tensor,
) -> torch.Tensor:
    chi = z * y_chi
    u = (1 - y_u) / y_u
    kappa_z = torch.sqrt(z * (1 - z) * Q.square() + mf**2)
    kappa_chi = torch.sqrt(chi * (1 - chi) * Q.square() + mf**2)
    shifted = torch.sqrt(kappa_z.square() + u * (1 - z) / (1 - chi) * kappa_chi.square())
    term1 = (
        1
        / (1 - chi)
        / (u + 1).square()
        * (-z - u / (1 + u) * (z + u * chi) / z * (chi - (1 - z)))
        * bessel_K0(r * shifted)
    )
    term2 = (
        1
        / (u + 1) ** 3
        * (
            kappa_z.square() / kappa_chi.square() * (1 + u * chi * (1 - chi) / (z * (1 - z)))
            - mf**2
            / kappa_chi.square()
            * chi
            / (1 - chi)
            * (2 * (1 + u).square() / u + u / (z * (1 - z)) * (z - chi).square())
        )
        * (bessel_K0(r * shifted) - bessel_K0(r * kappa_z))
    )
    return z / y_u.square() * (term1 + term2)


def IT_N(
    Q: torch.Tensor,
    z: torch.Tensor,
    mf: float,
    r: torch.Tensor,
    y_chi: torch.Tensor,
    y_u: torch.Tensor,
) -> torch.Tensor:
    return IT_N_unsymmetric(Q, z, mf, r, y_chi, y_u) - IT_N_unsymmetric(Q, 1 - z, mf, r, y_chi, y_u)


def IT_N_unsymmetric(
    Q: torch.Tensor,
    z: torch.Tensor,
    mf: float,
    r: torch.Tensor,
    y_chi: torch.Tensor,
    y_u: torch.Tensor,
) -> torch.Tensor:
    chi = z * y_chi
    u = (1 - y_u) / y_u
    kappa_z = torch.sqrt(z * (1 - z) * Q.square() + mf**2)
    kappa_chi = torch.sqrt(chi * (1 - chi) * Q.square() + mf**2)
    delta_kappa_sq = u * (1 - z) / (1 - chi) * kappa_chi.square()
    shifted = torch.sqrt(kappa_z.square() + delta_kappa_sq)
    term1 = (
        2
        * (1 - z)
        / z
        / (u + 1) ** 3
        * ((2 + u) * u * z + u.square() * chi)
        * shifted
        * bessel_K1(r * shifted)
    )
    term2 = (
        2
        * (1 - z)
        / z
        / (u + 1) ** 3
        * mf**2
        / kappa_chi.square()
        * (z / (1 - z) + chi / (1 - chi) * (u - 2 * z - 2 * u * chi))
        * (shifted * bessel_K1(r * shifted) - kappa_z * bessel_K1(r * kappa_z))
    )
    return z / y_u.square() * (term1 + term2)


def ITNLOqg_massive_dipole_uvsub(
    Q2: torch.Tensor,
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x01sq: torch.Tensor,
    x02sq: torch.Tensor,
    x21sq: torch.Tensor,
) -> torch.Tensor:
    Q = torch.sqrt(Q2)
    return IT_dipole_jk_I1(Q, mf, z1, z2, x01sq, x02sq, x21sq) + IT_dipole_jkm_I1(
        Q, mf, z1, z2, x01sq, x02sq, x21sq
    )


def ITNLOqg_massive_tripole_part_I1(
    Q2: torch.Tensor,
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x01sq: torch.Tensor,
    x02sq: torch.Tensor,
    x21sq: torch.Tensor,
) -> torch.Tensor:
    Q = torch.sqrt(Q2)
    return (
        IT_tripole_jk_I1(Q, mf, z1, z2, x01sq, x02sq, x21sq)
        + IT_tripole_jkm_I1(Q, mf, z1, z2, x01sq, x02sq, x21sq)
        + IT_tripole_F_I1(Q, mf, z1, z2, x01sq, x02sq, x21sq)
        + IT_tripole_Fm_I1(Q, mf, z1, z2, x01sq, x02sq, x21sq)
    )


def ITNLOqg_massive_tripole_part_I2(
    Q2: torch.Tensor,
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x01sq: torch.Tensor,
    x02sq: torch.Tensor,
    x21sq: torch.Tensor,
    y_t: torch.Tensor,
) -> torch.Tensor:
    Q = torch.sqrt(Q2)
    variables = _tripole_variables(Q, z1, z2, x01sq, x02sq, x21sq)
    return _IT_tripole_I2(mf, z1, z2, y_t, variables)


def ITNLOqg_massive_tripole_part_I3(
    Q2: torch.Tensor,
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x01sq: torch.Tensor,
    x02sq: torch.Tensor,
    x21sq: torch.Tensor,
    y_t1: torch.Tensor,
    y_t2: torch.Tensor,
) -> torch.Tensor:
    Q = torch.sqrt(Q2)
    variables = _tripole_variables(Q, z1, z2, x01sq, x02sq, x21sq)
    return _IT_tripole_I3(mf, z1, z2, y_t1, y_t2, variables)


def IT_dipole_jk_I1(
    Q: torch.Tensor,
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x01sq: torch.Tensor,
    x02sq: torch.Tensor,
    x21sq: torch.Tensor,
) -> torch.Tensor:
    v = _tripole_variables(Q, z1, z2, x01sq, x02sq, x21sq)
    z0, _, Qbar_j, Qbar_k, _, _, _, _, x2_j, x2_k, *_ = v
    term_j = (
        (2 * z0 * (z0 + z2) + z2.square())
        / (z0 + z2).square()
        * (1 - 2 * z1 * (1 - z1))
        * (Qbar_j.square() + mf**2)
        / x2_j.square()
        * -torch.exp(-x2_j.square() / (x01sq * math.exp(0.5772156649015329)))
        * bessel_K1(torch.sqrt(x01sq * (Qbar_j.square() + mf**2))).square()
    )
    term_k = (
        (2 * z1 * (z1 + z2) + z2.square())
        / (z1 + z2).square()
        * (1 - 2 * z0 * (1 - z0))
        * (Qbar_k.square() + mf**2)
        / x2_k.square()
        * -torch.exp(-x2_k.square() / (x01sq * math.exp(0.5772156649015329)))
        * bessel_K1(torch.sqrt(x01sq * (Qbar_k.square() + mf**2))).square()
    )
    return term_j + term_k


def IT_tripole_jk_I1(
    Q: torch.Tensor,
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x01sq: torch.Tensor,
    x02sq: torch.Tensor,
    x21sq: torch.Tensor,
) -> torch.Tensor:
    v = _tripole_variables(Q, z1, z2, x01sq, x02sq, x21sq)
    z0, _, Qbar_j, Qbar_k, omega_j, omega_k, _, _, x2_j, x2_k, x3_j, x3_k, *_ = v
    term_j = (
        (2 * z0 * (z0 + z2) + z2.square())
        / (z0 + z2).square()
        * (1 - 2 * z1 * (1 - z1))
        * (Qbar_j.square() + mf**2)
        / x2_j.square()
        * x3_j.square()
        / (x3_j.square() + omega_j * x2_j.square())
        * bessel_K1(
            torch.sqrt(x3_j.square() + omega_j * x2_j.square())
            * torch.sqrt(Qbar_j.square() + mf**2)
        ).square()
    )
    term_k = (
        (2 * z1 * (z1 + z2) + z2.square())
        / (z1 + z2).square()
        * (1 - 2 * z0 * (1 - z0))
        * (Qbar_k.square() + mf**2)
        / x2_k.square()
        * x3_k.square()
        / (x3_k.square() + omega_k * x2_k.square())
        * bessel_K1(
            torch.sqrt(x3_k.square() + omega_k * x2_k.square())
            * torch.sqrt(Qbar_k.square() + mf**2)
        ).square()
    )
    return term_j + term_k


def IT_dipole_jkm_I1(
    Q: torch.Tensor,
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x01sq: torch.Tensor,
    x02sq: torch.Tensor,
    x21sq: torch.Tensor,
) -> torch.Tensor:
    v = _tripole_variables(Q, z1, z2, x01sq, x02sq, x21sq)
    z0, _, Qbar_j, Qbar_k, _, _, _, _, x2_j, x2_k, *_ = v
    term_j = (
        (2 * z0 * (z0 + z2) + z2.square())
        / (z0 + z2).square()
        / x2_j.square()
        * -torch.exp(-x2_j.square() / (x01sq * math.exp(0.5772156649015329)))
        * bessel_K0(torch.sqrt(x01sq * (Qbar_j.square() + mf**2))).square()
    )
    term_k = (
        (2 * z1 * (z1 + z2) + z2.square())
        / (z1 + z2).square()
        / x2_k.square()
        * -torch.exp(-x2_k.square() / (x01sq * math.exp(0.5772156649015329)))
        * bessel_K0(torch.sqrt(x01sq * (Qbar_k.square() + mf**2))).square()
    )
    return mf**2 * (term_j + term_k)


def IT_tripole_jkm_I1(
    Q: torch.Tensor,
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x01sq: torch.Tensor,
    x02sq: torch.Tensor,
    x21sq: torch.Tensor,
) -> torch.Tensor:
    v = _tripole_variables(Q, z1, z2, x01sq, x02sq, x21sq)
    z0, _, Qbar_j, Qbar_k, omega_j, omega_k, _, _, x2_j, x2_k, x3_j, x3_k, *_ = v
    term_j = (
        (2 * z0 * (z0 + z2) + z2.square())
        / (z0 + z2).square()
        / x2_j.square()
        * bessel_K0(
            torch.sqrt(x3_j.square() + omega_j * x2_j.square())
            * torch.sqrt(Qbar_j.square() + mf**2)
        ).square()
    )
    term_k = (
        (2 * z1 * (z1 + z2) + z2.square())
        / (z1 + z2).square()
        / x2_k.square()
        * bessel_K0(
            torch.sqrt(x3_k.square() + omega_k * x2_k.square())
            * torch.sqrt(Qbar_k.square() + mf**2)
        ).square()
    )
    return mf**2 * (term_j + term_k)


def IT_tripole_F_I1(
    Q: torch.Tensor,
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x01sq: torch.Tensor,
    x02sq: torch.Tensor,
    x21sq: torch.Tensor,
) -> torch.Tensor:
    v = _tripole_variables(Q, z1, z2, x01sq, x02sq, x21sq)
    (
        z0,
        x20x21,
        Qbar_j,
        Qbar_k,
        omega_j,
        omega_k,
        lambda_j,
        lambda_k,
        x2_j,
        x2_k,
        x3_j,
        x3_k,
        x2j_x3j,
        x2k_x3k,
        x2j_x3k,
        x2k_x3j,
        x3j_x3k,
    ) = v
    G22_sing_j = _G22_sing(Qbar_j, mf, x2_j, x3_j, omega_j)
    G22_sing_k = _G22_sing(Qbar_k, mf, x2_k, x3_k, omega_k)
    H_j = _H(Qbar_j, mf, x2_j, x3_j, omega_j, lambda_j)
    H_k = _H(Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k)
    term_1 = (
        4
        / ((z0 + z2) * (z1 + z2))
        * (
            z2 * (z0 - z1).square() * (x2j_x3j * x2k_x3k - x2k_x3j * x2j_x3k)
            - (z1 * (z0 + z2) + z0 * (z1 + z2))
            * (z0 * (z0 + z2) + z1 * (z1 + z2))
            * x20x21
            * x3j_x3k
        )
        * G22_sing_j
        * G22_sing_k
    )
    term_2j = -(z0 + z2) * z1 * z2 / (z1 + z2).square() * x2j_x3j * H_k * G22_sing_j
    term_2k = (z1 + z2) * z0 * z2 / (z0 + z2).square() * x2k_x3k * H_j * G22_sing_k
    term_3j = -z0.square() * z1 * z2 / (z0 + z2).pow(3) * x2j_x3j * H_j * G22_sing_j
    term_3k = z1.square() * z0 * z2 / (z1 + z2).pow(3) * x2k_x3k * H_k * G22_sing_k
    term_4j = (z0 * z2).square() / (8 * (z0 + z2).pow(4)) * H_j.square()
    term_4k = (z1 * z2).square() / (8 * (z1 + z2).pow(4)) * H_k.square()
    return 0.5 * (term_1 + term_2j + term_2k + term_3j + term_3k + term_4j + term_4k)


def IT_tripole_Fm_I1(
    Q: torch.Tensor,
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x01sq: torch.Tensor,
    x02sq: torch.Tensor,
    x21sq: torch.Tensor,
) -> torch.Tensor:
    v = _tripole_variables(Q, z1, z2, x01sq, x02sq, x21sq)
    z0, x20x21, Qbar_j, Qbar_k, omega_j, omega_k, _, _, x2_j, x2_k, x3_j, x3_k, *_ = v
    G12_sing_j = _G12_sing(Qbar_j, mf, x2_j, x3_j, omega_j)
    G12_sing_k = _G12_sing(Qbar_k, mf, x2_k, x3_k, omega_k)
    return (
        0.5
        * mf**2
        * (
            -((2 * z0 + z2) * (2 * z1 + z2) + z2.square())
            * x20x21
            * 8
            * G12_sing_j
            * 8
            * G12_sing_k
            / (32 * (z0 + z2) * (z1 + z2))
        )
    )


def _IT_tripole_I2(
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    y_t: torch.Tensor,
    variables: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    (
        z0,
        x20x21,
        Qbar_j,
        Qbar_k,
        omega_j,
        omega_k,
        lambda_j,
        lambda_k,
        x2_j,
        x2_k,
        x3_j,
        x3_k,
        x2j_x3j,
        x2k_x3k,
        x2j_x3k,
        x2k_x3j,
        x3j_x3k,
    ) = variables

    # The four transverse I2 terms use the same geometry, subtracted G integrands, singular
    # limits, and H functions. Each shared quantity is evaluated once and used below.
    int_22_bar_j = _G_bar(2, 2, Qbar_j, mf, x2_j, x3_j, omega_j, lambda_j, y_t)
    int_22_bar_k = _G_bar(2, 2, Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k, y_t)
    int_12_bar_j = _G_bar(1, 2, Qbar_j, mf, x2_j, x3_j, omega_j, lambda_j, y_t)
    int_12_bar_k = _G_bar(1, 2, Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k, y_t)
    int_21_j = G_integrand_simplified(2, 1, Qbar_j, mf, x2_j, x3_j, omega_j, lambda_j, y_t)
    int_21_k = G_integrand_simplified(2, 1, Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k, y_t)
    int_11_j = G_integrand_simplified(1, 1, Qbar_j, mf, x2_j, x3_j, omega_j, lambda_j, y_t)
    int_11_k = G_integrand_simplified(1, 1, Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k, y_t)
    G12_sing_j = _G12_sing(Qbar_j, mf, x2_j, x3_j, omega_j)
    G12_sing_k = _G12_sing(Qbar_k, mf, x2_k, x3_k, omega_k)
    G22_sing_j = _G22_sing(Qbar_j, mf, x2_j, x3_j, omega_j)
    G22_sing_k = _G22_sing(Qbar_k, mf, x2_k, x3_k, omega_k)
    H_j = _H(Qbar_j, mf, x2_j, x3_j, omega_j, lambda_j)
    H_k = _H(Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k)

    jk_j = (
        (2 * z0 * (z0 + z2) + z2.square())
        / (z0 + z2).square()
        * (1 - 2 * z1 * (1 - z1))
        * int_22_bar_j
        * x3_j.square()
        / 8
        * torch.sqrt((Qbar_j.square() + mf**2) / (x3_j.square() + omega_j * x2_j.square()))
        * bessel_K1(
            torch.sqrt(x3_j.square() + omega_j * x2_j.square())
            * torch.sqrt(Qbar_j.square() + mf**2)
        )
    )
    jk_k = (
        (2 * z1 * (z1 + z2) + z2.square())
        / (z1 + z2).square()
        * (1 - 2 * z0 * (1 - z0))
        * int_22_bar_k
        * x3_k.square()
        / 8
        * torch.sqrt((Qbar_k.square() + mf**2) / (x3_k.square() + omega_k * x2_k.square()))
        * bessel_K1(
            torch.sqrt(x3_k.square() + omega_k * x2_k.square())
            * torch.sqrt(Qbar_k.square() + mf**2)
        )
    )
    term_jk = jk_j + jk_k

    jkm_j = (
        (2 * z0 * (z0 + z2) + z2.square())
        / (z0 + z2).square()
        * int_12_bar_j
        / 4
        * bessel_K0(
            torch.sqrt(x3_j.square() + omega_j * x2_j.square())
            * torch.sqrt(Qbar_j.square() + mf**2)
        )
    )
    jkm_k = (
        (2 * z1 * (z1 + z2) + z2.square())
        / (z1 + z2).square()
        * int_12_bar_k
        / 4
        * bessel_K0(
            torch.sqrt(x3_k.square() + omega_k * x2_k.square())
            * torch.sqrt(Qbar_k.square() + mf**2)
        )
    )
    term_jkm = mf**2 * (jkm_j + jkm_k)

    F_1 = (
        (
            z2 * (z0 - z1).square() * (x2j_x3j * x2k_x3k - x2k_x3j * x2j_x3k)
            - (z1 * (z0 + z2) + z0 * (z1 + z2))
            * (z0 * (z0 + z2) + z1 * (z1 + z2))
            * x20x21
            * x3j_x3k
        )
        / (4 * (z0 + z2) * (z1 + z2))
        * (int_22_bar_k * G22_sing_j + int_22_bar_j * G22_sing_k)
    )
    F_2j = -(z0 + z2) * z1 * z2 / (16 * (z1 + z2).square()) * x2j_x3j * H_k * int_22_bar_j
    F_2k = (z1 + z2) * z0 * z2 / (16 * (z0 + z2).square()) * x2k_x3k * H_j * int_22_bar_k
    F_3j = -z0.square() * z1 * z2 / (16 * (z0 + z2).pow(3)) * x2j_x3j * H_j * int_22_bar_j
    F_3k = z1.square() * z0 * z2 / (16 * (z1 + z2).pow(3)) * x2k_x3k * H_k * int_22_bar_k
    term_F = 0.5 * (F_1 + F_2j + F_2k + F_3j + F_3k)

    Fm_1j = -z0 * z1 * z2.square() / (16 * (z0 + z2).pow(3)) * x2j_x3j * int_21_j * 8 * G12_sing_j
    Fm_1k = z0 * z1 * z2.square() / (16 * (z1 + z2).pow(3)) * x2k_x3k * int_21_k * 8 * G12_sing_k
    Fm_2 = (
        -((2 * z0 + z2) * (2 * z1 + z2) + z2.square())
        * x20x21
        / (32 * (z0 + z2) * (z1 + z2))
        * (int_12_bar_k * 8 * G12_sing_j + int_12_bar_j * 8 * G12_sing_k)
    )
    Fm_3j = (
        -(z0 * z2).square()
        / (16 * (z0 + z2) * (z1 + z2).square())
        * x2j_x3k
        * int_21_k
        * 8
        * G12_sing_j
    )
    Fm_3k = (
        (z1 * z2).square()
        / (16 * (z0 + z2).square() * (z1 + z2))
        * x2k_x3j
        * int_21_j
        * 8
        * G12_sing_k
    )
    Fm_4j = -z0 * z1 * z2.square() / (16 * (z0 + z2).pow(3)) * x2j_x3j * int_11_j * 16 * G22_sing_j
    Fm_4k = z0 * z1 * z2.square() / (16 * (z1 + z2).pow(3)) * x2k_x3k * int_11_k * 16 * G22_sing_k
    Fm_5j = (
        -(z0 + z2) * z2.square() / (16 * (z1 + z2).square()) * x2j_x3j * int_11_k * 16 * G22_sing_j
    )
    Fm_5k = (
        (z1 + z2) * z2.square() / (16 * (z0 + z2).square()) * x2k_x3k * int_11_j * 16 * G22_sing_k
    )
    Fm_6j = z0 * z2.pow(3) / (4 * (z0 + z2).pow(4)) * H_j * int_11_j
    Fm_6k = z1 * z2.pow(3) / (4 * (z1 + z2).pow(4)) * H_k * int_11_k
    term_Fm = (
        0.5
        * mf**2
        * (Fm_1j + Fm_1k + Fm_2 + Fm_3j + Fm_3k + Fm_4j + Fm_4k + Fm_5j + Fm_5k + Fm_6j + Fm_6k)
    )
    return term_jk + term_jkm + term_F + term_Fm


def _IT_tripole_I3(
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    y_t1: torch.Tensor,
    y_t2: torch.Tensor,
    variables: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    (
        z0,
        x20x21,
        Qbar_j,
        Qbar_k,
        omega_j,
        omega_k,
        lambda_j,
        lambda_k,
        x2_j,
        x2_k,
        x3_j,
        x3_k,
        x2j_x3j,
        x2k_x3k,
        x2j_x3k,
        x2k_x3j,
        x3j_x3k,
    ) = variables

    # The four transverse I3 terms share these G integrands. Computing each unique value once
    # removes repeated Bessel evaluations while leaving the source-level terms explicit below.
    int_12_bar_j1 = _G_bar(1, 2, Qbar_j, mf, x2_j, x3_j, omega_j, lambda_j, y_t1)
    int_12_bar_k1 = _G_bar(1, 2, Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k, y_t1)
    int_12_bar_j2 = _G_bar(1, 2, Qbar_j, mf, x2_j, x3_j, omega_j, lambda_j, y_t2)
    int_12_bar_k2 = _G_bar(1, 2, Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k, y_t2)
    int_22_bar_j1 = _G_bar(2, 2, Qbar_j, mf, x2_j, x3_j, omega_j, lambda_j, y_t1)
    int_22_bar_k1 = _G_bar(2, 2, Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k, y_t1)
    int_22_bar_j2 = _G_bar(2, 2, Qbar_j, mf, x2_j, x3_j, omega_j, lambda_j, y_t2)
    int_22_bar_k2 = _G_bar(2, 2, Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k, y_t2)
    int_21_j1 = G_integrand_simplified(2, 1, Qbar_j, mf, x2_j, x3_j, omega_j, lambda_j, y_t1)
    int_21_k1 = G_integrand_simplified(2, 1, Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k, y_t1)
    int_21_j2 = G_integrand_simplified(2, 1, Qbar_j, mf, x2_j, x3_j, omega_j, lambda_j, y_t2)
    int_21_k2 = G_integrand_simplified(2, 1, Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k, y_t2)
    int_11_j1 = G_integrand_simplified(1, 1, Qbar_j, mf, x2_j, x3_j, omega_j, lambda_j, y_t1)
    int_11_k1 = G_integrand_simplified(1, 1, Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k, y_t1)
    int_11_j2 = G_integrand_simplified(1, 1, Qbar_j, mf, x2_j, x3_j, omega_j, lambda_j, y_t2)
    int_11_k2 = G_integrand_simplified(1, 1, Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k, y_t2)

    jkm_j = (
        (2 * z0 * (z0 + z2) + z2.square())
        / (z0 + z2).square()
        * x2_j.square()
        / 64
        * int_12_bar_j1
        * int_12_bar_j2
    )
    jkm_k = (
        (2 * z1 * (z1 + z2) + z2.square())
        / (z1 + z2).square()
        * x2_k.square()
        / 64
        * int_12_bar_k1
        * int_12_bar_k2
    )
    term_jkm = mf**2 * (jkm_j + jkm_k)

    jk_j = (
        (2 * z0 * (z0 + z2) + z2.square())
        / (z0 + z2).square()
        * (1 - 2 * z1 * (1 - z1))
        * x3_j.square()
        * x2_j.square()
        / 256
        * int_22_bar_j1
        * int_22_bar_j2
    )
    jk_k = (
        (2 * z1 * (z1 + z2) + z2.square())
        / (z1 + z2).square()
        * (1 - 2 * z0 * (1 - z0))
        * x3_k.square()
        * x2_k.square()
        / 256
        * int_22_bar_k1
        * int_22_bar_k2
    )
    term_jk = jk_j + jk_k

    term_F = (
        0.5
        * int_22_bar_j1
        * int_22_bar_k2
        / (64 * (z0 + z2) * (z1 + z2))
        * (
            z2 * (z0 - z1).square() * (x2j_x3j * x2k_x3k - x2k_x3j * x2j_x3k)
            - (z1 * (z0 + z2) + z0 * (z1 + z2))
            * (z0 * (z0 + z2) + z1 * (z1 + z2))
            * x20x21
            * x3j_x3k
        )
    )

    Fm_1j = (
        z2.pow(4)
        / (64 * (z0 + z2).pow(4))
        * (4 * z1 * (z1 - 1) + 2)
        * x3_j.square()
        * int_21_j1
        * int_21_j2
    )
    Fm_1k = (
        z2.pow(4)
        / (64 * (z1 + z2).pow(4))
        * (4 * z0 * (z0 - 1) + 2)
        * x3_k.square()
        * int_21_k1
        * int_21_k2
    )
    Fm_2j = -z0 * z1 * z2.square() / (16 * (z0 + z2).pow(3)) * x2j_x3j * int_12_bar_j1 * int_21_j2
    Fm_2k = z0 * z1 * z2.square() / (16 * (z1 + z2).pow(3)) * x2k_x3k * int_12_bar_k1 * int_21_k2
    Fm_3a = (
        -((2 * z0 + z2) * (2 * z1 + z2) + z2.square())
        * x20x21
        / (32 * (z0 + z2) * (z1 + z2))
        * int_12_bar_j1
        * int_12_bar_k2
    )
    Fm_3b = (
        z2.pow(4)
        / (32 * (z0 + z2).square() * (z1 + z2).square())
        * ((2 * z0 + z2) * (2 * z1 + z2) + z2.square())
        * x3j_x3k
        * int_21_j1
        * int_21_k2
    )
    Fm_4j = mf**2 / 8 * (z2 / (z0 + z2)).pow(4) * int_11_j1 * int_11_j2
    Fm_4k = mf**2 / 8 * (z2 / (z1 + z2)).pow(4) * int_11_k1 * int_11_k2
    Fm_5j = (
        -(z0 * z2).square()
        / (16 * (z0 + z2) * (z1 + z2).square())
        * x2j_x3k
        * int_12_bar_j1
        * int_21_k2
    )
    Fm_5k = (
        (z1 * z2).square()
        / (16 * (z1 + z2) * (z0 + z2).square())
        * x2k_x3j
        * int_12_bar_k1
        * int_21_j2
    )
    Fm_6j = -z0 * z1 * z2.square() / (16 * (z0 + z2).pow(3)) * x2j_x3j * int_11_j1 * int_22_bar_j2
    Fm_6k = z0 * z1 * z2.square() / (16 * (z1 + z2).pow(3)) * x2k_x3k * int_11_k1 * int_22_bar_k2
    Fm_7j = (
        -(z0 + z2) * z2.square() / (16 * (z1 + z2).square()) * x2j_x3j * int_11_k1 * int_22_bar_j2
    )
    Fm_7k = (
        (z1 + z2) * z2.square() / (16 * (z0 + z2).square()) * x2k_x3k * int_11_j1 * int_22_bar_k2
    )
    term_Fm = (
        0.5
        * mf**2
        * (
            Fm_1j
            + Fm_2j
            + Fm_1k
            + Fm_2k
            + Fm_3a
            + Fm_3b
            + Fm_4j
            + Fm_4k
            + Fm_5j
            + Fm_5k
            + Fm_6j
            + Fm_6k
            + Fm_7j
            + Fm_7k
        )
    )
    return term_jk + term_jkm + term_F + term_Fm


def _tripole_variables(
    Q: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x01sq: torch.Tensor,
    x02sq: torch.Tensor,
    x21sq: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    z0 = 1 - z1 - z2
    x20x21 = -0.5 * (x01sq - x21sq - x02sq)
    Qbar_j = Q * torch.sqrt(z1 * (1 - z1))
    Qbar_k = Q * torch.sqrt(z0 * (1 - z0))
    omega_j = z0 * z2 / (z1 * (z0 + z2).square())
    omega_k = z1 * z2 / (z0 * (z1 + z2).square())
    lambda_j = z1 * z2 / z0
    lambda_k = z0 * z2 / z1
    x2_j = torch.sqrt(x02sq)
    x2_k = torch.sqrt(x21sq)
    x3_j = torch.sqrt(
        z0.square() / (z0 + z2).square() * x02sq + x21sq - 2 * z0 / (z0 + z2) * x20x21
    )
    x3_k = torch.sqrt(
        z1.square() / (z1 + z2).square() * x21sq + x02sq - 2 * z1 / (z1 + z2) * x20x21
    )
    x2j_x3j = x20x21 - z0 / (z0 + z2) * x02sq
    x2k_x3k = -x20x21 + z1 / (z1 + z2) * x21sq
    x2j_x3k = -x02sq + z1 / (z1 + z2) * x20x21
    x2k_x3j = x21sq - z0 / (z0 + z2) * x20x21
    x3j_x3k = (
        z0 / (z0 + z2) * x02sq
        + z1 / (z1 + z2) * x21sq
        - (1 + z0 * z1 / ((z0 + z2) * (z1 + z2))) * x20x21
    )
    return (
        z0,
        x20x21,
        Qbar_j,
        Qbar_k,
        omega_j,
        omega_k,
        lambda_j,
        lambda_k,
        x2_j,
        x2_k,
        x3_j,
        x3_k,
        x2j_x3j,
        x2k_x3k,
        x2j_x3k,
        x2k_x3j,
        x3j_x3k,
    )


def _G_bar(
    a: int,
    b: int,
    Qbar: torch.Tensor,
    mf: float,
    x2: torch.Tensor,
    x3: torch.Tensor,
    omega: torch.Tensor,
    lambda_: torch.Tensor,
    y_t: torch.Tensor,
) -> torch.Tensor:
    return G_integrand_simplified(
        a, b, Qbar, mf, x2, x3, omega, lambda_, y_t
    ) - G_integrand_simplified(a, b, Qbar, mf, x2, x3, omega, torch.zeros_like(lambda_), y_t)


def _G12_sing(
    Qbar: torch.Tensor,
    mf: float,
    x2: torch.Tensor,
    x3: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    return (
        bessel_K0(torch.sqrt((Qbar.square() + mf**2) * (x3.square() + omega * x2.square())))
        / x2.square()
    )


def _G22_sing(
    Qbar: torch.Tensor,
    mf: float,
    x2: torch.Tensor,
    x3: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.sqrt((Qbar.square() + mf**2) / (x3.square() + omega * x2.square()))
        * bessel_K1(torch.sqrt((Qbar.square() + mf**2) * (x3.square() + omega * x2.square())))
        / x2.square()
    )


def _H(
    Qbar: torch.Tensor,
    mf: float,
    x2: torch.Tensor,
    x3: torch.Tensor,
    omega: torch.Tensor,
    lambda_: torch.Tensor,
) -> torch.Tensor:
    return (
        4
        * torch.sqrt((Qbar.square() + mf**2 * (1 + lambda_)) / (x3.square() + omega * x2.square()))
        * bessel_K1(
            torch.sqrt(
                (Qbar.square() + mf**2 * (1 + lambda_)) * (x3.square() + omega * x2.square())
            )
        )
    )


__all__ = [
    "ITNLOqg_massive_dipole_uvsub",
    "ITNLOqg_massive_tripole_part_I1",
    "ITNLOqg_massive_tripole_part_I2",
    "ITNLOqg_massive_tripole_part_I3",
    "ITdip_massive_0",
    "ITdip_massive_1",
    "ITdip_massive_2",
    "OmegaT_N",
    "OmegaT_V",
]
