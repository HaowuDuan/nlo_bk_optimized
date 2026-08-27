"""Massive longitudinal NLO DIS impact factors."""

from __future__ import annotations

import math

import torch

from nlo_torch.numerics.special import Li2, bessel_K0, bessel_K1, bessel_K2


def ILdip_massive_Icd(
    Q2: torch.Tensor,
    z1: torch.Tensor,
    r: torch.Tensor,
    mf: float,
    xi: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    front_factor = 4 * Q2 * (z1 * (1 - z1)).square()
    kappa_z = torch.sqrt(z1 * (1 - z1) * Q2 + mf**2)
    bessel_inner_fun = kappa_z * r
    active = bessel_inner_fun >= 1e-7
    safe_bessel_inner_fun = torch.where(active, bessel_inner_fun, torch.ones_like(bessel_inner_fun))

    CLm1 = (
        z1.square()
        * (1 - xi)
        / (1 - z1)
        * (
            -xi.square()
            + x
            * (1 - xi)
            * (1 + (1 - xi) * (1 + z1 * xi / (1 - z1)))
            / (x * (1 - xi) + xi / (1 - z1))
        )
    )
    CLm2 = (
        (1 - z1).square()
        * (1 - xi)
        / z1
        * (
            -xi.square()
            + x * (1 - xi) * (1 + (1 - xi) * (1 + (1 - z1) * xi / z1)) / (x * (1 - xi) + xi / z1)
        )
    )
    kappa1 = (
        xi
        * mf**2
        / ((1 - xi) * (1 - x) * (x * (1 - xi) + xi / (1 - z1)))
        * (xi * (1 - x) + x * (1 - z1 * (1 - xi) / (1 - z1)))
    )
    kappa2 = (
        xi
        * mf**2
        / ((1 - xi) * (1 - x) * (x * (1 - xi) + xi / z1))
        * (xi * (1 - x) + x * (1 - (1 - z1) * (1 - xi) / z1))
    )
    b1 = bessel_K0(safe_bessel_inner_fun)
    Icd_integrand = (
        b1
        * mf**2
        * (
            (b1 - bessel_K0(r * torch.sqrt(kappa_z.square() / (1 - x) + kappa1)))
            * CLm1
            / (
                (1 - xi)
                * (1 - x)
                * (x * (1 - xi) + xi / (1 - z1))
                * (x / (1 - x) * kappa_z.square() + kappa1)
            )
            + (b1 - bessel_K0(r * torch.sqrt(kappa_z.square() / (1 - x) + kappa2)))
            * CLm2
            / (
                (1 - xi)
                * (1 - x)
                * (x * (1 - xi) + xi / z1)
                * (x / (1 - x) * kappa_z.square() + kappa2)
            )
        )
    )
    return front_factor * torch.where(active, Icd_integrand, torch.zeros_like(Icd_integrand))


def ILdip_massive_Iab(
    Q2: torch.Tensor,
    z1: torch.Tensor,
    r: torch.Tensor,
    mf: float,
    xi: torch.Tensor,
) -> torch.Tensor:
    front_factor = 4 * Q2 * (z1 * (1 - z1)).square()
    kappa_z = torch.sqrt(z1 * (1 - z1) * Q2 + mf**2)
    bessel_inner_fun = kappa_z * r
    bessel_arg_2 = torch.sqrt(kappa_z.square() + (1 - z1) * xi / (1 - xi) * mf**2) * r
    bessel_arg_3 = torch.sqrt(kappa_z.square() + z1 * xi / (1 - xi) * mf**2) * r
    b1 = bessel_K0(bessel_inner_fun)
    b2 = bessel_K0(bessel_arg_2)
    b3 = bessel_K0(bessel_arg_3)
    Iab_integrand = b1 / xi * (-2 * torch.log(xi) / (1 - xi) + (1 + xi) / 2) * (2 * b1 - b2 - b3)
    return front_factor * Iab_integrand


def ILdip_massive_Omega_L_Const(
    Q2: torch.Tensor, z: torch.Tensor, r: torch.Tensor, mf: float
) -> torch.Tensor:
    front_factor = 4 * Q2 * (z * (1 - z)).square()
    bessel_inner_fun = torch.sqrt(Q2 * z * (1 - z) + mf**2) * r
    active = bessel_inner_fun >= 1e-7
    safe_argument = torch.where(active, bessel_inner_fun, torch.ones_like(bessel_inner_fun))
    value = (
        front_factor
        * bessel_K0(safe_argument).square()
        * (
            5 / 2
            + (-(math.pi**2) / 3 + math.pi**2 / 6)
            + (1 - 0.5) * torch.log(z / (1 - z)).square()
            + OmegaL_V(Q2, z, mf)
            + L_dip(Q2, z, mf)
        )
    )
    return torch.where(active, value, torch.zeros_like(value))


def OmegaL_V(Q2: torch.Tensor, z: torch.Tensor, mf: float) -> torch.Tensor:
    gamma = torch.sqrt(1 + 4 * mf**2 / Q2)
    return (
        1 / (2 * z) * (torch.log(1 - z) + gamma * torch.log((1 + gamma) / (1 + gamma - 2 * z)))
        + 1
        / (2 * (1 - z))
        * (torch.log(z) + gamma * torch.log((1 + gamma) / (1 + gamma - 2 * (1 - z))))
        + 1
        / (4 * z * (1 - z))
        * (gamma - 1 + 2 * mf**2 / Q2)
        * torch.log((z * (1 - z) * Q2 + mf**2) / mf**2)
    )


def L_dip(Q2: torch.Tensor, z: torch.Tensor, mf: float) -> torch.Tensor:
    gamma = torch.sqrt(1 + 4 * mf**2 / Q2)
    result = (
        Li2(1 / (1 - (1 - gamma) / (2 * z)))
        + Li2(1 / (1 - (1 + gamma) / (2 * z)))
        + Li2(1 / (1 - (1 - gamma) / (2 * (1 - z))))
        + Li2(1 / (1 - (1 + gamma) / (2 * (1 - z))))
    )
    result_mf_0 = math.pi**2 / 6 - 0.5 * torch.log(z / (1 - z)).square()
    return result - result_mf_0


def ILNLOqg_massive_tripole_part_I1(
    Q2: torch.Tensor,
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x01sq: torch.Tensor,
    x02sq: torch.Tensor,
    x21sq: torch.Tensor,
) -> torch.Tensor:
    front_factor = 4 * Q2
    Q = torch.sqrt(Q2)
    z0 = 1 - z1 - z2
    x20x21 = -0.5 * (x01sq - x21sq - x02sq)
    Qbar_k = Q * torch.sqrt(z1 * (1 - z1))
    Qbar_l = Q * torch.sqrt(z0 * (1 - z0))
    omega_k = z0 * z2 / (z1 * (z0 + z2).square())
    omega_l = z1 * z2 / (z0 * (z1 + z2).square())
    x3_k = torch.sqrt(
        z0.square() / (z0 + z2).square() * x02sq + x21sq - 2 * z0 / (z0 + z2) * x20x21
    )
    x3_l = torch.sqrt(
        z1.square() / (z1 + z2).square() * x21sq + x02sq - 2 * z1 / (z1 + z2) * x20x21
    )
    b_k = bessel_K0(
        torch.sqrt(Qbar_k.square() + mf**2) * torch.sqrt(x3_k.square() + omega_k * x02sq)
    )
    b_l = bessel_K0(
        torch.sqrt(Qbar_l.square() + mf**2) * torch.sqrt(x3_l.square() + omega_l * x21sq)
    )
    term_k = z1.square() * (2 * z0 * (z0 + z2) + z2.square()) / x02sq * b_k.square()
    term_l = z0.square() * (2 * z1 * (z1 + z2) + z2.square()) / x21sq * b_l.square()
    term_kl = -2 * z0 * z1 * (z0 * (1 - z0) + z1 * (1 - z1)) * x20x21 / (x02sq * x21sq) * b_k * b_l
    return front_factor * (term_k + term_l + term_kl)


def ILNLOqg_massive_dipole_uvsub(
    Q2: torch.Tensor,
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x01sq: torch.Tensor,
    x02sq: torch.Tensor,
    x21sq: torch.Tensor,
) -> torch.Tensor:
    front_factor = 4 * Q2
    z0 = 1 - z1 - z2
    Q = torch.sqrt(Q2)
    Qbar_k = Q * torch.sqrt(z1 * (1 - z1))
    Qbar_l = Q * torch.sqrt(z0 * (1 - z0))
    term_k = (
        -z1.square()
        * (2 * z0 * (z0 + z2) + z2.square())
        / x02sq
        * torch.exp(-x02sq / x01sq / math.exp(0.5772156649015329))
        * bessel_K0(torch.sqrt(Qbar_k.square() + mf**2) * torch.sqrt(x01sq)).square()
    )
    term_l = (
        -z0.square()
        * (2 * z1 * (z1 + z2) + z2.square())
        / x21sq
        * torch.exp(-x21sq / x01sq / math.exp(0.5772156649015329))
        * bessel_K0(torch.sqrt(Qbar_l.square() + mf**2) * torch.sqrt(x01sq)).square()
    )
    return front_factor * (term_k + term_l)


def ILNLOqg_massive_tripole_part_I2(
    Q2: torch.Tensor,
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x01sq: torch.Tensor,
    x02sq: torch.Tensor,
    x21sq: torch.Tensor,
    y_t: torch.Tensor,
) -> torch.Tensor:
    variables = _longitudinal_tripole_variables(Q2, mf, z1, z2, x01sq, x02sq, x21sq)
    z0, x20x21, Qbar_k, Qbar_l, omega_k, omega_l, lambda_k, lambda_l, x2_k, x2_l, x3_k, x3_l = (
        variables
    )
    int_12_bar_k = G_integrand_simplified(
        1, 2, Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k, y_t
    ) - G_integrand_simplified(1, 2, Qbar_k, mf, x2_k, x3_k, omega_k, 0.0, y_t)
    int_12_bar_l = G_integrand_simplified(
        1, 2, Qbar_l, mf, x2_l, x3_l, omega_l, lambda_l, y_t
    ) - G_integrand_simplified(1, 2, Qbar_l, mf, x2_l, x3_l, omega_l, 0.0, y_t)
    b_k = bessel_K0(
        torch.sqrt(Qbar_k.square() + mf**2) * torch.sqrt(x3_k.square() + omega_k * x02sq)
    )
    b_l = bessel_K0(
        torch.sqrt(Qbar_l.square() + mf**2) * torch.sqrt(x3_l.square() + omega_l * x21sq)
    )
    term_k = z1.square() * (2 * z0 * (z0 + z2) + z2.square()) / 4 * int_12_bar_k * b_k
    term_l = z0.square() * (2 * z1 * (z1 + z2) + z2.square()) / 4 * int_12_bar_l * b_l
    term_kl = (
        -z0
        * z1
        * (z0 * (1 - z0) + z1 * (1 - z1))
        * x20x21
        / 4
        * (int_12_bar_k * b_l / x21sq + int_12_bar_l * b_k / x02sq)
    )
    return 4 * Q2 * (term_k + term_l + term_kl)


def ILNLOqg_massive_tripole_part_I3(
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
    variables = _longitudinal_tripole_variables(Q2, mf, z1, z2, x01sq, x02sq, x21sq)
    z0, x20x21, Qbar_k, Qbar_l, omega_k, omega_l, lambda_k, lambda_l, x2_k, x2_l, x3_k, x3_l = (
        variables
    )
    bar_k1 = G_integrand_simplified(
        1, 2, Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k, y_t1
    ) - G_integrand_simplified(1, 2, Qbar_k, mf, x2_k, x3_k, omega_k, 0.0, y_t1)
    bar_l1 = G_integrand_simplified(
        1, 2, Qbar_l, mf, x2_l, x3_l, omega_l, lambda_l, y_t1
    ) - G_integrand_simplified(1, 2, Qbar_l, mf, x2_l, x3_l, omega_l, 0.0, y_t1)
    bar_k2 = G_integrand_simplified(
        1, 2, Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k, y_t2
    ) - G_integrand_simplified(1, 2, Qbar_k, mf, x2_k, x3_k, omega_k, 0.0, y_t2)
    bar_l2 = G_integrand_simplified(
        1, 2, Qbar_l, mf, x2_l, x3_l, omega_l, lambda_l, y_t2
    ) - G_integrand_simplified(1, 2, Qbar_l, mf, x2_l, x3_l, omega_l, 0.0, y_t2)
    int_11_k1 = G_integrand_simplified(1, 1, Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k, y_t1)
    int_11_l1 = G_integrand_simplified(1, 1, Qbar_l, mf, x2_l, x3_l, omega_l, lambda_l, y_t1)
    int_11_k2 = G_integrand_simplified(1, 1, Qbar_k, mf, x2_k, x3_k, omega_k, lambda_k, y_t2)
    int_11_l2 = G_integrand_simplified(1, 1, Qbar_l, mf, x2_l, x3_l, omega_l, lambda_l, y_t2)
    term_k = z1.square() * (2 * z0 * (z0 + z2) + z2.square()) * x02sq / 64 * bar_k1 * bar_k2
    term_l = z0.square() * (2 * z1 * (z1 + z2) + z2.square()) * x21sq / 64 * bar_l1 * bar_l2
    term_kl = -z1 * z0 * (z1 * (1 - z1) + z0 * (1 - z0)) * x20x21 / 32 * bar_k1 * bar_l2
    term_mf = (
        mf**2
        / 16
        * z2.pow(4)
        * (
            (z1 / (z0 + z2)).square() * int_11_k1 * int_11_k2
            + (z0 / (z1 + z2)).square() * int_11_l1 * int_11_l2
            - 2 * z0 / (z1 + z2) * z1 / (z0 + z2) * int_11_k1 * int_11_l2
        )
    )
    return 4 * Q2 * (term_k + term_l + term_kl + term_mf)


def G_integrand_simplified(
    a: int,
    b: int,
    Qbar: torch.Tensor,
    mf: float,
    x2: torch.Tensor,
    x3: torch.Tensor,
    omega: torch.Tensor,
    lambda_: float | torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    order = a + b - 2
    argument = torch.sqrt(
        (y * lambda_ * mf**2 + Qbar.square() + mf**2) * (y * x3.square() + omega * x2.square()) / y
    )
    if order == 0:
        bessel = bessel_K0(argument)
    elif order == 1:
        bessel = bessel_K1(argument)
    elif order == 2:
        bessel = bessel_K2(argument)
    else:
        raise ValueError("active G_integrand_simplified calls require Bessel order 0, 1, or 2")
    return (
        y.pow(-0.5 * (2 - a + b))
        * 2 ** (a + b - 1)
        * omega.pow(b - 1)
        * (
            (y * lambda_ * mf**2 + Qbar.square() + mf**2) / (y * x3.square() + omega * x2.square())
        ).pow(0.5 * (a + b - 2))
        * bessel
    )


def _longitudinal_tripole_variables(
    Q2: torch.Tensor,
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x01sq: torch.Tensor,
    x02sq: torch.Tensor,
    x21sq: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    z0 = 1 - z1 - z2
    x20x21 = -0.5 * (x01sq - x21sq - x02sq)
    Q = torch.sqrt(Q2)
    Qbar_k = Q * torch.sqrt(z1 * (1 - z1))
    Qbar_l = Q * torch.sqrt(z0 * (1 - z0))
    omega_k = z0 * z2 / (z1 * (z0 + z2).square())
    omega_l = z1 * z2 / (z0 * (z1 + z2).square())
    lambda_k = z1 * z2 / z0
    lambda_l = z0 * z2 / z1
    x2_k = torch.sqrt(x02sq)
    x2_l = torch.sqrt(x21sq)
    x3_k = torch.sqrt(
        z0.square() / (z0 + z2).square() * x02sq + x21sq - 2 * z0 / (z0 + z2) * x20x21
    )
    x3_l = torch.sqrt(
        z1.square() / (z1 + z2).square() * x21sq + x02sq - 2 * z1 / (z1 + z2) * x20x21
    )
    return z0, x20x21, Qbar_k, Qbar_l, omega_k, omega_l, lambda_k, lambda_l, x2_k, x2_l, x3_k, x3_l


__all__ = [
    "G_integrand_simplified",
    "ILNLOqg_massive_dipole_uvsub",
    "ILNLOqg_massive_tripole_part_I1",
    "ILNLOqg_massive_tripole_part_I2",
    "ILNLOqg_massive_tripole_part_I3",
    "ILdip_massive_Iab",
    "ILdip_massive_Icd",
    "ILdip_massive_Omega_L_Const",
    "L_dip",
    "OmegaL_V",
]
