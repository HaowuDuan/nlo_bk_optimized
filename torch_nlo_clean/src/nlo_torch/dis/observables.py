"""DIS phase-space maps, cross sections, and structure functions."""

from __future__ import annotations

import math

import torch

from nlo_torch.coupling import dis_alpha_s_freeze, dis_alpha_s_smooth
from nlo_torch.dipole.amplitude import GBW, BKDipole
from nlo_torch.dis.config import (
    CF,
    NC,
    AlphaEM,
    DISConfig,
    DISOrder,
    HeavyQuarkX,
    NcScheme,
    Polarization,
    Q0sqr,
    Quark,
    RunningCouplingIRScheme,
    RunningCouplingScheme,
)
from nlo_torch.dis.lo import integrand_photon_target_LO
from nlo_torch.dis.longitudinal import (
    ILdip_massive_Iab,
    ILdip_massive_Icd,
    ILdip_massive_Omega_L_Const,
    ILNLOqg_massive_dipole_uvsub,
    ILNLOqg_massive_tripole_part_I1,
    ILNLOqg_massive_tripole_part_I2,
    ILNLOqg_massive_tripole_part_I3,
)
from nlo_torch.dis.transverse import (
    ITdip_massive_0,
    ITdip_massive_1,
    ITdip_massive_2,
    ITNLOqg_massive_dipole_uvsub,
    ITNLOqg_massive_tripole_part_I1,
    ITNLOqg_massive_tripole_part_I2,
    ITNLOqg_massive_tripole_part_I3,
)
from nlo_torch.numerics.integration import IntegralResult, tensor_gauss_legendre, vegas

ZMIN = 1e-6
Dipole = GBW | BKDipole


def evolution_rapidity_qqg(xbj: torch.Tensor, Q2: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    W2 = Q2 / xbj
    return torch.log(W2 * z2 / Q0sqr)


def evolution_rapidity_dipole(
    xbj: torch.Tensor,
    Q2: torch.Tensor,
    mf: float,
    heavy_quark_x_scheme: HeavyQuarkX,
) -> torch.Tensor:
    W2 = (1 - xbj) * Q2 / xbj
    if heavy_quark_x_scheme is HeavyQuarkX.MassIndependentX:
        X = Q2 / (Q2 + W2)
    elif heavy_quark_x_scheme is HeavyQuarkX.MassDependentX:
        X = (Q2 + 4 * mf**2) / (Q2 + W2)
    else:
        raise ValueError(f"unknown heavy-quark x scheme: {heavy_quark_x_scheme}")
    return torch.log(1 / X)


def z2_lower_bound(xbj: torch.Tensor, Q2: torch.Tensor) -> torch.Tensor:
    W2 = Q2 / xbj
    return Q0sqr / W2


def running_coupling_scale(
    x01: torch.Tensor,
    x02: torch.Tensor,
    x21: torch.Tensor,
    rc_scheme: RunningCouplingScheme,
) -> torch.Tensor:
    if rc_scheme is RunningCouplingScheme.SMALLEST:
        return torch.minimum(x01, torch.minimum(x02, x21))
    if rc_scheme is RunningCouplingScheme.PARENT:
        return x01
    raise ValueError(f"unknown running-coupling scheme: {rc_scheme}")


def tripole_amplitude(
    dipole: Dipole,
    x01: torch.Tensor,
    x02: torch.Tensor,
    x21: torch.Tensor,
    Y: torch.Tensor,
    nc_scheme: NcScheme,
) -> torch.Tensor:
    S01 = 1 - dipole.dipole_amplitude(x01, Y)
    S02 = 1 - dipole.dipole_amplitude(x02, Y)
    S12 = 1 - dipole.dipole_amplitude(x21, Y)
    if nc_scheme is NcScheme.LargeNC:
        return 1 - S02 * S12
    if nc_scheme is NcScheme.FiniteNC:
        return 1 - NC / (2 * CF) * (S02 * S12 - S01 / NC**2)
    raise ValueError(f"unknown Nc scheme: {nc_scheme}")


def integrand_dip_massive(
    x: torch.Tensor,
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    polarization: Polarization,
    quark: Quark,
    contribution: str,
    dipole: Dipole,
    config: DISConfig,
) -> torch.Tensor:
    expected_dimensions = {
        (Polarization.L, "Omega_L_const"): 2,
        (Polarization.L, "ab"): 3,
        (Polarization.L, "cd"): 4,
        (Polarization.T, "T0"): 2,
        (Polarization.T, "T1"): 3,
        (Polarization.T, "T2"): 4,
    }
    dimensions = expected_dimensions.get((polarization, contribution))
    if dimensions is None or x.shape[1] != dimensions:
        raise ValueError(
            f"invalid contribution/dimension pair: {polarization}, {contribution}, {x.shape[1]}"
        )

    if _custom_dis_supported(x, config):
        from nlo_torch.custom_kernels.dis.lo_dipole.extension import (
            dis_dipole_integrand_cuda,
        )

        return dis_dipole_integrand_cuda(
            x,
            Q2,
            xbj,
            polarization,
            quark.mass,
            contribution,
            dipole,
            config,
        )

    z1 = ZMIN + x[:, 0] * (1 - 2 * ZMIN)
    x01 = config.maxr * x[:, 1]
    inside_table = (x01 >= dipole.min_r) & (x01 <= dipole.max_r)
    x01_eval = x01.clamp(min=dipole.min_r, max=dipole.max_r)
    x01sq = x01_eval.square()
    mf = quark.mass
    Y = evolution_rapidity_dipole(xbj, Q2, mf, config.heavy_quark_x_scheme).expand_as(x01_eval)
    N01 = dipole.dipole_amplitude(x01_eval, Y)

    if contribution == "Omega_L_const":
        result = N01 * ILdip_massive_Omega_L_Const(Q2, z1, x01_eval, mf)
    elif contribution == "ab":
        result = N01 * ILdip_massive_Iab(Q2, z1, x01_eval, mf, x[:, 2])
    elif contribution == "cd":
        result = N01 * ILdip_massive_Icd(Q2, z1, x01_eval, mf, x[:, 2], x[:, 3])
    elif contribution == "T0":
        result = N01 * ITdip_massive_0(Q2, z1, x01sq, mf)
    elif contribution == "T1":
        result = N01 * ITdip_massive_1(Q2, z1, x01sq, mf, x[:, 2])
    else:
        result = N01 * ITdip_massive_2(Q2, z1, x01sq, mf, x[:, 2], x[:, 3])

    alphabar = _alpha_s(x01_eval, config) * CF / math.pi
    jacobian = x01 * config.maxr * (1 - 2 * ZMIN)
    result = result * jacobian * alphabar
    return torch.where(inside_table & torch.isfinite(result), result, torch.zeros_like(result))


def integrand_qgunsub_massive(
    x: torch.Tensor,
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    polarization: Polarization,
    quark: Quark,
    contribution: str,
    dipole: Dipole,
    config: DISConfig,
) -> torch.Tensor:
    dimensions = {"I1": 5, "I2": 6, "I3": 7}.get(contribution)
    if dimensions is None or x.shape[1] != dimensions:
        raise ValueError(f"invalid qqg contribution/dimension pair: {contribution}, {x.shape[1]}")

    z2min = z2_lower_bound(xbj, Q2)
    if bool(z2min > 1):
        return x.new_zeros(x.shape[0])
    z1 = (1 - z2min) * x[:, 0]
    z2 = (1 - z1 - z2min) * x[:, 1] + z2min
    x01 = config.maxr * x[:, 2]
    x02 = config.maxr * x[:, 3]
    phix0102 = 2 * math.pi * x[:, 4]
    x01sq = x01.square()
    x02sq = x02.square()
    x21sq = (x01sq + x02sq - 2 * x01 * x02 * torch.cos(phix0102)).clamp_min(0)
    x21 = torch.sqrt(x21sq)
    jacobian = (1 - z2min) * (1 - z1 - z2min) * x01 * x02 * config.maxr**2 * 2 * math.pi
    Y = evolution_rapidity_qqg(xbj, Q2, z2)
    Y_eval = Y.clamp_min(0)
    N012 = tripole_amplitude(dipole, x01, x02, x21, Y_eval, config.nc_scheme)
    N01 = dipole.dipole_amplitude(x01, Y_eval)
    r_alpha = running_coupling_scale(x01, x02, x21, config.rc_scheme)
    alphafac = _alpha_s(r_alpha, config) * CF / math.pi
    mf = quark.mass

    if polarization is Polarization.L:
        if contribution == "I1":
            result = N01 * ILNLOqg_massive_dipole_uvsub(
                Q2, mf, z1, z2, x01sq, x02sq, x21sq
            ) + N012 * ILNLOqg_massive_tripole_part_I1(Q2, mf, z1, z2, x01sq, x02sq, x21sq)
        elif contribution == "I2":
            result = N012 * ILNLOqg_massive_tripole_part_I2(
                Q2, mf, z1, z2, x01sq, x02sq, x21sq, x[:, 5]
            )
        else:
            result = N012 * ILNLOqg_massive_tripole_part_I3(
                Q2, mf, z1, z2, x01sq, x02sq, x21sq, x[:, 5], x[:, 6]
            )
    elif polarization is Polarization.T:
        if contribution == "I1":
            result = N01 * ITNLOqg_massive_dipole_uvsub(
                Q2, mf, z1, z2, x01sq, x02sq, x21sq
            ) + N012 * ITNLOqg_massive_tripole_part_I1(Q2, mf, z1, z2, x01sq, x02sq, x21sq)
        elif contribution == "I2":
            impact_factor = ITNLOqg_massive_tripole_part_I2
            if _custom_dis_supported(Q2, config):
                from nlo_torch.custom_kernels.dis.i2.compiled import (
                    ITNLOqg_massive_tripole_part_I2_fused,
                )

                impact_factor = ITNLOqg_massive_tripole_part_I2_fused
            result = N012 * impact_factor(Q2, mf, z1, z2, x01sq, x02sq, x21sq, x[:, 5])
        else:
            impact_factor = ITNLOqg_massive_tripole_part_I3
            if _custom_dis_supported(Q2, config):
                from nlo_torch.custom_kernels.dis.i3.compiled import (
                    ITNLOqg_massive_tripole_part_I3_fused,
                )

                impact_factor = ITNLOqg_massive_tripole_part_I3_fused
            result = N012 * impact_factor(Q2, mf, z1, z2, x01sq, x02sq, x21sq, x[:, 5], x[:, 6])
    else:
        raise ValueError(f"unknown polarization: {polarization}")

    result = result * jacobian * alphafac / z2
    return torch.where((Y >= 0) & torch.isfinite(result), result, torch.zeros_like(result))


def integrand_qg_nested_massive(
    x: torch.Tensor,
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    polarization: Polarization,
    quark: Quark,
    contribution: str,
    dipole: Dipole,
    config: DISConfig,
) -> torch.Tensor:
    """Integrate the I2/I3 auxiliary variables inside a custom CUDA kernel."""

    if contribution not in {"I2", "I3"} or x.shape[1] != 5:
        raise ValueError(
            f"invalid nested qqg contribution/dimension pair: {contribution}, {x.shape[1]}"
        )
    if not x.is_cuda or x.dtype is not torch.float32:
        raise ValueError("nested qqg integration requires CUDA float32 input")

    from nlo_torch.custom_kernels.dis.i2_i3.extension import (
        dis_nested_gbw_integrand_cuda,
        dis_nested_i2_i3_cuda,
    )

    if isinstance(dipole, GBW):
        return dis_nested_gbw_integrand_cuda(
            Q2,
            xbj,
            quark.mass,
            x,
            transverse=polarization is Polarization.T,
            contribution=contribution,
            points=config.cuda_nested_points,
            maxr=config.maxr,
            Qs0sqr=dipole.Qs0sqr,
            lambda_=dipole.lambda_,
            gamma=dipole.gamma,
            x0=dipole.x0,
            finite_nc=config.nc_scheme is NcScheme.FiniteNC,
            parent_coupling=config.rc_scheme is RunningCouplingScheme.PARENT,
            smooth_coupling=config.rc_ir_scheme is RunningCouplingIRScheme.SMOOTH,
            coupling_C2=config.C2_alpha,
            active_flavors=config.active_flavors,
            maximum_alpha=config.max_alpha_s_freeze,
        )

    z2min = z2_lower_bound(xbj, Q2)
    if bool(z2min > 1):
        return x.new_zeros(x.shape[0])
    z1 = (1 - z2min) * x[:, 0]
    z2 = (1 - z1 - z2min) * x[:, 1] + z2min
    x01 = config.maxr * x[:, 2]
    x02 = config.maxr * x[:, 3]
    phix0102 = 2 * math.pi * x[:, 4]
    x01sq = x01.square()
    x02sq = x02.square()
    x21sq = (x01sq + x02sq - 2 * x01 * x02 * torch.cos(phix0102)).clamp_min(0)
    x21 = torch.sqrt(x21sq)
    jacobian = (1 - z2min) * (1 - z1 - z2min) * x01 * x02 * config.maxr**2 * 2 * math.pi
    Y = evolution_rapidity_qqg(xbj, Q2, z2)
    Y_eval = Y.clamp_min(0)
    N012 = tripole_amplitude(dipole, x01, x02, x21, Y_eval, config.nc_scheme)
    r_alpha = running_coupling_scale(x01, x02, x21, config.rc_scheme)
    alphafac = _alpha_s(r_alpha, config) * CF / math.pi

    I2, I3 = dis_nested_i2_i3_cuda(
        Q2,
        quark.mass,
        z1,
        z2,
        x01sq,
        x02sq,
        x21sq,
        transverse=polarization is Polarization.T,
        points=config.cuda_nested_points,
    )
    impact_factor = I2 if contribution == "I2" else I3
    result = N012 * impact_factor * jacobian * alphafac / z2
    return torch.where((Y >= 0) & torch.isfinite(result), result, torch.zeros_like(result))


def photon_proton_cross_section_LO_d2b(
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    polarization: Polarization,
    dipole: Dipole,
    config: DISConfig,
    *,
    quadrature_points: int = 24,
    batch_size: int = 65_536,
) -> IntegralResult:
    _validate_kinematics(Q2, xbj)
    use_custom = _custom_dis_supported(Q2, config)
    Y = max(math.log(dipole.X0() / float(xbj.item())), 0.0)
    if isinstance(dipole, BKDipole):
        dipole.initialize_interpolation(Y)
    bounds = Q2.new_tensor([[0.0, 1.0], [0.0, 1.0]])

    def function(x: torch.Tensor) -> torch.Tensor:
        z = x[:, 0]
        r = config.maxr * x[:, 1]
        jacobian = 2 * math.pi * r * config.maxr
        return jacobian * integrand_photon_target_LO(
            r, z, xbj, Q2, polarization, config.quarks, dipole
        )

    if use_custom:
        from nlo_torch.custom_kernels.dis.lo_dipole.extension import dis_lo_integrand_cuda
        from nlo_torch.custom_kernels.quadrature import unit_tensor_gauss_legendre_cuda

        def cuda_function(x: torch.Tensor) -> torch.Tensor:
            return dis_lo_integrand_cuda(x, Q2, xbj, polarization, dipole, config)

        value, error, n_eval = unit_tensor_gauss_legendre_cuda(
            cuda_function, Q2, 2, quadrature_points
        )
        result = IntegralResult(
            value,
            error,
            n_eval,
            bool(error <= config.epsrel * value.abs()),
            None,
        )
    else:
        result = tensor_gauss_legendre(
            function,
            bounds,
            points=quadrature_points,
            epsrel=config.epsrel,
            batch_size=batch_size,
        )
    return _scale_result(result, 4 * AlphaEM * NC / (2 * math.pi) ** 2)


def sigma_dip_d2b(
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    polarization: Polarization,
    dipole: Dipole,
    config: DISConfig,
    *,
    quadrature_points: int = 24,
    batch_size: int = 65_536,
    seed: int = 0,
) -> IntegralResult:
    _validate_kinematics(Q2, xbj)
    use_custom = _custom_dis_supported(Q2, config)
    contributions = (
        (("Omega_L_const", 2), ("ab", 3), ("cd", 4))
        if polarization is Polarization.L
        else (("T0", 2), ("T1", 3), ("T2", 4))
    )
    results: list[IntegralResult] = []
    seed_offset = 0
    for quark in config.quarks:
        quark_results: list[IntegralResult] = []
        maxeval = config.maxeval * (10 if quark.mass < 0.4 else 1)
        for contribution, dimensions in contributions:
            bounds = Q2.new_tensor([[0.0, 1.0]] * dimensions)

            def function(x: torch.Tensor, contribution: str = contribution) -> torch.Tensor:
                return integrand_dip_massive(
                    x, Q2, xbj, polarization, quark, contribution, dipole, config
                )

            if dimensions == 2:
                if use_custom:
                    from nlo_torch.custom_kernels.quadrature import (
                        unit_tensor_gauss_legendre_cuda,
                    )

                    value, error, n_eval = unit_tensor_gauss_legendre_cuda(
                        function, Q2, dimensions, quadrature_points
                    )
                    result = IntegralResult(
                        value,
                        error,
                        n_eval,
                        bool(error <= config.epsrel * value.abs()),
                        None,
                    )
                else:
                    result = tensor_gauss_legendre(
                        function,
                        bounds,
                        points=quadrature_points,
                        epsrel=config.epsrel,
                        batch_size=batch_size,
                    )
            else:
                result = _vegas_with_budget(
                    function,
                    bounds,
                    maxeval,
                    config.epsrel,
                    batch_size,
                    seed + seed_offset,
                    cuda_fusion=use_custom,
                )
                seed_offset += 1
            quark_results.append(_scale_result(result, quark.charge**2))
        results.append(_sum_results(quark_results, seed))
    factor = 4 * NC * AlphaEM / (2 * math.pi) ** 2 * 2 * math.pi
    return _scale_result(_sum_results(results, seed), factor)


def sigma_qg_d2b(
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    polarization: Polarization,
    dipole: Dipole,
    config: DISConfig,
    *,
    batch_size: int = 65_536,
    seed: int = 0,
) -> IntegralResult:
    _validate_kinematics(Q2, xbj)
    if _fixed_sobol_supported(Q2, dipole, config):
        from nlo_torch.dis.fixed_sobol import (
            gbw_qg_fixed_sobol,
        )

        fixed_results: list[IntegralResult] = []
        for index, quark in enumerate(config.quarks):
            result = gbw_qg_fixed_sobol(
                Q2,
                xbj,
                polarization,
                quark,
                dipole,
                config,
                seed=seed + index,
            ).total
            if not result.converged:
                break
            fixed_results.append(_scale_result(result, quark.charge**2))
        if len(fixed_results) == len(config.quarks):
            factor = 4 * NC * AlphaEM / (2 * math.pi) ** 3 * 2 * math.pi
            return _scale_result(_sum_results(fixed_results, seed), factor)

    results: list[IntegralResult] = []
    seed_offset = 0
    for quark in config.quarks:
        quark_results: list[IntegralResult] = []
        use_nested = config.cuda_nested and Q2.is_cuda and Q2.dtype is torch.float32
        contribution_dimensions = (
            (("I1", 5), ("I2", 5), ("I3", 5))
            if use_nested
            else (
                ("I1", 5),
                ("I2", 6),
                ("I3", 7),
            )
        )
        for contribution, dimensions in contribution_dimensions:
            bounds = Q2.new_tensor([[0.0, 1.0]] * dimensions)

            def function(x: torch.Tensor, contribution: str = contribution) -> torch.Tensor:
                if use_nested and contribution != "I1":
                    return integrand_qg_nested_massive(
                        x, Q2, xbj, polarization, quark, contribution, dipole, config
                    )
                return integrand_qgunsub_massive(
                    x, Q2, xbj, polarization, quark, contribution, dipole, config
                )

            result = _vegas_with_budget(
                function,
                bounds,
                config.maxeval,
                config.epsrel,
                batch_size,
                seed + seed_offset,
                cuda_fusion=use_nested and config.cuda_fusion,
            )
            seed_offset += 1
            quark_results.append(_scale_result(result, quark.charge**2))
        results.append(_sum_results(quark_results, seed))
    factor = 4 * NC * AlphaEM / (2 * math.pi) ** 3 * 2 * math.pi
    return _scale_result(_sum_results(results, seed), factor)


def photon_proton_cross_section_d2b(
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    polarization: Polarization,
    dipole: Dipole,
    config: DISConfig,
    *,
    quadrature_points: int = 24,
    batch_size: int = 65_536,
    seed: int = 0,
) -> IntegralResult:
    if config.order is DISOrder.LO:
        return photon_proton_cross_section_LO_d2b(
            Q2,
            xbj,
            polarization,
            dipole,
            config,
            quadrature_points=quadrature_points,
            batch_size=batch_size,
        )
    sigma_LO = photon_proton_cross_section_LO_d2b(
        Q2,
        Q2.new_tensor(dipole.X0()),
        polarization,
        dipole,
        config,
        quadrature_points=quadrature_points,
        batch_size=batch_size,
    )
    sigma_dip = sigma_dip_d2b(
        Q2,
        xbj,
        polarization,
        dipole,
        config,
        quadrature_points=quadrature_points,
        batch_size=batch_size,
        seed=seed,
    )
    sigma_qg = sigma_qg_d2b(
        Q2, xbj, polarization, dipole, config, batch_size=batch_size, seed=seed + 100
    )
    return _sum_results((sigma_LO, sigma_dip, sigma_qg), seed)


def FL(
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    dipole: Dipole,
    config: DISConfig,
    **integration_options: int,
) -> IntegralResult:
    sigmaL = photon_proton_cross_section_d2b(
        Q2, xbj, Polarization.L, dipole, config, **integration_options
    )
    return _scale_result(
        sigmaL, float(Q2.item()) / (4 * math.pi**2 * AlphaEM) * config.transverse_area
    )


def FT(
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    dipole: Dipole,
    config: DISConfig,
    **integration_options: int,
) -> IntegralResult:
    sigmaT = photon_proton_cross_section_d2b(
        Q2, xbj, Polarization.T, dipole, config, **integration_options
    )
    return _scale_result(
        sigmaT, float(Q2.item()) / (4 * math.pi**2 * AlphaEM) * config.transverse_area
    )


def F2(
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    dipole: Dipole,
    config: DISConfig,
    **integration_options: int,
) -> IntegralResult:
    options_T = dict(integration_options)
    options_L = dict(integration_options)
    if "seed" in options_L:
        options_L["seed"] += 1_000
    result_T = FT(Q2, xbj, dipole, config, **options_T)
    result_L = FL(Q2, xbj, dipole, config, **options_L)
    return _sum_results((result_T, result_L), integration_options.get("seed"))


def _alpha_s(r: torch.Tensor, config: DISConfig) -> torch.Tensor:
    if config.rc_ir_scheme is RunningCouplingIRScheme.FREEZE:
        return dis_alpha_s_freeze(
            r,
            Nf=config.active_flavors,
            C2=config.C2_alpha,
            max_alpha_s=config.max_alpha_s_freeze,
        )
    if config.rc_ir_scheme is RunningCouplingIRScheme.SMOOTH:
        return dis_alpha_s_smooth(r, Nf=config.active_flavors, C2=config.C2_alpha)
    raise ValueError(f"unknown running-coupling IR scheme: {config.rc_ir_scheme}")


def _custom_dis_supported(reference: torch.Tensor, config: DISConfig) -> bool:
    return config.cuda_fusion and reference.is_cuda and reference.dtype is torch.float32


def _fixed_sobol_supported(Q2: torch.Tensor, dipole: Dipole, config: DISConfig) -> bool:
    """Return whether the validated GBW fixed-Sobol workload covers this calculation."""

    supported = (
        _custom_dis_supported(Q2, config)
        and config.cuda_nested
        and config.cuda_nested_points == 48
        and isinstance(dipole, GBW)
    )
    if not supported:
        return False

    from nlo_torch.dis.fixed_sobol import FIXED_SOBOL_EVALUATIONS

    return config.maxeval >= FIXED_SOBOL_EVALUATIONS


def _vegas_with_budget(
    function,
    bounds: torch.Tensor,
    maxeval: int,
    epsrel: float,
    batch_size: int,
    seed: int,
    *,
    cuda_fusion: bool = False,
) -> IntegralResult:
    if maxeval < 2:
        raise ValueError("NLO integration requires maxeval >= 2")
    max_iterations = min(8, max(1, maxeval // 2))
    samples_per_iteration = max(2, maxeval // max_iterations)
    min_iterations = min(3, max_iterations)
    return vegas(
        function,
        bounds,
        samples_per_iteration=samples_per_iteration,
        max_iterations=max_iterations,
        min_iterations=min_iterations,
        batch_size=batch_size,
        epsrel=epsrel,
        seed=seed,
        cuda_mask_fusion=cuda_fusion,
    )


def _scale_result(result: IntegralResult, factor: float) -> IntegralResult:
    return IntegralResult(
        result.value * factor,
        result.error * abs(factor),
        result.n_eval,
        result.converged,
        result.seed,
    )


def _sum_results(
    results: tuple[IntegralResult, ...] | list[IntegralResult], seed: int | None
) -> IntegralResult:
    if not results:
        raise ValueError("at least one integration result is required")
    value = torch.stack([result.value for result in results]).sum()
    error = torch.sqrt(torch.stack([result.error.square() for result in results]).sum())
    return IntegralResult(
        value,
        error,
        sum(result.n_eval for result in results),
        all(result.converged for result in results),
        seed,
    )


def _validate_kinematics(Q2: torch.Tensor, xbj: torch.Tensor) -> None:
    if Q2.numel() != 1 or xbj.numel() != 1:
        raise ValueError("Q2 and xbj must be scalar tensors")
    if Q2.device != xbj.device or Q2.dtype != xbj.dtype:
        raise ValueError("Q2 and xbj must have the same device and dtype")
    if not bool(torch.isfinite(Q2)) or not bool(torch.isfinite(xbj)):
        raise ValueError("Q2 and xbj must be finite")
    if not bool(Q2 > 0) or not bool((xbj > 0) & (xbj <= 1)):
        raise ValueError("Q2 must be positive and xbj must lie in (0, 1]")


__all__ = [
    "F2",
    "FL",
    "FT",
    "evolution_rapidity_dipole",
    "evolution_rapidity_qqg",
    "integrand_dip_massive",
    "integrand_qg_nested_massive",
    "integrand_qgunsub_massive",
    "photon_proton_cross_section_LO_d2b",
    "photon_proton_cross_section_d2b",
    "running_coupling_scale",
    "sigma_dip_d2b",
    "sigma_qg_d2b",
    "tripole_amplitude",
    "z2_lower_bound",
]
