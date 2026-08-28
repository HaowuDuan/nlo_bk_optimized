"""Integrated rapidity derivatives for BK evolution."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from nlo_torch.bk.config import (
    BKConfig,
    BKOrder,
    IntegrationMethod,
    ResummationCoupling,
    RunningCouplingLO,
    RunningCouplingNLO,
)
from nlo_torch.bk_kernels.k1 import Kernel_lo
from nlo_torch.bk_kernels.k2 import Kernel_nlo
from nlo_torch.bk_kernels.kf import Kernel_nlo_fermion
from nlo_torch.coupling import bk_alpha_s
from nlo_torch.numerics.integration import (
    IntegralResult,
    VegasState,
    _gauss_kronrod_intervals,
    adaptive_gauss_kronrod_1d,
    miser,
    vegas,
)
from nlo_torch.numerics.interpolation import LogLogSpline, NaturalCubicSpline


@dataclass(frozen=True, slots=True)
class BatchedIntegralResult:
    """Adaptive integration results for independent parent-dipole sizes."""

    value: torch.Tensor
    error: torch.Tensor
    n_eval: torch.Tensor
    converged: torch.Tensor


def rapidity_derivative_lo(
    r: torch.Tensor,
    interpolator_N: LogLogSpline,
    r_grid: torch.Tensor,
    config: BKConfig,
    *,
    rapidity: torch.Tensor | None = None,
    history_y: torch.Tensor | None = None,
    history_N: torch.Tensor | None = None,
) -> IntegralResult:
    """Integrate the LO-like derivative over ``log(z)`` and ``theta``."""

    if r.numel() != 1:
        raise ValueError("rapidity_derivative_lo requires one parent r")
    if config.KINEMATICAL_CONSTRAINT and (
        rapidity is None or history_y is None or history_N is None
    ):
        raise ValueError("kinematical evolution requires rapidity and amplitude history")

    def integrate_theta(log_z: torch.Tensor) -> torch.Tensor:
        if config.CUDA_FUSION and log_z.is_cuda and log_z.dtype is torch.float32:
            return _batched_lo_theta_integrals(
                r,
                log_z,
                interpolator_N,
                r_grid,
                config,
                rapidity,
                history_y,
                history_N,
            )

        values = []
        for log_z_value in log_z.reshape(-1):
            z = torch.exp(log_z_value)

            def theta_integrand(theta: torch.Tensor) -> torch.Tensor:
                return _lo_integrand(
                    r,
                    z,
                    theta,
                    interpolator_N,
                    r_grid,
                    config,
                    rapidity,
                    history_y,
                    history_N,
                )

            theta_result = adaptive_gauss_kronrod_1d(
                theta_integrand,
                r.new_tensor(0.0),
                r.new_tensor(math.pi),
                epsrel=config.INTACCURACY,
                max_intervals=config.THETAINTPOINTS,
                rule=21,
            )
            values.append(theta_result.value * torch.exp(2 * log_z_value) * 2)
        return torch.stack(values).reshape(log_z.shape)

    return adaptive_gauss_kronrod_1d(
        integrate_theta,
        torch.log(r_grid[0]),
        torch.log(r_grid[-1]),
        epsrel=config.INTACCURACY,
        max_intervals=config.RINTPOINTS,
        rule=21,
    )


def rapidity_derivative_lo_batch(
    r: torch.Tensor,
    interpolator_N: LogLogSpline,
    r_grid: torch.Tensor,
    config: BKConfig,
) -> BatchedIntegralResult:
    """Integrate complete K1 terms for many parent sizes in shared CUDA launches.

    Each row retains its own largest-error interval, stopping decision, evaluation
    count, and convergence result. Only the scheduling across rows is batched.
    """

    if r.ndim != 1 or r.numel() == 0:
        raise ValueError("r must be a non-empty one-dimensional tensor")
    if not config.CUDA_FUSION or not r.is_cuda or r.dtype is not torch.float32:
        raise ValueError("batched K1 requires CUDA float32 with CUDA_FUSION enabled")
    if not _persistent_k1_supported(config):
        raise ValueError("the selected BK configuration is not supported by batched K1")

    from nlo_torch.custom_kernels.bk.k1.extension import k1_radial_integrals_cuda

    result, result_error, result_evaluations, result_converged = k1_radial_integrals_cuda(
        r, interpolator_N, r_grid, config
    )
    return BatchedIntegralResult(
        value=result,
        error=result_error,
        n_eval=result_evaluations,
        converged=result_converged,
    )


def _persistent_k1_supported(config: BKConfig) -> bool:
    return (
        not config.KINEMATICAL_CONSTRAINT
        and config.RC_LO is RunningCouplingLO.BALITSKY_LO
        and config.RESUM_RC is ResummationCoupling.RESUM_RC_SMALLEST
        and config.Order is BKOrder.NLO_RESUM_DLOG_SLOG
        and config.NF <= 3
        and config.RINTPOINTS <= 85
        and config.THETAINTPOINTS <= 85
    )


def _fixed_k1_supported(config: BKConfig) -> bool:
    """Return whether the configuration matches the validated fixed K1 rule."""

    return (
        _persistent_k1_supported(config)
        and config.RPOINTS == 100
        and config.RINTPOINTS == 85
        and config.THETAINTPOINTS == 85
        and config.INTACCURACY == 0.001
    )


def _batched_lo_theta_integrals(
    r: torch.Tensor,
    log_z: torch.Tensor,
    interpolator_N: LogLogSpline,
    r_grid: torch.Tensor,
    config: BKConfig,
    rapidity: torch.Tensor | None,
    history_y: torch.Tensor | None,
    history_N: torch.Tensor | None,
) -> torch.Tensor:
    """Integrate independent K1 theta slices together without changing their adaptive rules."""

    flat_log_z = log_z.reshape(-1)
    z = torch.exp(flat_log_z)
    use_persistent_cuda = _persistent_k1_supported(config)
    if use_persistent_cuda:
        from nlo_torch.custom_kernels.bk.k1.extension import k1_theta_integrals_cuda

        theta_integral = k1_theta_integrals_cuda(r, z, interpolator_N, r_grid, config)
        return (theta_integral * torch.exp(2 * flat_log_z) * 2).reshape(log_z.shape)

    batch = z.numel()
    interval_lower = z.new_zeros((batch, 1))
    interval_upper = z.new_full((batch, 1), math.pi)
    active_z = z
    active_index = torch.arange(batch, device=z.device)
    result = torch.empty_like(z)

    def theta_integrand(theta: torch.Tensor, paired_z: torch.Tensor) -> torch.Tensor:
        return _lo_integrand(
            r,
            paired_z,
            theta,
            interpolator_N,
            r_grid,
            config,
            rapidity,
            history_y,
            history_N,
        )

    estimate, error = _gauss_kronrod_intervals(
        theta_integrand,
        interval_lower[:, 0],
        interval_upper[:, 0],
        21,
        active_z,
    )
    estimate = estimate.unsqueeze(1)
    error = error.unsqueeze(1)

    while active_z.numel() > 0:
        total_estimate = torch.sum(estimate, dim=1)
        total_error = torch.sum(error, dim=1)
        host_estimate = total_estimate.detach().cpu().tolist()
        host_error = total_error.detach().cpu().tolist()
        interval_count = estimate.shape[1]
        finished = torch.tensor(
            [
                interval_count >= config.THETAINTPOINTS
                or item_error <= config.INTACCURACY * abs(item_estimate)
                for item_estimate, item_error in zip(host_estimate, host_error, strict=True)
            ],
            device=z.device,
            dtype=torch.bool,
        )
        result[active_index[finished]] = total_estimate[finished]
        if bool(finished.all()):
            break

        keep = ~finished
        active_z = active_z[keep]
        active_index = active_index[keep]
        interval_lower = interval_lower[keep]
        interval_upper = interval_upper[keep]
        estimate = estimate[keep]
        error = error[keep]

        split = torch.argmax(error, dim=1)
        row = torch.arange(active_z.numel(), device=z.device)
        selected_lower = interval_lower[row, split]
        selected_upper = interval_upper[row, split]
        midpoint = (selected_lower + selected_upper) / 2
        split_lower = torch.stack((selected_lower, midpoint), dim=1)
        split_upper = torch.stack((midpoint, selected_upper), dim=1)
        paired_z = active_z.unsqueeze(1).expand_as(split_lower)
        split_estimate, split_error = _gauss_kronrod_intervals(
            theta_integrand,
            split_lower.reshape(-1),
            split_upper.reshape(-1),
            21,
            paired_z.reshape(-1),
        )
        replacements = (
            split_lower,
            split_upper,
            split_estimate.reshape(-1, 2),
            split_error.reshape(-1, 2),
        )
        interval_lower, interval_upper, estimate, error = _replace_quadrature_splits(
            (interval_lower, interval_upper, estimate, error),
            split,
            replacements,
        )

    return (result * torch.exp(2 * flat_log_z) * 2).reshape(log_z.shape)


def _replace_quadrature_splits(
    values: tuple[torch.Tensor, ...],
    split: torch.Tensor,
    replacements: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Replace one interval per row with its two ordered children."""

    rows, old_count = values[0].shape
    positions = torch.arange(old_count + 1, device=split.device).expand(rows, -1)
    source = torch.where(positions <= split.unsqueeze(1), positions, positions - 1)
    source = source.clamp_max(old_count - 1)
    first = positions == split.unsqueeze(1)
    second = positions == split.unsqueeze(1) + 1
    updated = []
    for value, replacement in zip(values, replacements, strict=True):
        kept = torch.gather(value, 1, source)
        kept = torch.where(first, replacement[:, :1], kept)
        updated.append(torch.where(second, replacement[:, 1:], kept))
    return tuple(updated)


def rapidity_derivative_nlo(
    r: torch.Tensor,
    interpolator_S: LogLogSpline,
    r_grid: torch.Tensor,
    config: BKConfig,
    *,
    regular_interpolator_N: LogLogSpline | None = None,
    sensitive_interpolator_S: LogLogSpline | None = None,
    vegas_state: VegasState | None = None,
    seed: int | None = None,
    integration_bounds: torch.Tensor | None = None,
) -> IntegralResult:
    """Integrate the K2 and Kf derivative with the configured source path."""

    if r.numel() != 1:
        raise ValueError("rapidity_derivative_nlo requires one parent r")

    if config.INTMETHOD_NLO is IntegrationMethod.MULTIPLE:
        minlnr = torch.log(0.5 * r_grid[0])
        maxlnr = torch.log(2 * r_grid[-1])
        return _rapidity_derivative_nlo_multiple(r, interpolator_S, r_grid, config, minlnr, maxlnr)

    bounds = nlo_vegas_bounds(r_grid) if integration_bounds is None else integration_bounds

    def integrand(x: torch.Tensor) -> torch.Tensor:
        log_z = x[:, 0]
        log_z2 = x[:, 1]
        value = _nlo_integrand(
            r,
            torch.exp(log_z),
            x[:, 2],
            torch.exp(log_z2),
            x[:, 3],
            interpolator_S,
            r_grid,
            replace(config, CUDA_FUSION=False),
        )
        return value * torch.exp(2 * log_z + 2 * log_z2)

    cuda_weighted_function = None
    if (
        config.CUDA_FUSION
        and bounds.is_cuda
        and bounds.dtype is torch.float32
        and config.RC_NLO is RunningCouplingNLO.SMALLEST_NLO
        and config.NF <= 3
    ):
        if regular_interpolator_N is None:
            raise ValueError("CUDA BK requires the mixed-precision N/S production splines")
        from nlo_torch.custom_kernels.bk.k2_kf.extension import bk_nlo_mixed_vegas_summaries_cuda

        if sensitive_interpolator_S is None:
            spline = interpolator_S._spline
            sensitive_interpolator_S = LogLogSpline.from_coefficients(
                spline.x.double(),
                spline.a.double(),
                spline.b.double(),
                spline.c.double(),
                spline.d.double(),
            )

        def cuda_weighted_function(
            edges: torch.Tensor,
            bounds_lower: torch.Tensor,
            bounds_width: torch.Tensor,
            bin_index: torch.Tensor,
            random: torch.Tensor,
            volume: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            return bk_nlo_mixed_vegas_summaries_cuda(
                r,
                edges,
                bounds_lower,
                bounds_width,
                bin_index,
                random,
                volume,
                regular_interpolator_N,
                sensitive_interpolator_S,
                r_grid,
                config,
            )

    if config.INTMETHOD_NLO is IntegrationMethod.VEGAS:
        return vegas(
            integrand,
            bounds,
            samples_per_iteration=config.MCINTPOINTS,
            warmup_samples=max(2, config.MCINTPOINTS // 5),
            max_iterations=4,
            min_iterations=2,
            epsrel=0.2,
            seed=seed,
            cuda_mask_fusion=config.CUDA_FUSION,
            cuda_weighted_function=cuda_weighted_function,
            state=vegas_state,
            reuse_warmup_fraction=config.VEGAS_REUSE_WARMUP_FRACTION,
            validate_bounds=False,
        )

    result = miser(
        integrand,
        bounds,
        n_eval=config.MCINTPOINTS,
        epsrel=config.MCINTACCURACY,
        seed=seed,
    )
    if result.converged:
        return result
    return IntegralResult(
        value=torch.zeros_like(result.value),
        error=result.error,
        n_eval=result.n_eval,
        converged=False,
        seed=result.seed,
    )


def nlo_vegas_bounds(r_grid: torch.Tensor) -> torch.Tensor:
    """Construct the shared four-dimensional K2+Kf Vegas bounds."""

    minlnr = torch.log(0.5 * r_grid[0])
    maxlnr = torch.log(2 * r_grid[-1])
    radial = torch.stack((minlnr, maxlnr))
    angle = r_grid.new_tensor((0.0, 2 * math.pi))
    return torch.stack((radial, radial, angle, angle))


def _lo_integrand(
    r: torch.Tensor,
    z: torch.Tensor,
    theta: torch.Tensor,
    interpolator_N: LogLogSpline,
    r_grid: torch.Tensor,
    config: BKConfig,
    rapidity: torch.Tensor | None,
    history_y: torch.Tensor | None,
    history_N: torch.Tensor | None,
) -> torch.Tensor:
    Xsqr = r.square() + z.square() - 2 * r * z * torch.cos(theta)
    invalid = (Xsqr < 1e-40) | (z < 1e-20) | (r < 1e-20)
    X = torch.sqrt(torch.clamp_min(Xsqr, 0))
    Y = z
    safe_X = torch.where(invalid, torch.ones_like(X), X)
    safe_Y = torch.where(invalid, torch.ones_like(Y), Y)

    N_X = torch.minimum(
        _evaluate_loglog(interpolator_N, safe_X, r_grid, 0.0, 1.0),
        torch.ones_like(safe_X),
    )
    N_Y = torch.minimum(
        _evaluate_loglog(interpolator_N, safe_Y, r_grid, 0.0, 1.0),
        torch.ones_like(safe_Y),
    )
    N_r = torch.minimum(
        _evaluate_loglog(interpolator_N, r.expand_as(safe_X), r_grid, 0.0, 1.0),
        torch.ones_like(safe_X),
    )

    if not config.KINEMATICAL_CONSTRAINT:
        dipole = N_X + N_Y - N_r - N_X * N_Y
        value = Kernel_lo(r, z, theta, config) * dipole
        return torch.where(invalid, torch.zeros_like(value), value)

    assert rapidity is not None and history_y is not None and history_N is not None
    delta012 = torch.clamp_min(
        torch.log(torch.minimum(safe_X.square(), safe_Y.square()) / r.square()), 0
    )
    shifted_rapidity = rapidity - delta012
    negative_rapidity = shifted_rapidity < 0
    safe_rapidity = torch.where(
        negative_rapidity, torch.zeros_like(shifted_rapidity), shifted_rapidity
    )
    s02 = 1 - _interpolate_history_N(
        safe_X, safe_rapidity, interpolator_N, r_grid, history_y, history_N
    )
    s12 = 1 - _interpolate_history_N(
        safe_Y, safe_rapidity, interpolator_N, r_grid, history_y, history_N
    )
    s01 = 1 - N_r
    value = Kernel_lo(r, z, theta, config) * (-s02 * s12 + s01)
    return torch.where(invalid | negative_rapidity, torch.zeros_like(value), value)


def _nlo_integrand(
    r: torch.Tensor,
    z: torch.Tensor,
    theta_z: torch.Tensor,
    z2: torch.Tensor,
    theta_z2: torch.Tensor,
    interpolator_S: LogLogSpline,
    r_grid: torch.Tensor,
    config: BKConfig,
) -> torch.Tensor:
    cuda_fusion = config.CUDA_FUSION and z.is_cuda and z.dtype is torch.float32
    if cuda_fusion:
        from nlo_torch.custom_kernels.bk.k2_kf.triton import nlo_geometry_spline_fused

        (
            safe_X,
            safe_Y,
            safe_X2,
            safe_Y2,
            safe_z_m_z2,
            invalid,
            smallest_distance,
            ratio_argument,
            ratio_argument_swap,
            S_X,
            S_Y,
            S_X2,
            S_Y2,
            S_z_m_z2,
        ) = nlo_geometry_spline_fused(interpolator_S, r_grid, r, z, theta_z, z2, theta_z2)
    else:
        Xsqr = r.square() + z.square() - 2 * r * z * torch.cos(theta_z)
        X2sqr = r.square() + z2.square() - 2 * r * z2 * torch.cos(theta_z2)
        z_m_z2sqr = z.square() + z2.square() - 2 * z * z2 * torch.cos(theta_z - theta_z2)
        negative_square = (Xsqr < 0) | (X2sqr < 0) | (z_m_z2sqr < 0)
        X = torch.sqrt(torch.clamp_min(Xsqr, 0))
        Y = z
        X2 = torch.sqrt(torch.clamp_min(X2sqr, 0))
        Y2 = z2
        z_m_z2 = torch.sqrt(torch.clamp_min(z_m_z2sqr, 0))
        invalid = negative_square | (
            (X < 1e-20) | (Y < 1e-20) | (X2 < 1e-20) | (Y2 < 1e-20) | (z_m_z2 < 1e-20)
        )
        safe_X = torch.where(invalid, torch.ones_like(X), X)
        safe_Y = torch.where(invalid, torch.ones_like(Y), Y)
        safe_X2 = torch.where(invalid, torch.ones_like(X2), X2)
        safe_Y2 = torch.where(invalid, torch.ones_like(Y2), Y2)
        safe_z_m_z2 = torch.where(invalid, torch.ones_like(z_m_z2), z_m_z2)

    if not cuda_fusion:
        safe_distances = torch.stack((safe_X, safe_Y, safe_X2, safe_Y2, safe_z_m_z2))
        S_X, S_Y, S_X2, S_Y2, S_z_m_z2 = _evaluate_loglog(
            interpolator_S, safe_distances, r_grid, 1.0, 0.0
        ).unbind()

    if cuda_fusion and config.RC_NLO is RunningCouplingNLO.SMALLEST_NLO:
        if config.NF <= 3:
            alpha_s = None
        else:
            alpha_s = _alpha_s(smallest_distance, config)
    else:
        alpha_s = _nlo_alpha_s(r, safe_X, safe_Y, safe_X2, safe_Y2, safe_z_m_z2, config)
    kernel_f = None
    kernel_f_swap = None
    if cuda_fusion:
        from nlo_torch.custom_kernels.bk.k2_kf.triton import nlo_integrand_fused

        return nlo_integrand_fused(
            r,
            safe_X,
            safe_Y,
            safe_X2,
            safe_Y2,
            safe_z_m_z2,
            S_X,
            S_Y,
            S_X2,
            S_Y2,
            S_z_m_z2,
            alpha_s,
            smallest_distance,
            invalid,
            ratio_argument,
            ratio_argument_swap,
            config,
        )

    X_pair = torch.stack((safe_X, safe_X2))
    Y_pair = torch.stack((safe_Y, safe_Y2))
    X2_pair = X_pair.flip(0)
    Y2_pair = Y_pair.flip(0)
    z_m_z2_pair = safe_z_m_z2.expand_as(X_pair)
    k, kswap = Kernel_nlo(r, X_pair, Y_pair, X2_pair, Y2_pair, z_m_z2_pair).unbind()
    if config.NF > 0:
        kernel_f, kernel_f_swap = Kernel_nlo_fermion(
            r, X_pair, Y_pair, X2_pair, Y2_pair, z_m_z2_pair, config
        ).unbind()
    dipole = -(S_X * S_z_m_z2 * S_Y2 - S_X * S_Y)
    dipole_swap = -(S_X2 * S_z_m_z2 * S_Y - S_X2 * S_Y2)
    cut = (torch.abs(k) > 1e10) & (torch.abs(dipole) < 1e-10)
    cut_swap = (torch.abs(kswap) > 1e10) & (torch.abs(dipole_swap) < 1e-10)
    k = torch.where(cut, torch.zeros_like(k), k)
    dipole = torch.where(cut, torch.zeros_like(dipole), dipole)
    kswap = torch.where(cut_swap, torch.zeros_like(kswap), kswap)
    dipole_swap = torch.where(cut_swap, torch.zeros_like(dipole_swap), dipole_swap)

    if config.SYMMETRIZE_Z_Z2_INTEGRATION:
        result = (k * dipole + kswap * dipole_swap) / 2
    else:
        result = k * dipole

    if config.NF > 0:
        assert kernel_f is not None and kernel_f_swap is not None
        dipole_f = S_Y * (S_X2 - S_X)
        dipole_f_swap = S_Y2 * (S_X - S_X2)
        if config.SYMMETRIZE_Z_Z2_INTEGRATION:
            result = result - (kernel_f * dipole_f + kernel_f_swap * dipole_f_swap) / 2
        else:
            result = result - kernel_f * dipole_f

    result = result * (alpha_s * config.NC).square() / (8 * math.pi**4)
    valid = ~invalid & torch.isfinite(result)
    return torch.where(valid, result, torch.zeros_like(result))


def _nlo_alpha_s(
    r: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    X2: torch.Tensor,
    Y2: torch.Tensor,
    z_m_z2: torch.Tensor,
    config: BKConfig,
) -> torch.Tensor:
    if config.RC_NLO is RunningCouplingNLO.FIXED_NLO:
        return torch.full_like(X, config.FIXED_AS)
    if config.RC_NLO is RunningCouplingNLO.PARENT_NLO:
        return _alpha_s(r.expand_as(X), config)
    min_size = torch.minimum(
        torch.minimum(torch.minimum(r.expand_as(X), X), Y),
        torch.minimum(torch.minimum(X2, Y2), z_m_z2),
    )
    return _alpha_s(min_size, config)


def _rapidity_derivative_nlo_multiple(
    r: torch.Tensor,
    interpolator_S: LogLogSpline,
    r_grid: torch.Tensor,
    config: BKConfig,
    minlnr: torch.Tensor,
    maxlnr: torch.Tensor,
) -> IntegralResult:
    def integrate_theta_z(log_z: torch.Tensor) -> torch.Tensor:
        values_z = []
        for log_z_value in log_z.reshape(-1):
            z = torch.exp(log_z_value)

            def integrate_z2(theta_z: torch.Tensor) -> torch.Tensor:
                values_theta = []
                for theta_z_value in theta_z.reshape(-1):

                    def integrate_theta_z2(log_z2: torch.Tensor) -> torch.Tensor:
                        values_z2 = []
                        for log_z2_value in log_z2.reshape(-1):
                            z2 = torch.exp(log_z2_value)

                            def theta_z2_integrand(theta_z2: torch.Tensor) -> torch.Tensor:
                                return _nlo_integrand(
                                    r,
                                    z,
                                    theta_z_value,
                                    z2,
                                    theta_z2,
                                    interpolator_S,
                                    r_grid,
                                    config,
                                )

                            theta_z2_result = adaptive_gauss_kronrod_1d(
                                theta_z2_integrand,
                                r.new_tensor(0.0),
                                r.new_tensor(2 * math.pi),
                                epsrel=config.INTACCURACY,
                                max_intervals=config.THETAINTPOINTS,
                                rule=15,
                            )
                            values_z2.append(theta_z2_result.value * torch.exp(2 * log_z2_value))
                        return torch.stack(values_z2).reshape(log_z2.shape)

                    z2_result = adaptive_gauss_kronrod_1d(
                        integrate_theta_z2,
                        minlnr,
                        maxlnr,
                        epsrel=config.INTACCURACY,
                        max_intervals=config.RINTPOINTS,
                        rule=15,
                    )
                    values_theta.append(z2_result.value)
                return torch.stack(values_theta).reshape(theta_z.shape)

            theta_z_result = adaptive_gauss_kronrod_1d(
                integrate_z2,
                r.new_tensor(0.0),
                r.new_tensor(2 * math.pi),
                epsrel=config.INTACCURACY,
                max_intervals=config.THETAINTPOINTS,
                rule=15,
            )
            values_z.append(theta_z_result.value * torch.exp(2 * log_z_value))
        return torch.stack(values_z).reshape(log_z.shape)

    return adaptive_gauss_kronrod_1d(
        integrate_theta_z,
        minlnr,
        maxlnr,
        epsrel=config.INTACCURACY,
        max_intervals=config.RINTPOINTS,
        rule=15,
    )


def _interpolate_history_N(
    r: torch.Tensor,
    rapidity: torch.Tensor,
    interpolator_N: LogLogSpline,
    r_grid: torch.Tensor,
    history_y: torch.Tensor,
    history_N: torch.Tensor,
) -> torch.Tensor:
    rapidity = torch.clamp_min(rapidity, 0)
    if bool((rapidity > history_y[-1]).any()):
        raise ValueError("shifted rapidity is above the available BK history")

    yind = torch.searchsorted(history_y, rapidity, right=True).sub(1).clamp_min(0)
    latest = yind == history_y.numel() - 1
    rind = torch.searchsorted(r_grid, r, right=True).sub(1)
    start = torch.clamp_min(rind - 3, 0)
    too_large = start + 6 > r_grid.numel()
    result = torch.empty_like(r)

    if bool(latest.any()):
        result[latest] = _evaluate_loglog(interpolator_N, r[latest], r_grid, 0.0, 1.0)
    high = ~latest & too_large
    if bool(high.any()):
        result[high] = 1
    regular = ~latest & ~too_large
    if bool(regular.any()):
        regular_start = start[regular]
        rind_nodes = regular_start.unsqueeze(-1) + torch.arange(
            6, dtype=regular_start.dtype, device=regular_start.device
        )
        r_nodes = r_grid[rind_nodes]
        lower_N = history_N[yind[regular].unsqueeze(-1), rind_nodes]
        upper_N = history_N[(yind[regular] + 1).unsqueeze(-1), rind_nodes]
        evaluation_r = r[regular]
        evaluation_r = torch.where(
            evaluation_r < r_nodes[:, 0], r_nodes[:, 0] * 1.00001, evaluation_r
        )
        evaluation_r = torch.where(
            evaluation_r > r_nodes[:, -1], r_nodes[:, -1] * 0.999999, evaluation_r
        )
        N_lower = NaturalCubicSpline(r_nodes, lower_N)(evaluation_r)
        N_upper = NaturalCubicSpline(r_nodes, upper_N)(evaluation_r)
        fraction = (rapidity[regular] - history_y[yind[regular]]) / (
            history_y[yind[regular] + 1] - history_y[yind[regular]]
        )
        result[regular] = N_lower + fraction * (N_upper - N_lower)
    return result.clamp(0, 1)


def _evaluate_loglog(
    interpolator: LogLogSpline,
    r: torch.Tensor,
    r_grid: torch.Tensor,
    underflow: float,
    overflow: float,
) -> torch.Tensor:
    evaluation_r = r.clamp(r_grid[0], r_grid[-1])
    value = interpolator(evaluation_r)
    value = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
    value = torch.where(r < r_grid[0], torch.full_like(value, underflow), value)
    return torch.where(r > r_grid[-1], torch.full_like(value, overflow), value)


def _alpha_s(r: torch.Tensor, config: BKConfig) -> torch.Tensor:
    fixed = (
        config.RC_LO is RunningCouplingLO.FIXED_LO or config.RC_NLO is RunningCouplingNLO.FIXED_NLO
    )
    return bk_alpha_s(
        r,
        NC=config.NC,
        NF=config.NF,
        LambdaQCD=config.LambdaQCD,
        C2=config.C2,
        fixed=fixed,
        fixed_alpha_s=config.FIXED_AS,
    )


__all__ = [
    "BatchedIntegralResult",
    "nlo_vegas_bounds",
    "rapidity_derivative_lo",
    "rapidity_derivative_lo_batch",
    "rapidity_derivative_nlo",
]
