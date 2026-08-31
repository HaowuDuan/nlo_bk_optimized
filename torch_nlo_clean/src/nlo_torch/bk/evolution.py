"""BK radial grid, rapidity stepping, and amplitude history."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import torch

from nlo_torch.bk.config import BKConfig, IntegrationMethod
from nlo_torch.bk.derivatives import (
    _fixed_k1_supported,
    _persistent_k1_supported,
    nlo_vegas_bounds,
    rapidity_derivative_lo,
    rapidity_derivative_lo_batch,
    rapidity_derivative_nlo,
)
from nlo_torch.bk.initial_conditions import MV, ICDataFile
from nlo_torch.dipole.table import DipoleTable
from nlo_torch.numerics.integration import VegasState
from nlo_torch.numerics.interpolation import LogLogSpline


@dataclass(frozen=True, slots=True)
class DNDYResult:
    r: torch.Tensor
    K1: torch.Tensor
    K2_Kf: torch.Tensor
    N: torch.Tensor


def solve_bk(
    initial_condition: MV | ICDataFile,
    maxy: float,
    config: BKConfig = BKConfig(),
    *,
    device: torch.device | str = "mps",
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
) -> DipoleTable:
    """Evolve through full output steps and return the final in-memory table."""

    if isinstance(initial_condition, ICDataFile):
        device = initial_condition.r.device
        dtype = initial_condition.r.dtype
    config = _production_config(config, device, dtype)
    r = _r_grid(config, device=device, dtype=dtype)
    N = initial_condition.dipole_amplitude(r)
    table = DipoleTable(
        r=r,
        y=torch.zeros(1, dtype=dtype, device=device),
        N=N.unsqueeze(0),
        x0=initial_condition.x0,
    )
    for table in _bk_steps(table, maxy, config, seed=seed):
        pass
    return table


def compute_dndy(
    initial_condition: MV | ICDataFile,
    config: BKConfig = BKConfig(),
    *,
    device: torch.device | str = "mps",
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
) -> DNDYResult:
    """Return the source diagnostic columns at the initial rapidity."""

    if isinstance(initial_condition, ICDataFile):
        device = initial_condition.r.device
        dtype = initial_condition.r.dtype
    config = _production_config(config, device, dtype)
    r = _r_grid(config, device=device, dtype=dtype)
    N = initial_condition.dipole_amplitude(r)
    y = torch.zeros(1, dtype=dtype, device=device)
    history_N = N.unsqueeze(0)
    K1, K2_Kf = _derivative_components(y[0], N, r, y, history_N, config, seed=seed)
    keep = torch.arange(r.numel(), device=r.device) > 0
    keep = keep & (N <= 0.99999)
    return DNDYResult(r=r[keep], K1=K1[keep], K2_Kf=K2_Kf[keep], N=N[keep])


def _bk_steps(
    table: DipoleTable,
    maxy: float,
    config: BKConfig,
    *,
    seed: int | None,
) -> Iterator[DipoleTable]:
    r = table.r
    history_y = table.y
    history_N = table.N
    y = float(history_y[-1].item())
    N = history_N[-1]
    h = config.DE_SOLVER_STEP
    vegas_states = (
        {}
        if config.VEGAS_REUSE_GRID
        and config.CUDA_FUSION
        and r.is_cuda
        and r.dtype is torch.float32
        and config.INTMETHOD_NLO is IntegrationMethod.VEGAS
        else None
    )

    while True:
        nexty = y + config.DE_SOLVER_STEP
        if config.EULER_METHOD:
            dNdy = _evolve_derivative(
                N,
                r,
                r.new_tensor(y),
                history_y,
                history_N,
                config,
                seed=seed,
                vegas_states=vegas_states,
            )
            N = N + config.DE_SOLVER_STEP * dNdy
            y = nexty
        elif _fixed_rk23_supported(r, config):
            N = _fixed_rk23_step(
                N,
                y,
                config.DE_SOLVER_STEP,
                r,
                history_y,
                history_N,
                config,
                seed,
                vegas_states=vegas_states,
            )
            y = nexty
        else:
            N, y, h = _adaptive_rk2_to(
                N,
                y,
                nexty,
                h,
                r,
                history_y,
                history_N,
                config,
                seed,
                vegas_states=vegas_states,
            )

        if config.FORCE_POSITIVE_N:
            N = N.clamp(0, 1)
        if not bool(torch.isfinite(N).all()):
            raise FloatingPointError(f"non-finite BK amplitude after rapidity step y={y}")
        history_y = torch.cat((history_y, r.new_tensor([y])))
        history_N = torch.cat((history_N, N.unsqueeze(0)), dim=0)
        yield DipoleTable(r=r, y=history_y, N=history_N, x0=table.x0)
        rapidity_tolerance = 1e-12 * max(1.0, abs(maxy), abs(config.DE_SOLVER_STEP))
        if y >= maxy - rapidity_tolerance:
            break


def _production_config(
    config: BKConfig,
    device: torch.device | str,
    dtype: torch.dtype,
) -> BKConfig:
    """Select the fastest validated production implementation automatically."""

    selected_device = torch.device(device)
    if selected_device.type == "cuda" and dtype is torch.float32:
        return replace(
            config,
            CUDA_FUSION=True,
            K1_FIXED=True,
            K1_FIXED_REFINE=False,
        )
    return config


def _fixed_rk23_supported(r: torch.Tensor, config: BKConfig) -> bool:
    return (
        config.CUDA_FUSION
        and r.is_cuda
        and r.dtype is torch.float32
        and config.INTMETHOD_NLO is IntegrationMethod.VEGAS
        and _fixed_k1_supported(config)
    )


def _fixed_rk23_step(
    N: torch.Tensor,
    y: float,
    step: float,
    r: torch.Tensor,
    history_y: torch.Tensor,
    history_N: torch.Tensor,
    config: BKConfig,
    seed: int | None,
    *,
    vegas_states: dict[int, VegasState] | None = None,
) -> torch.Tensor:
    """Apply one accepted GSL RK2(3) step using exactly three derivatives."""

    K1 = _evolve_derivative(
        N,
        r,
        r.new_tensor(y),
        history_y,
        history_N,
        config,
        seed=seed,
        vegas_states=vegas_states,
    )
    midpoint_N = N + 0.5 * step * K1
    K2 = _evolve_derivative(
        midpoint_N,
        r,
        r.new_tensor(y + 0.5 * step),
        history_y,
        history_N,
        config,
        seed=seed,
        vegas_states=vegas_states,
    )
    endpoint_N = N + step * (-K1 + 2.0 * K2)
    K3 = _evolve_derivative(
        endpoint_N,
        r,
        r.new_tensor(y + step),
        history_y,
        history_N,
        config,
        seed=seed,
        vegas_states=vegas_states,
    )
    return N + step * (K1 + 4.0 * K2 + K3) / 6.0


def _adaptive_rk2_to(
    N: torch.Tensor,
    y: float,
    nexty: float,
    h: float,
    r: torch.Tensor,
    history_y: torch.Tensor,
    history_N: torch.Tensor,
    config: BKConfig,
    seed: int | None,
    *,
    vegas_states: dict[int, VegasState] | None = None,
) -> tuple[torch.Tensor, float, float]:
    while y < nexty:
        step = min(h, nexty - y)
        y_tensor = r.new_tensor(y)
        K1 = _evolve_derivative(
            N,
            r,
            y_tensor,
            history_y,
            history_N,
            config,
            seed=seed,
            vegas_states=vegas_states,
        )
        midpoint_N = N + 0.5 * step * K1
        K2 = _evolve_derivative(
            midpoint_N,
            r,
            r.new_tensor(y + 0.5 * step),
            history_y,
            history_N,
            config,
            seed=seed,
            vegas_states=vegas_states,
        )
        full_step = N + step * K2

        half_step = step / 2
        first_half_midpoint = N + 0.5 * half_step * K1
        first_half_K2 = _evolve_derivative(
            first_half_midpoint,
            r,
            r.new_tensor(y + 0.5 * half_step),
            history_y,
            history_N,
            config,
            seed=seed,
            vegas_states=vegas_states,
        )
        half_N = N + half_step * first_half_K2
        second_half_K1 = _evolve_derivative(
            half_N,
            r,
            r.new_tensor(y + half_step),
            history_y,
            history_N,
            config,
            seed=seed,
            vegas_states=vegas_states,
        )
        second_half_midpoint = half_N + 0.5 * half_step * second_half_K1
        second_half_K2 = _evolve_derivative(
            second_half_midpoint,
            r,
            r.new_tensor(y + half_step + 0.5 * half_step),
            history_y,
            history_N,
            config,
            seed=seed,
            vegas_states=vegas_states,
        )
        two_half_steps = half_N + half_step * second_half_K2

        error = torch.abs(two_half_steps - full_step) / 3
        tolerance = config.DE_SOLVER_ABSERR + config.DE_SOLVER_RELERR * torch.abs(two_half_steps)
        error_ratio = float(torch.max(error / tolerance).item())
        if error_ratio <= 1:
            N = two_half_steps
            y += step
            factor = 5.0 if error_ratio == 0 else min(5.0, 0.9 * error_ratio ** (-1 / 3))
            h = step * factor
        else:
            h = step * max(0.1, 0.9 * error_ratio ** (-1 / 3))
            if h <= torch.finfo(r.dtype).eps * max(1.0, abs(y)):
                raise RuntimeError(f"BK RK2 step underflow at y={y}")
    return N, y, h


def _evolve_derivative(
    N: torch.Tensor,
    r: torch.Tensor,
    rapidity: torch.Tensor,
    history_y: torch.Tensor,
    history_N: torch.Tensor,
    config: BKConfig,
    *,
    seed: int | None,
    vegas_states: dict[int, VegasState] | None = None,
) -> torch.Tensor:
    K1, K2_Kf = _derivative_components(
        rapidity,
        N,
        r,
        history_y,
        history_N,
        config,
        seed=seed,
        vegas_states=vegas_states,
    )
    dNdy = K1 + K2_Kf
    return torch.where(torch.isfinite(dNdy), dNdy, torch.zeros_like(dNdy))


def _derivative_components(
    rapidity: torch.Tensor,
    N: torch.Tensor,
    r: torch.Tensor,
    history_y: torch.Tensor,
    history_N: torch.Tensor,
    config: BKConfig,
    *,
    seed: int | None,
    vegas_states: dict[int, VegasState] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    (
        interpolator_N,
        interpolator_S,
        sensitive_interpolator_N,
        sensitive_interpolator_S,
    ) = _mixed_evolution_interpolators(N, r, config)
    K2_Kf = torch.zeros_like(N)
    integration_bounds = (
        nlo_vegas_bounds(r)
        if config.Order.has_nlo_kernels and config.INTMETHOD_NLO is IntegrationMethod.VEGAS
        else None
    )
    use_batched_k1 = (
        config.CUDA_FUSION
        and r.is_cuda
        and r.dtype is torch.float32
        and _persistent_k1_supported(config)
    )
    if use_batched_k1:
        K1, active_index = _batched_k1_component(
            N,
            r,
            interpolator_N,
            config,
            sensitive_interpolator_N=sensitive_interpolator_N,
        )
        if config.Order.has_nlo_kernels:
            active_parents = active_index.detach().cpu().tolist()
            for rind in active_parents:
                nlo = rapidity_derivative_nlo(
                    r[rind],
                    interpolator_S,
                    r,
                    config,
                    regular_interpolator_N=interpolator_N,
                    sensitive_interpolator_S=sensitive_interpolator_S,
                    vegas_state=(
                        None
                        if vegas_states is None
                        else vegas_states.setdefault(rind, VegasState())
                    ),
                    seed=seed,
                    integration_bounds=integration_bounds,
                )
                K2_Kf[rind] = nlo.value

            if vegas_states is not None and seed is not None:
                ready = [
                    rind
                    for rind in active_parents
                    if vegas_states[rind].last_components is not None
                ]
                if ready:
                    components = torch.stack(
                        tuple(vegas_states[rind].last_components for rind in ready)
                    )
                    k1 = K1[ready].double()
                    scale = k1.abs() + components[:, 0].abs() + components[:, 1].abs()
                    net = k1 + components[:, 0] + components[:, 1]
                    condition = scale / torch.maximum(net.abs(), 0.05 * scale)
                    selected = [
                        rind
                        for rind, value in zip(ready, condition.tolist(), strict=True)
                        if value >= 4.0 or float(N[rind]) > 0.9
                    ]
                    for rind in selected:
                        values = [K2_Kf[rind]]
                        replica_components = [vegas_states[rind].last_components]
                        errors = [vegas_states[rind].last_error]
                        for replica in range(1, 7):
                            replica_state = VegasState()
                            estimate = rapidity_derivative_nlo(
                                r[rind],
                                interpolator_S,
                                r,
                                config,
                                regular_interpolator_N=interpolator_N,
                                sensitive_interpolator_S=sensitive_interpolator_S,
                                vegas_state=replica_state,
                                seed=seed + 1_000 * replica,
                                integration_bounds=integration_bounds,
                            )
                            values.append(estimate.value)
                            replica_components.append(replica_state.last_components)
                            errors.append(replica_state.last_error)
                        K2_Kf[rind] = torch.stack(values).mean()
                        vegas_states[rind].last_components = torch.stack(
                            tuple(replica_components)
                        ).mean(0)
                        vegas_states[rind].last_error = (
                            sum(error * error for error in errors) ** 0.5 / len(errors)
                        )
        return K1, K2_Kf

    K1 = torch.zeros_like(N)
    for rind in range(1, r.numel()):
        if bool(N[rind] > 0.99999):
            continue
        lo = rapidity_derivative_lo(
            r[rind],
            interpolator_N,
            r,
            config,
            rapidity=rapidity,
            history_y=history_y,
            history_N=history_N,
        )
        K1[rind] = lo.value
        if config.Order.has_nlo_kernels:
            nlo = rapidity_derivative_nlo(
                r[rind],
                interpolator_S,
                r,
                config,
                regular_interpolator_N=interpolator_N,
                sensitive_interpolator_S=sensitive_interpolator_S,
                vegas_state=(
                    None if vegas_states is None else vegas_states.setdefault(rind, VegasState())
                ),
                seed=seed,
                integration_bounds=integration_bounds,
            )
            K2_Kf[rind] = nlo.value
    return K1, K2_Kf


def _batched_k1_component(
    N: torch.Tensor,
    r: torch.Tensor,
    interpolator_N: LogLogSpline,
    config: BKConfig,
    *,
    sensitive_interpolator_N: LogLogSpline | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the production K1 radial component with shared CUDA launches."""

    active_index = torch.nonzero(N[1:] <= 0.99999, as_tuple=False).flatten() + 1
    K1 = torch.zeros_like(N)
    if active_index.numel() > 0:
        parent_r = r[active_index]
        if config.K1_FIXED and _fixed_k1_supported(config):
            if sensitive_interpolator_N is None:
                from nlo_torch.custom_kernels.bk.k1.extension import k1_fixed_grid_integrals_cuda

                _, fine = k1_fixed_grid_integrals_cuda(
                    parent_r,
                    interpolator_N,
                    r,
                    config,
                    refine=config.K1_FIXED_REFINE,
                )
                K1[active_index] = fine.to(dtype=K1.dtype)
            else:
                from nlo_torch.custom_kernels.bk.k1.extension import (
                    k1_mixed_fixed_grid_integrals_cuda,
                )

                K1[active_index] = k1_mixed_fixed_grid_integrals_cuda(
                    parent_r,
                    active_index,
                    interpolator_N,
                    sensitive_interpolator_N,
                    r,
                    config,
                )
        else:
            lo = rapidity_derivative_lo_batch(parent_r, interpolator_N, r, config)
            K1[active_index] = lo.value
    return K1, active_index


def _evolution_interpolators(
    N: torch.Tensor, r: torch.Tensor, config: BKConfig
) -> tuple[LogLogSpline, LogLogSpline]:
    if config.CUDA_FUSION and N.is_cuda and N.dtype is torch.float32:
        from nlo_torch.custom_kernels.interpolation import evolution_loglog_splines_cuda

        return evolution_loglog_splines_cuda(
            r,
            N,
            force_positive=config.FORCE_POSITIVE_N,
        )

    interpolation_N = torch.where(N > 1, torch.ones_like(N), N)
    if config.FORCE_POSITIVE_N:
        interpolation_N = torch.where(
            interpolation_N < 0, torch.zeros_like(interpolation_N), interpolation_N
        )

    S = 1 - N
    interpolation_S = torch.where(S < 0, torch.zeros_like(S), S)
    if config.FORCE_POSITIVE_N:
        interpolation_S = torch.where(
            interpolation_S > 1, torch.ones_like(interpolation_S), interpolation_S
        )
    return LogLogSpline(r, interpolation_N), LogLogSpline(r, interpolation_S)


def _mixed_evolution_interpolators(
    N: torch.Tensor,
    r: torch.Tensor,
    config: BKConfig,
) -> tuple[LogLogSpline, LogLogSpline, LogLogSpline | None, LogLogSpline | None]:
    """Build both precision branches once for one production derivative."""

    if config.CUDA_FUSION and N.is_cuda and N.dtype is torch.float32:
        from nlo_torch.custom_kernels.interpolation import (
            evolution_loglog_splines_mixed_cuda,
        )

        return evolution_loglog_splines_mixed_cuda(
            r,
            N,
            force_positive=config.FORCE_POSITIVE_N,
        )
    interpolator_n, interpolator_s = _evolution_interpolators(N, r, config)
    return interpolator_n, interpolator_s, None, None


def _r_grid(
    config: BKConfig,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    rind = torch.arange(config.RPOINTS + 1, dtype=dtype, device=device)
    step = (config.MAXR / config.MINR) ** (1 / config.RPOINTS)
    return config.MINR * torch.pow(rind.new_tensor(step), rind)


__all__ = ["DNDYResult", "compute_dndy", "solve_bk"]
