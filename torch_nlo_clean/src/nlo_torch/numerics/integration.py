"""Deterministic and Monte Carlo integration used by BK and DIS."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class IntegralResult:
    value: torch.Tensor
    error: torch.Tensor
    n_eval: int
    converged: bool
    seed: int | None


VegasWeightedFunction = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    torch.Tensor | tuple[torch.Tensor, ...],
]


def adaptive_gauss_kronrod_1d(
    function: Callable[[torch.Tensor], torch.Tensor],
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    epsrel: float = 1e-3,
    epsabs: float = 0.0,
    max_intervals: int = 85,
    rule: int = 21,
) -> IntegralResult:
    """Adaptive scalar integration with the Gauss–Kronrod 15 or 21 rule."""

    if lower.numel() != 1 or upper.numel() != 1:
        raise ValueError("lower and upper must be scalar tensors")
    if lower.device != upper.device or lower.dtype != upper.dtype:
        raise ValueError("lower and upper must have the same device and dtype")
    if not bool(lower < upper):
        raise ValueError("lower must be smaller than upper")
    if rule not in {15, 21}:
        raise ValueError("rule must be 15 or 21")
    if max_intervals < 1:
        raise ValueError("max_intervals must be positive")

    interval_lower = lower.reshape(1)
    interval_upper = upper.reshape(1)
    estimate, error = _gauss_kronrod_intervals(function, interval_lower, interval_upper, rule)
    nodes = rule

    while interval_lower.numel() < max_intervals:
        value = estimate.sum()
        total_error = error.sum()
        tolerance = max(epsabs, epsrel * abs(float(value.item())))
        if float(total_error.item()) <= tolerance:
            return IntegralResult(value, total_error, nodes, True, None)

        split = int(torch.argmax(error).item())
        midpoint = (interval_lower[split] + interval_upper[split]) / 2
        new_lower = torch.stack((interval_lower[split], midpoint))
        new_upper = torch.stack((midpoint, interval_upper[split]))
        new_estimate, new_error = _gauss_kronrod_intervals(function, new_lower, new_upper, rule)
        interval_lower = torch.cat((interval_lower[:split], new_lower, interval_lower[split + 1 :]))
        interval_upper = torch.cat((interval_upper[:split], new_upper, interval_upper[split + 1 :]))
        estimate = torch.cat((estimate[:split], new_estimate, estimate[split + 1 :]))
        error = torch.cat((error[:split], new_error, error[split + 1 :]))
        nodes += 2 * rule

    value = estimate.sum()
    total_error = error.sum()
    tolerance = max(epsabs, epsrel * abs(float(value.item())))
    return IntegralResult(value, total_error, nodes, float(total_error.item()) <= tolerance, None)


def tensor_gauss_legendre(
    function: Callable[[torch.Tensor], torch.Tensor],
    bounds: torch.Tensor,
    *,
    points: int = 24,
    epsrel: float = 1e-6,
    epsabs: float = 0.0,
    batch_size: int = 65_536,
) -> IntegralResult:
    """Tensor-product Gauss–Legendre integration with a half-order error check."""

    _validate_bounds(bounds)
    if points < 4:
        raise ValueError("points must be at least four")

    value = _tensor_gauss_legendre_once(function, bounds, points, batch_size)
    coarse_points = max(2, points // 2)
    coarse_value = _tensor_gauss_legendre_once(function, bounds, coarse_points, batch_size)
    error = torch.abs(value - coarse_value)
    tolerance = max(epsabs, epsrel * abs(float(value.item())))
    dimensions = bounds.shape[0]
    n_eval = points**dimensions + coarse_points**dimensions
    return IntegralResult(value, error, n_eval, float(error.item()) <= tolerance, None)


def vegas(
    function: Callable[[torch.Tensor], torch.Tensor],
    bounds: torch.Tensor,
    *,
    samples_per_iteration: int,
    max_iterations: int = 8,
    min_iterations: int = 3,
    warmup_samples: int = 0,
    bins: int = 32,
    batch_size: int = 65_536,
    epsrel: float = 0.2,
    epsabs: float = 0.0,
    seed: int | None = None,
    cuda_mask_fusion: bool = False,
    cuda_weighted_function: VegasWeightedFunction | None = None,
    validate_bounds: bool = True,
) -> IntegralResult:
    """Adaptive, importance-sampled Vegas with tensorized sample batches."""

    if validate_bounds:
        _validate_bounds(bounds)
    elif bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] < 1:
        raise ValueError("bounds must have shape (dimensions, 2)")
    elif not bounds.is_floating_point():
        raise TypeError("bounds must be floating point")
    if samples_per_iteration < 2 or max_iterations < 1:
        raise ValueError("Vegas requires at least two samples and one iteration")
    if min_iterations < 1 or min_iterations > max_iterations:
        raise ValueError("min_iterations must lie between one and max_iterations")
    if warmup_samples == 1 or warmup_samples < 0:
        raise ValueError("warmup_samples must be zero or at least two")
    if bins < 2 or batch_size < 1:
        raise ValueError("bins and batch_size must be positive")

    seed = torch.seed() if seed is None else seed
    generator = torch.Generator(device=bounds.device)
    generator.manual_seed(seed)
    dimensions = bounds.shape[0]
    cuda_state_initialization = (
        cuda_mask_fusion
        and bounds.is_cuda
        and bounds.dtype is torch.float32
        and dimensions == 4
        and bins == 32
    )
    if cuda_state_initialization:
        from nlo_torch.custom_kernels.integration import vegas_initialize_cuda

        edges, bounds_lower, bounds_width, volume, histogram_semaphores = vegas_initialize_cuda(
            bounds
        )
    else:
        edges = (
            torch.linspace(0, 1, bins + 1, dtype=bounds.dtype, device=bounds.device)
            .expand(dimensions, -1)
            .clone()
        )
        bounds_lower = bounds[:, 0].contiguous()
        bounds_width = bounds[:, 1] - bounds[:, 0]
        volume = torch.prod(bounds_width)
        histogram_semaphores = None

    cuda_kernel_accumulators = cuda_state_initialization and batch_size <= 65_536
    estimates: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    cuda_buffered_summaries = (
        cuda_mask_fusion
        and bounds.is_cuda
        and bounds.dtype is torch.float32
        and dimensions == 4
        and bins == 32
        and max_iterations <= 4
    )
    estimate_buffer = bounds.new_empty(max_iterations) if cuda_buffered_summaries else None
    variance_buffer = bounds.new_empty(max_iterations) if cuda_buffered_summaries else None
    reported_iterations = 0
    n_eval = 0
    converged = False

    iterations = max_iterations + int(warmup_samples > 0)
    for iteration_index in range(iterations):
        warmup = warmup_samples > 0 and iteration_index == 0
        iteration_samples = warmup_samples if warmup else samples_per_iteration
        if cuda_kernel_accumulators:
            total = bounds.new_empty(())
            total_square = bounds.new_empty(())
            histogram = bounds.new_empty((dimensions, bins))
            bin_count = bounds.new_empty((dimensions, bins))
        else:
            total = bounds.new_zeros(())
            total_square = bounds.new_zeros(())
            histogram = bounds.new_zeros((dimensions, bins))
            bin_count = bounds.new_zeros((dimensions, bins))
        sensitive_total = bounds.new_zeros((), dtype=torch.float64)
        sensitive_total_square = bounds.new_zeros((), dtype=torch.float64)
        mixed_precision_iteration = False
        completed = 0

        while completed < iteration_samples:
            count = min(batch_size, iteration_samples - completed)
            cuda_random_fusion = cuda_state_initialization and count <= 65_536
            if cuda_random_fusion:
                from nlo_torch.custom_kernels.integration import vegas_random_cuda

                bin_index, random = vegas_random_cuda(count, bounds, generator)
            else:
                bin_index = torch.randint(
                    bins,
                    (count, dimensions),
                    device=bounds.device,
                    generator=generator,
                )
                random = torch.rand(
                    (count, dimensions),
                    dtype=bounds.dtype,
                    device=bounds.device,
                    generator=generator,
                )
            direct_cuda_weights = (
                cuda_weighted_function is not None
                and cuda_mask_fusion
                and bounds.is_cuda
                and bounds.dtype is torch.float32
                and dimensions == 4
                and bins == 32
                and count <= 65_536
            )
            if direct_cuda_weights:
                weighted_output = cuda_weighted_function(
                    edges,
                    bounds_lower,
                    bounds_width,
                    bin_index,
                    random,
                    volume,
                )
                mixed_precision_batch = isinstance(weighted_output, tuple)
                if mixed_precision_batch:
                    if completed > 0 and not mixed_precision_iteration:
                        raise ValueError("a weighted Vegas function cannot change precision modes")
                    mixed_precision_iteration = True
                    if completed == 0 and cuda_kernel_accumulators:
                        total.zero_()
                        total_square.zero_()
                    if len(weighted_output) == 5:
                        (
                            absolute_weight,
                            regular_sum,
                            regular_square,
                            sensitive_sum,
                            sensitive_square,
                        ) = weighted_output
                        if absolute_weight.shape != (count,):
                            raise ValueError("mixed Vegas summaries require one histogram weight")
                        if regular_sum.numel() != 1 or regular_square.numel() != 1:
                            raise ValueError("regular Vegas summaries must be scalar")
                        if sensitive_sum.shape != regular_sum.shape or sensitive_square.shape != (
                            regular_sum.shape
                        ):
                            raise ValueError(
                                "sensitive Vegas summaries must match the regular ones"
                            )
                        if (
                            absolute_weight.dtype is not bounds.dtype
                            or regular_sum.dtype is not bounds.dtype
                            or regular_square.dtype is not bounds.dtype
                        ):
                            raise TypeError("regular Vegas summaries must use the bounds dtype")
                        if (
                            sensitive_sum.dtype is not torch.float64
                            or sensitive_square.dtype is not torch.float64
                        ):
                            raise TypeError("sensitive Vegas summaries must use float64")
                        total = total + regular_sum
                        total_square = total_square + regular_square
                        sensitive_total = sensitive_total + sensitive_sum
                        sensitive_total_square = sensitive_total_square + sensitive_square
                    elif len(weighted_output) == 2:
                        regular_weight, sensitive_weight = weighted_output
                        if regular_weight.shape != (count,) or sensitive_weight.shape != (count,):
                            raise ValueError(
                                "mixed weighted Vegas functions must return two values per sample"
                            )
                        if regular_weight.dtype is not bounds.dtype:
                            raise TypeError("regular Vegas weights must use the bounds dtype")
                        if sensitive_weight.dtype is not torch.float64:
                            raise TypeError("sensitive Vegas weights must use float64")
                        total = total + regular_weight.sum()
                        total_square = total_square + regular_weight.square().sum()
                        sensitive_total = sensitive_total + sensitive_weight.sum()
                        sensitive_total_square = (
                            sensitive_total_square + sensitive_weight.square().sum()
                        )
                        # This cast trains the grid only; it is not used in the estimate.
                        absolute_weight = (regular_weight + sensitive_weight.float()).abs()
                    else:
                        raise ValueError(
                            "mixed weighted Vegas functions returned invalid summaries"
                        )
                else:
                    weighted_value = weighted_output
                    if mixed_precision_iteration:
                        raise ValueError("a weighted Vegas function cannot change precision modes")
                    if weighted_value.shape != (count,):
                        raise ValueError(
                            "weighted Vegas functions must return one value per sample"
                        )
            elif cuda_mask_fusion and bounds.is_cuda and bounds.dtype is torch.float32:
                from nlo_torch.custom_kernels.integration_triton import vegas_samples_fused

                x, width = vegas_samples_fused(
                    edges,
                    bounds_lower,
                    bounds_width,
                    bin_index,
                    random,
                )
            else:
                left = torch.gather(edges, 1, bin_index.T).T
                right = torch.gather(edges, 1, (bin_index + 1).T).T
                width = right - left
                u = left + width * random
                x = bounds[:, 0] + bounds_width * u
            if not direct_cuda_weights:
                value = function(x)
                if value.shape != (count,):
                    raise ValueError("Vegas integrands must return one value per sample")

                if (
                    cuda_mask_fusion
                    and bounds.is_cuda
                    and bounds.dtype is torch.float32
                    and dimensions == 4
                    and bins == 32
                    and count <= 65_536
                ):
                    from nlo_torch.custom_kernels.integration import (
                        vegas_accumulate_moments_cuda,
                    )

                    absolute_weight = vegas_accumulate_moments_cuda(
                        value,
                        width,
                        volume,
                        total,
                        total_square,
                        initialize=completed == 0,
                    )
                else:
                    inverse_density = torch.prod(bins * width, dim=1)
                    weighted_value = value * volume * inverse_density
                    weighted_square = weighted_value.square()
                    absolute_weight = weighted_value.abs()
                    total = total + weighted_value.sum()
                    total_square = total_square + weighted_square.sum()
            # CUDA scatter_add_ uses atomic additions, so equal seeds can produce slightly
            # different histograms and therefore different grids on the next iteration.
            # Reducing explicit bin masks preserves the same Vegas statistics and makes a
            # seeded run reproducible on every supported device.
            if (
                cuda_mask_fusion
                and bounds.is_cuda
                and bounds.dtype is torch.float32
                and dimensions == 4
                and bins == 32
            ):
                if direct_cuda_weights:
                    assert histogram_semaphores is not None
                    if mixed_precision_iteration:
                        from nlo_torch.custom_kernels.integration import (
                            vegas_histogram_cuda,
                        )

                        vegas_histogram_cuda(
                            bin_index,
                            absolute_weight,
                            histogram,
                            bin_count,
                            semaphores=histogram_semaphores,
                            initialize=completed == 0,
                        )
                    else:
                        from nlo_torch.custom_kernels.integration import (
                            vegas_weighted_histogram_cuda,
                        )

                        vegas_weighted_histogram_cuda(
                            bin_index,
                            weighted_value,
                            total,
                            total_square,
                            histogram,
                            bin_count,
                            histogram_semaphores,
                            initialize=completed == 0,
                        )
                else:
                    from nlo_torch.custom_kernels.integration import vegas_histogram_cuda

                    vegas_histogram_cuda(
                        bin_index,
                        absolute_weight,
                        histogram,
                        bin_count,
                        semaphores=histogram_semaphores,
                        initialize=completed == 0,
                    )
            elif cuda_mask_fusion and bounds.is_cuda and bounds.dtype is torch.float32:
                from nlo_torch.custom_kernels.integration_triton import vegas_masks_fused

                weighted_mask, count_mask = vegas_masks_fused(
                    bin_index,
                    absolute_weight,
                    bins,
                )
                for dimension in range(dimensions):
                    histogram[dimension] += torch.sum(weighted_mask[dimension], dim=0)
                    bin_count[dimension] += torch.sum(count_mask[dimension], dim=0)
            else:
                bin_labels = torch.arange(bins, device=bounds.device)
                for dimension in range(dimensions):
                    in_bin = bin_index[:, dimension].unsqueeze(1) == bin_labels
                    histogram[dimension] += torch.sum(absolute_weight.unsqueeze(1) * in_bin, dim=0)
                    bin_count[dimension] += torch.sum(in_bin, dim=0)
            completed += count

        n_eval += iteration_samples
        edges = _adapt_vegas_edges(
            edges,
            histogram,
            bin_count,
            cuda_fusion=cuda_mask_fusion,
        )
        if warmup:
            continue

        use_buffered_summary = cuda_buffered_summaries and not mixed_precision_iteration
        if mixed_precision_iteration:
            # Convert the completed float64 regional sums exactly once, at the
            # boundary where they are combined with the float32 regular region.
            total = total + sensitive_total.float()
            total_square = total_square + sensitive_total_square.float()
            estimate = total / iteration_samples
            sample_variance = (total_square - total.square() / iteration_samples).clamp_min(0) / (
                iteration_samples - 1
            )
            variance = (sample_variance / iteration_samples).clamp_min(
                torch.finfo(bounds.dtype).tiny
            )
        elif (
            cuda_mask_fusion
            and bounds.is_cuda
            and bounds.dtype is torch.float32
            and dimensions == 4
            and bins == 32
        ):
            if use_buffered_summary:
                from nlo_torch.custom_kernels.integration import (
                    vegas_store_and_combine_cuda,
                )

                combined = vegas_store_and_combine_cuda(
                    total,
                    total_square,
                    iteration_samples,
                    estimate_buffer,
                    variance_buffer,
                    reported_iterations,
                )
            else:
                from nlo_torch.custom_kernels.integration import (
                    vegas_estimate_variance_cuda,
                )

                summary = vegas_estimate_variance_cuda(total, total_square, iteration_samples)
                estimate = summary[0]
                variance = summary[1]
        else:
            estimate = total / iteration_samples
            sample_variance = (total_square - total.square() / iteration_samples).clamp_min(0) / (
                iteration_samples - 1
            )
            variance = (sample_variance / iteration_samples).clamp_min(
                torch.finfo(bounds.dtype).tiny
            )

        reported_iterations += 1
        if use_buffered_summary:
            combined_value = combined[0]
            error = combined[1]
        else:
            estimates.append(estimate)
            variances.append(variance)
            weights = 1 / torch.stack(variances)
            combined_value = torch.sum(weights * torch.stack(estimates)) / weights.sum()
            error = torch.sqrt(1 / weights.sum())
        tolerance = max(epsabs, epsrel * abs(float(combined_value.item())))
        converged = reported_iterations >= min_iterations and float(error.item()) <= tolerance
        if converged:
            break

    return IntegralResult(combined_value, error, n_eval, converged, seed)


def miser(
    function: Callable[[torch.Tensor], torch.Tensor],
    bounds: torch.Tensor,
    *,
    n_eval: int,
    epsrel: float = 0.2,
    epsabs: float = 0.0,
    seed: int | None = None,
    batch_size: int = 65_536,
    min_leaf_samples: int = 256,
) -> IntegralResult:
    """Recursive stratified Monte Carlo compatible with the source's Miser path."""

    _validate_bounds(bounds)
    if n_eval < 2:
        raise ValueError("Miser requires at least two evaluations")
    seed = torch.seed() if seed is None else seed
    generator = torch.Generator(device=bounds.device)
    generator.manual_seed(seed)
    value, variance, evaluations = _miser_region(
        function,
        bounds,
        n_eval,
        generator,
        batch_size,
        min_leaf_samples,
    )
    error = torch.sqrt(variance.clamp_min(0))
    tolerance = max(epsabs, epsrel * abs(float(value.item())))
    return IntegralResult(value, error, evaluations, float(error.item()) <= tolerance, seed)


def _gauss_kronrod_intervals(
    function: Callable[..., torch.Tensor],
    lower: torch.Tensor,
    upper: torch.Tensor,
    rule: int,
    parameter: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rule == 21:
        xgk = lower.new_tensor(
            [
                0.9956571630258081,
                0.9739065285171717,
                0.9301574913557082,
                0.8650633666889845,
                0.7808177265864169,
                0.6794095682990244,
                0.5627571346686047,
                0.4333953941292472,
                0.2943928627014602,
                0.1488743389816312,
            ]
        )
        wgk = lower.new_tensor(
            [
                0.011694638867371874,
                0.03255816230796473,
                0.054755896574351996,
                0.07503967481091995,
                0.0931254545836976,
                0.10938715880229764,
                0.12349197626206585,
                0.13470921731147333,
                0.14277593857706008,
                0.14773910490133849,
            ]
        )
        center_weight = 0.1494455540029169
        gauss_index = [1, 3, 5, 7, 9]
        gauss_weight = lower.new_tensor(
            [
                0.06667134430868814,
                0.1494513491505806,
                0.21908636251598204,
                0.26926671930999635,
                0.29552422471475287,
            ]
        )
        gauss_center_weight = 0.0
    else:
        xgk = lower.new_tensor(
            [
                0.9914553711208126,
                0.9491079123427585,
                0.8648644233597691,
                0.7415311855993945,
                0.5860872354676911,
                0.4058451513773972,
                0.20778495500789847,
            ]
        )
        wgk = lower.new_tensor(
            [
                0.022935322010529225,
                0.06309209262997856,
                0.10479001032225018,
                0.14065325971552592,
                0.1690047266392679,
                0.19035057806478542,
                0.20443294007529889,
            ]
        )
        center_weight = 0.20948214108472782
        gauss_index = [1, 3, 5]
        gauss_weight = lower.new_tensor(
            [0.1294849661688697, 0.27970539148927667, 0.38183005050511894]
        )
        gauss_center_weight = 0.4179591836734694

    center = (lower + upper) / 2
    half_width = (upper - lower) / 2
    offset = half_width.unsqueeze(-1) * xgk
    x = torch.cat(
        (center.unsqueeze(-1) - offset, center.unsqueeze(-1), center.unsqueeze(-1) + offset),
        dim=-1,
    )
    if parameter is None:
        value = function(x.reshape(-1)).reshape(x.shape)
    else:
        if parameter.shape != lower.shape:
            raise ValueError("quadrature parameters must match the interval shape")
        expanded_parameter = parameter.unsqueeze(-1).expand_as(x)
        value = function(x.reshape(-1), expanded_parameter.reshape(-1)).reshape(x.shape)
    negative = value[:, : xgk.numel()]
    center_value = value[:, xgk.numel()]
    positive = value[:, xgk.numel() + 1 :]
    pair = negative + positive

    kronrod_unscaled = torch.sum(pair * wgk, dim=-1) + center_weight * center_value
    gauss_unscaled = (
        torch.sum(pair[:, gauss_index] * gauss_weight, dim=-1) + gauss_center_weight * center_value
    )
    estimate = half_width * kronrod_unscaled
    error = half_width * torch.abs(kronrod_unscaled - gauss_unscaled)

    mean = kronrod_unscaled / 2
    resabs = half_width * (
        torch.sum((negative.abs() + positive.abs()) * wgk, dim=-1)
        + center_weight * center_value.abs()
    )
    resasc = half_width * (
        torch.sum(
            (torch.abs(negative - mean.unsqueeze(-1)) + torch.abs(positive - mean.unsqueeze(-1)))
            * wgk,
            dim=-1,
        )
        + center_weight * torch.abs(center_value - mean)
    )
    scale = torch.pow(200 * error / torch.where(resasc > 0, resasc, torch.ones_like(resasc)), 1.5)
    error = torch.where(
        (resasc > 0) & (error > 0), resasc * torch.minimum(scale, torch.ones_like(scale)), error
    )
    minimum_error = 50 * torch.finfo(lower.dtype).eps * resabs
    return estimate, torch.maximum(error, minimum_error)


def _tensor_gauss_legendre_once(
    function: Callable[[torch.Tensor], torch.Tensor],
    bounds: torch.Tensor,
    points: int,
    batch_size: int,
) -> torch.Tensor:
    nodes, weights = _gauss_legendre_nodes_weights(points, bounds)
    coordinates = []
    dimension_weights = []
    for dimension in range(bounds.shape[0]):
        midpoint = (bounds[dimension, 0] + bounds[dimension, 1]) / 2
        half_width = (bounds[dimension, 1] - bounds[dimension, 0]) / 2
        coordinates.append(midpoint + half_width * nodes)
        dimension_weights.append(half_width * weights)

    coordinate_grid = torch.meshgrid(*coordinates, indexing="ij")
    weight_grid = torch.meshgrid(*dimension_weights, indexing="ij")
    x = torch.stack([coordinate.reshape(-1) for coordinate in coordinate_grid], dim=1)
    combined_weight = torch.ones(x.shape[0], dtype=bounds.dtype, device=bounds.device)
    for weight in weight_grid:
        combined_weight = combined_weight * weight.reshape(-1)

    result = bounds.new_zeros(())
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        value = function(x[start:stop])
        if value.shape != (stop - start,):
            raise ValueError("quadrature integrands must return one value per point")
        result = result + torch.sum(combined_weight[start:stop] * value)
    return result


def _gauss_legendre_nodes_weights(
    points: int, reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.arange(points, dtype=reference.dtype, device=reference.device)
    x = torch.cos(math.pi * (index + 0.75) / (points + 0.5))
    derivative = torch.empty_like(x)
    for _ in range(12):
        p0 = torch.ones_like(x)
        p1 = x
        for n in range(2, points + 1):
            p0, p1 = p1, ((2 * n - 1) * x * p1 - (n - 1) * p0) / n
        derivative = points * (x * p1 - p0) / (x.square() - 1)
        x = x - p1 / derivative
    p0 = torch.ones_like(x)
    p1 = x
    for n in range(2, points + 1):
        p0, p1 = p1, ((2 * n - 1) * x * p1 - (n - 1) * p0) / n
    derivative = points * (x * p1 - p0) / (x.square() - 1)
    weights = 2 / ((1 - x.square()) * derivative.square())
    order = torch.argsort(x)
    return x[order], weights[order]


def _adapt_vegas_edges(
    edges: torch.Tensor,
    histogram: torch.Tensor,
    bin_count: torch.Tensor,
    *,
    cuda_fusion: bool = False,
) -> torch.Tensor:
    if (
        cuda_fusion
        and edges.is_cuda
        and edges.dtype is torch.float32
        and edges.shape == (4, 33)
        and histogram.shape == (4, 32)
        and bin_count.shape == histogram.shape
    ):
        from nlo_torch.custom_kernels.integration import vegas_adapt_edges_cuda

        return vegas_adapt_edges_cuda(edges, histogram, bin_count)

    if cuda_fusion and edges.is_cuda and edges.dtype is torch.float32:
        from nlo_torch.custom_kernels.integration_triton import vegas_smoothed_fused

        smoothed = vegas_smoothed_fused(histogram, bin_count)
    else:
        importance = histogram / bin_count.clamp_min(1)
        smoothed = importance.clone()
        smoothed[:, 1:-1] = (importance[:, :-2] + importance[:, 1:-1] + importance[:, 2:]) / 3
        smoothed[:, 0] = (importance[:, 0] + importance[:, 1]) / 2
        smoothed[:, -1] = (importance[:, -2] + importance[:, -1]) / 2
        smoothed = smoothed + torch.finfo(edges.dtype).eps

    new_edges = []
    cumulative_values = []
    bins = histogram.shape[1]
    for dimension in range(histogram.shape[0]):
        cumulative = torch.cat((edges.new_zeros(1), torch.cumsum(smoothed[dimension], dim=0)))
        if cuda_fusion and edges.is_cuda and edges.dtype is torch.float32:
            cumulative_values.append(cumulative)
            continue
        target = torch.linspace(
            0,
            float(cumulative[-1].item()),
            bins + 1,
            dtype=edges.dtype,
            device=edges.device,
        )
        index = torch.searchsorted(cumulative, target, right=True).sub(1).clamp(0, bins - 1)
        fraction = (target - cumulative[index]) / smoothed[dimension, index]
        updated = edges[dimension, index] + fraction * (
            edges[dimension, index + 1] - edges[dimension, index]
        )
        updated[0] = 0
        updated[-1] = 1
        new_edges.append(updated)
    if cumulative_values:
        from nlo_torch.custom_kernels.integration_triton import vegas_edges_fused

        return vegas_edges_fused(
            edges,
            smoothed,
            torch.stack(cumulative_values),
        )
    return torch.stack(new_edges)


def _miser_region(
    function: Callable[[torch.Tensor], torch.Tensor],
    bounds: torch.Tensor,
    n_eval: int,
    generator: torch.Generator,
    batch_size: int,
    min_leaf_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    dimensions = bounds.shape[0]
    pilot = min(max(32 * dimensions, min_leaf_samples), n_eval // 5)
    if n_eval < 2 * min_leaf_samples + pilot:
        return _uniform_region(function, bounds, n_eval, generator, batch_size)

    u = torch.rand(
        (pilot, dimensions),
        dtype=bounds.dtype,
        device=bounds.device,
        generator=generator,
    )
    x = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * u
    value = function(x)
    midpoint = (bounds[:, 0] + bounds[:, 1]) / 2
    scores = []
    left_sigma = []
    right_sigma = []
    for dimension in range(dimensions):
        left = x[:, dimension] <= midpoint[dimension]
        right = ~left
        if int(left.sum().item()) < 2 or int(right.sum().item()) < 2:
            scores.append(bounds.new_tensor(torch.inf))
            left_sigma.append(bounds.new_tensor(1.0))
            right_sigma.append(bounds.new_tensor(1.0))
        else:
            sigma_left = value[left].std()
            sigma_right = value[right].std()
            scores.append(sigma_left + sigma_right)
            left_sigma.append(sigma_left)
            right_sigma.append(sigma_right)

    split = int(torch.argmin(torch.stack(scores)).item())
    sigma_left = left_sigma[split]
    sigma_right = right_sigma[split]
    remaining = n_eval - pilot
    fraction = sigma_left / (sigma_left + sigma_right).clamp_min(torch.finfo(bounds.dtype).tiny)
    left_eval = int(round(remaining * float(fraction.item())))
    left_eval = max(min_leaf_samples, min(remaining - min_leaf_samples, left_eval))
    right_eval = remaining - left_eval

    left_bounds = bounds.clone()
    right_bounds = bounds.clone()
    left_bounds[split, 1] = midpoint[split]
    right_bounds[split, 0] = midpoint[split]
    left_value, left_variance, left_count = _miser_region(
        function, left_bounds, left_eval, generator, batch_size, min_leaf_samples
    )
    right_value, right_variance, right_count = _miser_region(
        function, right_bounds, right_eval, generator, batch_size, min_leaf_samples
    )
    return (
        left_value + right_value,
        left_variance + right_variance,
        pilot + left_count + right_count,
    )


def _uniform_region(
    function: Callable[[torch.Tensor], torch.Tensor],
    bounds: torch.Tensor,
    n_eval: int,
    generator: torch.Generator,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    dimensions = bounds.shape[0]
    total = bounds.new_zeros(())
    total_square = bounds.new_zeros(())
    completed = 0
    while completed < n_eval:
        count = min(batch_size, n_eval - completed)
        u = torch.rand(
            (count, dimensions),
            dtype=bounds.dtype,
            device=bounds.device,
            generator=generator,
        )
        x = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * u
        value = function(x)
        total = total + value.sum()
        total_square = total_square + value.square().sum()
        completed += count

    volume = torch.prod(bounds[:, 1] - bounds[:, 0])
    mean = total / n_eval
    sample_variance = (total_square - total.square() / n_eval).clamp_min(0) / (n_eval - 1)
    return volume * mean, volume.square() * sample_variance / n_eval, n_eval


def _validate_bounds(bounds: torch.Tensor) -> None:
    if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] < 1:
        raise ValueError("bounds must have shape (dimensions, 2)")
    if not bounds.is_floating_point():
        raise TypeError("bounds must be floating point")
    if not bool(torch.isfinite(bounds).all()) or not bool((bounds[:, 1] > bounds[:, 0]).all()):
        raise ValueError("bounds must be finite with upper > lower")


__all__ = [
    "IntegralResult",
    "adaptive_gauss_kronrod_1d",
    "miser",
    "tensor_gauss_legendre",
    "vegas",
]
