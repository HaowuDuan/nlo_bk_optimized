"""Fixed-Sobol integration for the supported GBW NLO qg cross section."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from nlo_torch.custom_kernels.dis.i1.extension import (
    dis_gbw_i1_integrand_regions_cuda,
    dis_gbw_i1_region_sums_cuda,
)
from nlo_torch.custom_kernels.dis.i2_i3.extension import (
    dis_nested_gbw_i2_i3_integrand_cuda,
)
from nlo_torch.custom_kernels.dis.sampling.extension import dis_sobol_endpoint_map_cuda
from nlo_torch.dipole.amplitude import GBW
from nlo_torch.dis.config import (
    DISConfig,
    NcScheme,
    Polarization,
    Quark,
    RunningCouplingIRScheme,
    RunningCouplingScheme,
)
from nlo_torch.numerics.integration import IntegralResult, _adapt_vegas_edges

TRAINING_SAMPLES = 65_536
TRAINING_ROUNDS = 3
SAMPLES_PER_REPLICATE = 32_768
REPLICATES = 7
GRID_BINS = 32
FIXED_SOBOL_EVALUATIONS = 2 * (
    TRAINING_ROUNDS * TRAINING_SAMPLES + REPLICATES * SAMPLES_PER_REPLICATE
)

TripleIntegrand = Callable[
    [torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]
I1SumIntegrand = Callable[
    [torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]
NestedIntegrand = Callable[
    [torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]


@dataclass(frozen=True, slots=True)
class SobolPointSet:
    training: tuple[torch.Tensor, ...]
    replicates: tuple[torch.Tensor, ...]
    seed: int


@dataclass(frozen=True, slots=True)
class FixedSobolResult:
    contributions: tuple[IntegralResult, IntegralResult, IntegralResult]
    total: IntegralResult


def make_sobol_points(
    dimensions: int,
    *,
    training_samples: int,
    training_rounds: int,
    samples_per_replicate: int,
    replicates: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> SobolPointSet:
    """Create reusable, independently scrambled Sobol point sets."""

    if dimensions < 1 or training_rounds < 1 or replicates < 2:
        raise ValueError(
            "dimensions and training rounds must be positive; replicates must exceed one"
        )
    for count in (training_samples, samples_per_replicate):
        if count < 2 or count & (count - 1):
            raise ValueError("Sobol sample counts must be powers of two greater than one")

    def draw(count: int, offset: int) -> torch.Tensor:
        points = torch.quasirandom.SobolEngine(
            dimensions,
            scramble=True,
            seed=seed + offset,
        ).draw_base2(count.bit_length() - 1, dtype=dtype)
        return points.to(device=device)

    training = tuple(draw(training_samples, index) for index in range(training_rounds))
    estimates = tuple(draw(samples_per_replicate, 10_000 + index) for index in range(replicates))
    return SobolPointSet(training, estimates, seed)


def map_importance_grid(
    points: torch.Tensor,
    edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bins = edges.shape[1] - 1
    scaled = points * bins
    bin_index = scaled.floor().to(torch.long).clamp_max(bins - 1)
    fraction = scaled - bin_index
    left = torch.gather(edges, 1, bin_index.T).T
    right = torch.gather(edges, 1, (bin_index + 1).T).T
    width = right - left
    source = left + width * fraction
    inverse_density = torch.prod(bins * width, dim=1)
    return source.contiguous(), inverse_density, bin_index


def paired_endpoint_inputs(
    source: torch.Tensor,
    maxr: float,
    *,
    fold_angle: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the gluon plane by its nearest dipole endpoint."""

    parent = maxr * source[:, 2]
    radial_fraction = source[:, 3]
    angle_fraction = 0.5 * source[:, 4] if fold_angle else source[:, 4]
    angle = 2 * torch.pi * angle_fraction
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    closer_boundary = torch.where(
        cosine > 0,
        parent / (2 * cosine.clamp_min(torch.finfo(source.dtype).tiny)),
        torch.full_like(parent, torch.inf),
    )

    first_limit = torch.minimum(torch.full_like(parent, maxr), closer_boundary)
    first = source.clone()
    first[:, 3] = radial_fraction * first_limit / maxr
    first[:, 4] = angle_fraction
    first_weight = first_limit / maxr

    disk_boundary = parent * cosine + torch.sqrt(
        (maxr**2 - parent.square() * sine.square()).clamp_min(0)
    )
    second_limit = torch.minimum(disk_boundary, closer_boundary).clamp_min(0)
    second_radius = radial_fraction * second_limit
    global_x = parent - second_radius * cosine
    global_y = second_radius * sine
    global_radius = torch.sqrt(global_x.square() + global_y.square())
    second = source.clone()
    second[:, 3] = global_radius / maxr
    second[:, 4] = torch.atan2(global_y, global_x).remainder(2 * torch.pi) / (2 * torch.pi)
    second_weight = (
        second_radius
        * second_limit
        / (global_radius.clamp_min(torch.finfo(source.dtype).tiny) * maxr)
    )
    return first.contiguous(), first_weight, second.contiguous(), second_weight


@torch.no_grad()
def learn_triple_importance_grid(
    function: TripleIntegrand,
    training_points: tuple[torch.Tensor, ...],
    *,
    maxr: float,
    bins: int = GRID_BINS,
    fold_angle: bool = True,
    i1_weight: float = 1.0,
) -> torch.Tensor:
    """Learn one frozen grid from normalized I1, I2, and I3 magnitudes."""

    if not training_points:
        raise ValueError("at least one independent training point set is required")
    dimensions = training_points[0].shape[1]
    reference = training_points[0]
    edges = (
        torch.linspace(0, 1, bins + 1, device=reference.device, dtype=reference.dtype)
        .expand(dimensions, -1)
        .clone()
    )
    for points in training_points:
        source, inverse_density, bin_index = map_importance_grid(points, edges)
        first, first_weight, second, second_weight = paired_endpoint_inputs(
            source,
            maxr,
            fold_angle=fold_angle,
        )
        first_regular, first_sensitive, first_I2, first_I3 = function(first)
        second_regular, second_sensitive, second_I2, second_I3 = function(second)
        first_values = first_regular + first_sensitive.float(), first_I2, first_I3
        second_values = second_regular + second_sensitive.float(), second_I2, second_I3
        weighted = tuple(
            (first_value * first_weight + second_value * second_weight) * inverse_density
            for first_value, second_value in zip(first_values, second_values, strict=True)
        )
        tiny = torch.finfo(reference.dtype).tiny
        weights = (i1_weight, 1.0, 1.0)
        score = sum(
            weight * value.abs() / value.abs().mean().clamp_min(tiny)
            for weight, value in zip(weights, weighted, strict=True)
        )
        histogram = reference.new_zeros((dimensions, bins))
        bin_count = reference.new_zeros((dimensions, bins))
        ones = torch.ones_like(score)
        for dimension in range(dimensions):
            histogram[dimension].scatter_add_(0, bin_index[:, dimension], score)
            bin_count[dimension].scatter_add_(0, bin_index[:, dimension], ones)
        edges = _adapt_vegas_edges(edges, histogram, bin_count, cuda_fusion=False)
    return edges


@torch.no_grad()
def integrate_triple_fixed_sobol(
    i1_sum_function: I1SumIntegrand,
    nested_function: NestedIntegrand,
    edges: torch.Tensor,
    replicate_points: tuple[torch.Tensor, ...],
    *,
    maxr: float,
    epsrel: float,
    seed: int,
    training_evaluations: int = 0,
) -> FixedSobolResult:
    """Estimate I1, I2, I3, and their correlated sum."""

    if len(replicate_points) < 2:
        raise ValueError("at least two scrambled Sobol replicates are required")
    estimates = []
    for points in replicate_points:
        first, first_weight, second, second_weight = dis_sobol_endpoint_map_cuda(
            points.contiguous(),
            edges,
            maxr,
            fold_angle=True,
        )
        first_regular_sums, first_sensitive_sums = i1_sum_function(first, first_weight)
        second_regular_sums, second_sensitive_sums = i1_sum_function(second, second_weight)
        first_I2, first_I3 = nested_function(first)
        second_I2, second_I3 = nested_function(second)
        sample_count = points.shape[0]
        regular_I1 = (first_regular_sums.sum() + second_regular_sums.sum()) / sample_count
        sensitive_I1 = (
            (first_sensitive_sums.sum() + second_sensitive_sums.sum()) / sample_count
        ).float()
        estimates.append(
            torch.stack(
                (
                    (regular_I1 + sensitive_I1).double(),
                    (first_I2 * first_weight + second_I2 * second_weight).double().mean(),
                    (first_I3 * first_weight + second_I3 * second_weight).double().mean(),
                )
            )
        )

    stacked = torch.stack(estimates)
    samples = training_evaluations + 2 * sum(points.shape[0] for points in replicate_points)

    def result(values: torch.Tensor) -> IntegralResult:
        value = values.mean()
        error = values.std(unbiased=True) / math.sqrt(values.numel())
        return IntegralResult(
            value,
            error,
            samples,
            bool(error <= epsrel * value.abs()),
            seed,
        )

    contributions = tuple(result(stacked[:, index]) for index in range(3))
    return FixedSobolResult(contributions, result(stacked.sum(dim=1)))


def gbw_qg_fixed_sobol(
    Q2: torch.Tensor,
    xbj: torch.Tensor,
    polarization: Polarization,
    quark: Quark,
    dipole: GBW,
    config: DISConfig,
    *,
    seed: int,
) -> FixedSobolResult:
    """Evaluate one quark's complete qg contribution with the validated fast path."""

    if (
        not Q2.is_cuda
        or Q2.dtype is not torch.float32
        or Q2.numel() != 1
        or xbj.device != Q2.device
        or xbj.dtype is not torch.float32
        or xbj.numel() != 1
    ):
        raise ValueError("fixed-Sobol DIS requires matching CUDA float32 scalars")

    point_set = make_sobol_points(
        5,
        training_samples=TRAINING_SAMPLES,
        training_rounds=TRAINING_ROUNDS,
        samples_per_replicate=SAMPLES_PER_REPLICATE,
        replicates=REPLICATES,
        device=Q2.device,
        dtype=Q2.dtype,
        seed=seed,
    )
    kernel_options = {
        "transverse": polarization is Polarization.T,
        "maxr": config.maxr,
        "Qs0sqr": dipole.Qs0sqr,
        "lambda_": dipole.lambda_,
        "gamma": dipole.gamma,
        "x0": dipole.x0,
        "finite_nc": config.nc_scheme is NcScheme.FiniteNC,
        "parent_coupling": config.rc_scheme is RunningCouplingScheme.PARENT,
        "smooth_coupling": config.rc_ir_scheme is RunningCouplingIRScheme.SMOOTH,
        "coupling_C2": config.C2_alpha,
        "active_flavors": config.active_flavors,
        "maximum_alpha": config.max_alpha_s_freeze,
    }

    def i1_regions(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return dis_gbw_i1_integrand_regions_cuda(Q2, xbj, quark.mass, x, **kernel_options)

    def i1_sums(
        x: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return dis_gbw_i1_region_sums_cuda(
            Q2,
            xbj,
            quark.mass,
            x,
            weights,
            **kernel_options,
        )

    def i2_i3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return dis_nested_gbw_i2_i3_integrand_cuda(
            Q2,
            xbj,
            quark.mass,
            x,
            points=config.cuda_nested_points,
            **kernel_options,
        )

    def contributions(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        regular_I1, sensitive_I1 = i1_regions(x)
        I2, I3 = i2_i3(x)
        return regular_I1, sensitive_I1, I2, I3

    edges = learn_triple_importance_grid(
        contributions,
        point_set.training,
        maxr=config.maxr,
    )
    return integrate_triple_fixed_sobol(
        i1_sums,
        i2_i3,
        edges,
        point_set.replicates,
        maxr=config.maxr,
        epsrel=config.epsrel,
        seed=seed,
        training_evaluations=2 * TRAINING_ROUNDS * TRAINING_SAMPLES,
    )


__all__ = [
    "FIXED_SOBOL_EVALUATIONS",
    "FixedSobolResult",
    "SobolPointSet",
    "gbw_qg_fixed_sobol",
    "integrate_triple_fixed_sobol",
    "learn_triple_importance_grid",
    "make_sobol_points",
    "map_importance_grid",
    "paired_endpoint_inputs",
]
