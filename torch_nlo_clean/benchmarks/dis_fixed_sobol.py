"""Standalone fixed-Sobol experiment for the five-dimensional DIS I2/I3 integrals."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable

import torch

from nlo_torch.custom_kernels.dis.sampling.extension import dis_sobol_endpoint_map_cuda
from nlo_torch.dipole import GBW
from nlo_torch.dis import DISConfig, Polarization, Quark, QuarkType
from nlo_torch.dis.fixed_sobol import (
    make_sobol_points,
    map_importance_grid,
    paired_endpoint_inputs,
)
from nlo_torch.dis.observables import integrand_qg_nested_massive
from nlo_torch.numerics.integration import IntegralResult, _adapt_vegas_edges

Integrand = Callable[[torch.Tensor], torch.Tensor]


@torch.no_grad()
def learn_importance_grid(
    function: Integrand,
    training_points: tuple[torch.Tensor, ...],
    *,
    maxr: float,
    bins: int = 32,
) -> torch.Tensor:
    """Learn a separable grid, then freeze it before the reported fixed-sample estimates."""

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
        if points.shape[1] != dimensions or points.device != reference.device:
            raise ValueError("all training point sets must have matching shape and device")
        source, inverse_density, bin_index = map_importance_grid(points, edges)
        weighted = _paired_endpoint_values(function, source, maxr) * inverse_density
        histogram = reference.new_zeros((dimensions, bins))
        bin_count = reference.new_zeros((dimensions, bins))
        ones = torch.ones_like(weighted)
        for dimension in range(dimensions):
            histogram[dimension].scatter_add_(
                0,
                bin_index[:, dimension],
                weighted.abs(),
            )
            bin_count[dimension].scatter_add_(
                0,
                bin_index[:, dimension],
                ones,
            )
        edges = _adapt_vegas_edges(edges, histogram, bin_count, cuda_fusion=False)
    return edges


@torch.no_grad()
def integrate_fixed_sobol(
    function: Integrand,
    edges: torch.Tensor,
    replicate_points: tuple[torch.Tensor, ...],
    *,
    maxr: float,
    epsrel: float,
    seed: int,
    cuda_mapping: bool = True,
) -> IntegralResult:
    """Integrate with a frozen grid and report uncertainty across scrambled Sobol nets."""

    if len(replicate_points) < 2:
        raise ValueError("at least two scrambled Sobol replicates are required")
    estimates = []
    for points in replicate_points:
        if cuda_mapping and points.is_cuda and points.dtype is torch.float32:
            first, first_weight, second, second_weight = dis_sobol_endpoint_map_cuda(
                points.contiguous(),
                edges,
                maxr,
            )
            weighted = function(first) * first_weight + function(second) * second_weight
        else:
            source, inverse_density, _ = map_importance_grid(points, edges)
            weighted = _paired_endpoint_values(function, source, maxr) * inverse_density
        estimates.append(weighted.double().mean())
    stacked = torch.stack(estimates)
    value = stacked.mean()
    error = stacked.std(unbiased=True) / math.sqrt(len(estimates))
    converged = float(error) <= epsrel * abs(float(value))
    samples = sum(points.shape[0] for points in replicate_points)
    return IntegralResult(value, error, 2 * samples, converged, seed)


def _paired_endpoint_values(
    function: Integrand,
    source: torch.Tensor,
    maxr: float,
    *,
    fold_angle: bool = False,
) -> torch.Tensor:
    first, first_weight, second, second_weight = paired_endpoint_inputs(
        source, maxr, fold_angle=fold_angle
    )
    return function(first) * first_weight + function(second) * second_weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-samples", type=int, default=65_536)
    parser.add_argument("--training-rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=65_536)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3_100)
    parser.add_argument(
        "--matched-cpu-error",
        action="store_true",
        help="use the smallest exploratory power-of-two count that cleared the CPU20 error",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")

    device = torch.device("cuda")
    point_start = time.perf_counter()
    point_set = make_sobol_points(
        5,
        training_samples=args.training_samples,
        training_rounds=args.training_rounds,
        samples_per_replicate=args.samples,
        replicates=args.replicates,
        device=device,
        dtype=torch.float32,
        seed=args.seed,
    )
    torch.cuda.synchronize()
    point_seconds = time.perf_counter() - point_start

    Q2 = torch.tensor(9.0, device=device)
    xbj = torch.tensor(0.001, device=device)
    dipole = GBW(Qs0sqr=1.0, lambda_=0.3, gamma=1.0, x0=1.0)
    quark = Quark(QuarkType.C, 1.4)
    config = DISConfig(
        quarks=(quark,),
        maxr=5.0,
        nf_alphas=1,
        epsrel=0.001,
        cuda_fusion=True,
        cuda_nested=True,
        cuda_nested_points=48,
    )
    references = {
        "L_I2": (-0.024439455902300525, 0.00004764427185133986, 0.315235827),
        "L_I3": (0.009414936380564324, 0.000026852425497947778, 0.427307533),
        "T_I2": (-0.08719375357937224, 0.00026850014906050766, 0.932526175),
        "T_I3": (0.04848797141670969, 0.00013338145018266037, 1.359838948),
    }
    matched_samples = {
        "L_I2": 16_384,
        "L_I3": 16_384,
        "T_I2": 32_768,
        "T_I3": 32_768,
    }
    report: dict[str, object] = {
        "method": (
            "independently trained frozen grid, custom CUDA paired-endpoint mapping, "
            "fixed scrambled Sobol replicates"
        ),
        "point_generation_seconds": point_seconds,
        "training_samples": args.training_samples,
        "training_rounds": args.training_rounds,
        "samples_per_replicate": args.samples,
        "replicates": args.replicates,
        "results": {},
    }
    results: dict[str, object] = {}
    for polarization in (Polarization.L, Polarization.T):
        for contribution in ("I2", "I3"):
            name = f"{polarization.name}_{contribution}"

            def function(x: torch.Tensor) -> torch.Tensor:
                return integrand_qg_nested_massive(
                    x,
                    Q2,
                    xbj,
                    polarization,
                    quark,
                    contribution,
                    dipole,
                    config,
                )

            # Exclude one-time physics-kernel specialization from every reported path.
            _paired_endpoint_values(function, point_set.training[0][:1024], config.maxr)
            torch.cuda.synchronize()
            start = time.perf_counter()
            edges = learn_importance_grid(
                function,
                point_set.training,
                maxr=config.maxr,
            )
            torch.cuda.synchronize()
            training_seconds = time.perf_counter() - start

            # Load and warm the standalone mapper outside the reported evaluation.
            warm_first, _, warm_second, _ = dis_sobol_endpoint_map_cuda(
                point_set.training[0][:1024].contiguous(),
                edges,
                config.maxr,
            )
            function(warm_first)
            function(warm_second)

            measurements = []
            result = None
            contribution_samples = matched_samples[name] if args.matched_cpu_error else args.samples
            if contribution_samples > args.samples:
                raise ValueError("--samples is smaller than the matched contribution count")
            replicate_points = tuple(
                points[:contribution_samples] for points in point_set.replicates
            )
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = integrate_fixed_sobol(
                    function,
                    edges,
                    replicate_points,
                    maxr=config.maxr,
                    epsrel=config.epsrel,
                    seed=args.seed,
                )
                torch.cuda.synchronize()
                measurements.append(time.perf_counter() - start)
            assert result is not None
            reference_value, reference_error, cpp_seconds = references[name]
            evaluation_seconds = statistics.median(measurements)
            total_seconds = training_seconds + evaluation_seconds
            results[name] = {
                "value": float(result.value),
                "error": float(result.error),
                "reference_value": reference_value,
                "reference_error": reference_error,
                "samples_per_replicate": contribution_samples,
                "error_over_reference": float(result.error) / reference_error,
                "difference_in_combined_errors": abs(float(result.value) - reference_value)
                / math.hypot(float(result.error), reference_error),
                "training_seconds": training_seconds,
                "evaluation_seconds": measurements,
                "evaluation_seconds_median": evaluation_seconds,
                "first_use_seconds": total_seconds,
                "cpp20_over_first_use_speedup": cpp_seconds / total_seconds,
                "physics_evaluations": 2
                * (
                    args.training_rounds * args.training_samples
                    + args.replicates * contribution_samples
                ),
            }
    report["results"] = results
    evaluation_total = sum(
        float(result["evaluation_seconds_median"])
        for result in results.values()
        if isinstance(result, dict)
    )
    training_total = sum(
        float(result["training_seconds"]) for result in results.values() if isinstance(result, dict)
    )
    cpp_total = sum(reference[2] for reference in references.values())
    report["aggregate"] = {
        "frozen_grid_evaluation_seconds": evaluation_total,
        "grid_training_seconds": training_total,
        "first_use_seconds": training_total + evaluation_total,
        "cpp20_over_frozen_grid_evaluation_speedup": cpp_total / evaluation_total,
        "cpp20_over_first_use_speedup": cpp_total / (training_total + evaluation_total),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
