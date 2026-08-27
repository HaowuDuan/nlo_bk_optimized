"""Benchmark fixed-Sobol DIS I1, I2, and I3 CUDA kernels."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

from nlo_torch.custom_kernels.dis.i1.extension import (
    dis_gbw_i1_integrand_regions_cuda,
    dis_gbw_i1_region_sums_cuda,
)
from nlo_torch.custom_kernels.dis.i2_i3.extension import (
    dis_nested_gbw_i2_i3_integrand_cuda,
)
from nlo_torch.custom_kernels.dis.sampling.extension import dis_sobol_endpoint_map_cuda
from nlo_torch.dipole import GBW
from nlo_torch.dis import DISConfig, Polarization, Quark, QuarkType
from nlo_torch.dis.config import NcScheme, RunningCouplingIRScheme, RunningCouplingScheme
from nlo_torch.dis.fixed_sobol import (
    integrate_triple_fixed_sobol,
    learn_triple_importance_grid,
    make_sobol_points,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-samples", type=int, default=65_536)
    parser.add_argument("--training-rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=32_768)
    parser.add_argument("--replicates", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=3_100)
    parser.add_argument("--i1-grid-weight", type=float, default=1.0)
    parser.add_argument("--bins", type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")

    device = torch.device("cuda")
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
        "L_I1": (0.10206739088678346, 0.0002535000977528945, 0.248004322),
        "L_I2": (-0.024439455902300525, 0.00004764427185133986, 0.315235827),
        "L_I3": (0.009414936380564324, 0.000026852425497947778, 0.427307533),
        "T_I1": (0.40263591173620866, 0.0017956984452001683, 0.341898261),
        "T_I2": (-0.08719375357937224, 0.00026850014906050766, 0.932526175),
        "T_I3": (0.04848797141670969, 0.00013338145018266037, 1.359838948),
    }
    report: dict[str, object] = {
        "method": "one frozen grid; split-precision I1 and combined I2/I3 per polarization",
        "seed": args.seed,
        "fold_angle": True,
        "replicates": args.replicates,
        "samples_per_replicate": args.samples,
        "i1_grid_weight": args.i1_grid_weight,
        "grid_bins": args.bins,
        "results": {},
    }
    results: dict[str, object] = {}
    evaluation_total = 0.0
    training_total = 0.0
    for polarization in (Polarization.L, Polarization.T):
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
            return dis_gbw_i1_integrand_regions_cuda(
                Q2,
                xbj,
                quark.mass,
                x,
                **kernel_options,
            )

        def i1_sum_function(
            x: torch.Tensor,
            sample_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return dis_gbw_i1_region_sums_cuda(
                Q2,
                xbj,
                quark.mass,
                x,
                sample_weights,
                **kernel_options,
            )

        def nested_function(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return dis_nested_gbw_i2_i3_integrand_cuda(
                Q2,
                xbj,
                quark.mass,
                x,
                points=config.cuda_nested_points,
                **kernel_options,
            )

        def function(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            regular_I1, sensitive_I1 = i1_regions(x)
            I2, I3 = nested_function(x)
            return regular_I1, sensitive_I1, I2, I3

        function(point_set.training[0][:1024])
        torch.cuda.synchronize()
        start = time.perf_counter()
        edges = learn_triple_importance_grid(
            function,
            point_set.training,
            maxr=config.maxr,
            bins=args.bins,
            i1_weight=args.i1_grid_weight,
        )
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - start
        training_total += training_seconds

        warm_first, warm_first_weight, warm_second, warm_second_weight = (
            dis_sobol_endpoint_map_cuda(
                point_set.replicates[0][:1024].contiguous(),
                edges,
                config.maxr,
                fold_angle=True,
            )
        )
        i1_sum_function(warm_first, warm_first_weight)
        i1_sum_function(warm_second, warm_second_weight)
        nested_function(warm_first)
        nested_function(warm_second)
        measurements = []
        result = None
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = integrate_triple_fixed_sobol(
                i1_sum_function,
                nested_function,
                edges,
                point_set.replicates,
                maxr=config.maxr,
                epsrel=config.epsrel,
                seed=args.seed,
            )
            torch.cuda.synchronize()
            measurements.append(time.perf_counter() - start)
        assert result is not None
        evaluation_seconds = statistics.median(measurements)
        evaluation_total += evaluation_seconds
        prefix = polarization.name
        for contribution, estimate in zip(("I1", "I2", "I3"), result.contributions, strict=True):
            name = f"{prefix}_{contribution}"
            reference_value, reference_error, _ = references[name]
            results[name] = {
                "value": float(estimate.value),
                "error": float(estimate.error),
                "reference_value": reference_value,
                "reference_error": reference_error,
                "error_over_reference": float(estimate.error) / reference_error,
                "difference_in_combined_errors": abs(float(estimate.value) - reference_value)
                / math.hypot(float(estimate.error), reference_error),
            }
        results[f"{prefix}_joint_timing"] = {
            "training_seconds": training_seconds,
            "evaluation_seconds": measurements,
            "evaluation_seconds_median": evaluation_seconds,
        }
    report["results"] = results
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
