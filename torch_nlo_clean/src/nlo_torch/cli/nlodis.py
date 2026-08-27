"""Command line for DIS structure-function grids."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import torch

from nlo_torch.dipole.amplitude import AmplitudeInterpolation, BKDipole
from nlo_torch.dipole.table import load_bk_table
from nlo_torch.dis.config import (
    DISConfig,
    DISOrder,
    Quark,
    QuarkType,
    RunningCouplingIRScheme,
    RunningCouplingScheme,
)
from nlo_torch.dis.observables import F2, FL, FT
from nlo_torch.numerics.integration import IntegralResult

_HERA_FL_POINTS = (
    (1.5, 0.000028),
    (2.0, 0.000043),
    (2.5, 0.000059),
    (3.5, 0.000088),
    (5.0, 0.000129),
    (6.5, 0.000169),
    (8.5, 0.000224),
    (12.0, 0.000319),
    (15.0, 0.000402),
    (20.0, 0.000540),
    (25.0, 0.000687),
    (35.0, 0.000958),
    (45.0, 0.001210),
    (60.0, 0.001570),
    (90.0, 0.002430),
    (120.0, 0.003030),
    (150.0, 0.004020),
    (200.0, 0.005410),
    (250.0, 0.007360),
    (346.0, 0.009860),
)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available on this machine; pass --device cpu")

    table = load_bk_table(args.datafile, dtype=dtype, device=device)
    dipole = BKDipole(table, interpolation=AmplitudeInterpolation.LINEAR_LINEAR)
    light = Quark(QuarkType.LIGHT, args.light_mass)
    charm = Quark(QuarkType.C, args.charm_mass)
    config = DISConfig(
        order=DISOrder[args.order],
        rc_scheme=RunningCouplingScheme[args.rc_scheme],
        rc_ir_scheme=RunningCouplingIRScheme.SMOOTH,
        maxr=args.maxr,
        C2_alpha=args.C2,
        nf_alphas=args.nf,
        transverse_area=args.proton_area * 2.56819,
        maxeval=args.mcintpoints,
        epsrel=args.epsrel,
        cuda_nested_points=args.cuda_nested_points,
        quarks=(light, charm),
    )
    integration_options = {
        "quadrature_points": args.quadrature_points,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }

    if args.runmode == "F2FL_GRID":
        if not args.no_header:
            print("fit,x,Q2,F2 light,FL light,F2 charm,FL charm,F2_LO")
        _run_F2FL_grid(args.name, dipole, config, light, charm, integration_options)
    else:
        if not args.no_header:
            print("fit,x,Q2,FL")
        for Q2_value, xbj_value in _HERA_FL_POINTS:
            Q2 = torch.tensor(Q2_value, dtype=dtype, device=device)
            xbj = torch.tensor(xbj_value, dtype=dtype, device=device)
            result = FL(Q2, xbj, dipole, config, **integration_options)
            print(f"{args.name},{xbj_value:.15g},{Q2_value:.15g},{result.value.item():.15e}")
            _report_integration("FL", Q2_value, xbj_value, result)
    return 0


def _run_F2FL_grid(
    name: str,
    dipole: BKDipole,
    config: DISConfig,
    light: Quark,
    charm: Quark,
    integration_options: dict[str, int],
) -> None:
    dtype = dipole.table.r.dtype
    device = dipole.table.r.device
    for Q2_value in (4.5, 45.0, 100.0):
        xbj_value = 1e-5
        while xbj_value <= 1e-2:
            Q2 = torch.tensor(Q2_value, dtype=dtype, device=device)
            xbj = torch.tensor(xbj_value, dtype=dtype, device=device)
            light_config = replace(config, quarks=(light,))
            charm_config = replace(config, quarks=(charm,))
            FT_light = FT(Q2, xbj, dipole, light_config, **integration_options)
            FL_light = FL(Q2, xbj, dipole, light_config, **integration_options)
            FT_charm = FT(Q2, xbj, dipole, charm_config, **integration_options)
            FL_charm = FL(Q2, xbj, dipole, charm_config, **integration_options)
            F2_light = FT_light.value + FL_light.value
            F2_charm = FT_charm.value + FL_charm.value
            LO_config = replace(config, order=DISOrder.LO, quarks=(light, charm))
            F2_LO = F2(Q2, xbj, dipole, LO_config, **integration_options)
            print(
                f"{name},{xbj_value:.15g},{Q2_value:.15g},"
                f"{F2_light.item():.15e},{FL_light.value.item():.15e},"
                f"{F2_charm.item():.15e},{FL_charm.value.item():.15e},"
                f"{F2_LO.value.item():.15e}"
            )
            _report_integration("FT_light", Q2_value, xbj_value, FT_light)
            _report_integration("FL_light", Q2_value, xbj_value, FL_light)
            _report_integration("FT_charm", Q2_value, xbj_value, FT_charm)
            _report_integration("FL_charm", Q2_value, xbj_value, FL_charm)
            xbj_value *= 1.5


def _report_integration(contribution: str, Q2: float, xbj: float, result: IntegralResult) -> None:
    print(
        f"# {contribution} Q2={Q2:.15g} x={xbj:.15g} seed={result.seed} "
        f"n_eval={result.n_eval} stderr={result.error.item():.8e} "
        f"converged={result.converged}",
        file=sys.stderr,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate DIS structure-function grids")
    parser.add_argument("--name", default="nlodis")
    parser.add_argument("--datafile", required=True)
    parser.add_argument("--C2", type=float, required=True)
    parser.add_argument("--charm-mass", type=float, required=True)
    parser.add_argument("--light-mass", type=float, default=0.01)
    parser.add_argument("--proton-area", type=float, required=True, help="sigma_0/2 in mb")
    parser.add_argument("--rc-scheme", choices=("PARENT", "SMALLEST"), required=True)
    parser.add_argument("--order", choices=("LO", "NLO"), default="NLO")
    parser.add_argument("--runmode", choices=("F2FL_GRID", "HERA_FL"), default="HERA_FL")
    parser.add_argument("--epsrel", type=float, default=0.01)
    parser.add_argument("--mcintpoints", type=int, default=1_000_000)
    parser.add_argument("--nf", type=int, default=-1)
    parser.add_argument("--maxr", type=float, default=30.0)
    parser.add_argument("--quadrature-points", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--cuda-nested-points", type=int, default=48)
    parser.add_argument("--no-header", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
