"""Command line for BK evolution."""

from __future__ import annotations

import argparse

import torch

from nlo_torch.bk.config import (
    BKConfig,
    BKOrder,
    IntegrationMethod,
    ResummationCoupling,
    RunningCouplingLO,
    RunningCouplingNLO,
)
from nlo_torch.bk.evolution import compute_dndy, solve_bk
from nlo_torch.bk.initial_conditions import MV, ICDataFile
from nlo_torch.dipole.table import save_bk_table


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available on this machine; pass --device cpu")

    initial_condition = (
        ICDataFile(args.ic_data, x0=args.x0, device=device, dtype=dtype)
        if args.ic_data
        else MV(
            qs0sqr=args.qs0sqr,
            x0=args.x0,
            ec=args.ec,
            lambdaqcd=args.lambdaqcd,
            gamma=args.gamma,
        )
    )
    fixed = args.rc_lo == "fixed"
    config = BKConfig(
        NF=args.nf,
        MAXR=args.maxr,
        MINR=args.minr,
        RPOINTS=args.rpoints,
        MCINTPOINTS=args.mcintpoints,
        DE_SOLVER_STEP=args.step,
        FIXED_AS=args.fixed_alpha_s,
        RC_LO=_RC_LO[args.rc_lo],
        RC_NLO=RunningCouplingNLO.FIXED_NLO if fixed else _RC_NLO[args.rc_nlo],
        RESUM_RC=_RESUM_RC[args.resum_rc],
        INTMETHOD_NLO=_INTEGRATION[args.integration],
        FORCE_POSITIVE_N=not args.nolimit,
        DNDY=args.dndy,
        Order=_ORDER[args.order],
        KSUB=args.Ksub,
        KINEMATICAL_CONSTRAINT=args.kinematical_constraint,
        EULER_METHOD=args.euler,
        C2=args.C2,
    )

    if args.dndy:
        result = compute_dndy(initial_condition, config, device=device, dtype=dtype, seed=args.seed)
        print("# r dN/dy[K1] dN/dy[K2+Kf] N")
        rows = torch.stack((result.r, result.K1, result.K2_Kf, result.N), dim=1)
        for row in rows.detach().cpu().tolist():
            print(" ".join(f"{value:.15e}" for value in row))
        return 0

    table = solve_bk(
        initial_condition,
        args.maxy,
        config,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    save_bk_table(args.output, table)
    print(
        f"saved {table.y.numel()} rapidity slices and {table.r.numel()} r points to {args.output}"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evolve a dipole amplitude with the BK equation")
    parser.add_argument("--ic-data", help="two-column r N initial condition")
    parser.add_argument("--x0", type=float, default=0.01)
    parser.add_argument("--qs0sqr", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--ec", type=float, default=1.0)
    parser.add_argument("--lambdaqcd", type=float, default=0.241)
    parser.add_argument("--maxy", type=float, default=10.0)
    parser.add_argument("--output", default="output.dat")
    parser.add_argument("--order", choices=tuple(_ORDER), default="nlo_resum_dlog_slog")
    parser.add_argument("--rc-lo", choices=tuple(_RC_LO), default="balitsky")
    parser.add_argument("--rc-nlo", choices=tuple(_RC_NLO), default="smallest")
    parser.add_argument("--resum-rc", choices=tuple(_RESUM_RC), default="smallest")
    parser.add_argument("--integration", choices=tuple(_INTEGRATION), default="vegas")
    parser.add_argument("--nf", type=int, choices=(0, 3, 5), default=3)
    parser.add_argument("--C2", type=float, default=1.0)
    parser.add_argument("--fixed-alpha-s", type=float, default=0.2)
    parser.add_argument("--Ksub", type=float, default=0.65)
    parser.add_argument("--minr", type=float, default=1e-6)
    parser.add_argument("--maxr", type=float, default=30.0)
    parser.add_argument("--rpoints", type=int, default=100)
    parser.add_argument("--mcintpoints", type=int, default=100_000)
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--euler", action="store_true")
    parser.add_argument("--kinematical-constraint", action="store_true")
    parser.add_argument("--nolimit", action="store_true")
    parser.add_argument("--dndy", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    return parser


_ORDER = {
    "lo": BKOrder.LO,
    "lo_resum_dlog": BKOrder.LO_RESUM_DLOG,
    "lo_resum_dlog_slog": BKOrder.LO_RESUM_DLOG_SLOG,
    "nlo": BKOrder.NLO,
    "nlo_resum_dlog": BKOrder.NLO_RESUM_DLOG,
    "nlo_resum_dlog_slog": BKOrder.NLO_RESUM_DLOG_SLOG,
}
_RC_LO = {
    "fixed": RunningCouplingLO.FIXED_LO,
    "parent": RunningCouplingLO.PARENT_LO,
    "smallest": RunningCouplingLO.SMALLEST_LO,
    "balitsky": RunningCouplingLO.BALITSKY_LO,
    "fac": RunningCouplingLO.FAC_LO,
    "beuf": RunningCouplingLO.BEUF_LO,
}
_RC_NLO = {
    "parent": RunningCouplingNLO.PARENT_NLO,
    "smallest": RunningCouplingNLO.SMALLEST_NLO,
}
_RESUM_RC = {
    "parent": ResummationCoupling.RESUM_RC_PARENT,
    "smallest": ResummationCoupling.RESUM_RC_SMALLEST,
    "fixed": ResummationCoupling.RESUM_RC_FIXED,
}
_INTEGRATION = {
    "vegas": IntegrationMethod.VEGAS,
    "miser": IntegrationMethod.MISER,
    "multiple": IntegrationMethod.MULTIPLE,
}


if __name__ == "__main__":
    raise SystemExit(main())
