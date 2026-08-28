"""Implemented BK evolution choices and their source defaults."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class RunningCouplingLO(Enum):
    FIXED_LO = auto()
    PARENT_LO = auto()
    SMALLEST_LO = auto()
    BALITSKY_LO = auto()
    FAC_LO = auto()
    BEUF_LO = auto()


class RunningCouplingNLO(Enum):
    FIXED_NLO = auto()
    PARENT_NLO = auto()
    SMALLEST_NLO = auto()


class ResummationCoupling(Enum):
    RESUM_RC_PARENT = auto()
    RESUM_RC_SMALLEST = auto()
    RESUM_RC_FIXED = auto()


class IntegrationMethod(Enum):
    VEGAS = auto()
    MISER = auto()
    MULTIPLE = auto()


class BKOrder(Enum):
    LO = auto()
    LO_RESUM_DLOG = auto()
    LO_RESUM_DLOG_SLOG = auto()
    NLO = auto()
    NLO_RESUM_DLOG = auto()
    NLO_RESUM_DLOG_SLOG = auto()

    @property
    def has_nlo_kernels(self) -> bool:
        return self in {
            BKOrder.NLO,
            BKOrder.NLO_RESUM_DLOG,
            BKOrder.NLO_RESUM_DLOG_SLOG,
        }

    @property
    def resum_dlog(self) -> bool:
        return self in {
            BKOrder.LO_RESUM_DLOG,
            BKOrder.LO_RESUM_DLOG_SLOG,
            BKOrder.NLO_RESUM_DLOG,
            BKOrder.NLO_RESUM_DLOG_SLOG,
        }

    @property
    def resum_slog(self) -> bool:
        return self in {BKOrder.LO_RESUM_DLOG_SLOG, BKOrder.NLO_RESUM_DLOG_SLOG}


@dataclass(frozen=True, slots=True)
class BKConfig:
    NC: float = 3.0
    NF: int = 3
    LambdaQCD: float = 0.241
    RINTPOINTS: int = 85
    THETAINTPOINTS: int = 85
    INTACCURACY: float = 0.001
    MCINTACCURACY: float = 0.2
    MAXR: float = 30.0
    MINR: float = 1e-6
    RPOINTS: int = 100
    MCINTPOINTS: int = 100_000
    DE_SOLVER_STEP: float = 0.2
    DE_SOLVER_ABSERR: float = 1e-6
    DE_SOLVER_RELERR: float = 1e-4
    FIXED_AS: float = 0.2
    RC_LO: RunningCouplingLO = RunningCouplingLO.BALITSKY_LO
    RC_NLO: RunningCouplingNLO = RunningCouplingNLO.SMALLEST_NLO
    RESUM_RC: ResummationCoupling = ResummationCoupling.RESUM_RC_SMALLEST
    INTMETHOD_NLO: IntegrationMethod = IntegrationMethod.VEGAS
    FORCE_POSITIVE_N: bool = True
    SYMMETRIZE_Z_Z2_INTEGRATION: bool = True
    DNDY: bool = False
    Order: BKOrder = BKOrder.NLO_RESUM_DLOG_SLOG
    KSUB: float = 0.65
    KINEMATICAL_CONSTRAINT: bool = False
    EULER_METHOD: bool = False
    C2: float = 1.0
    CUDA_FUSION: bool = True
    K1_FIXED: bool = True
    K1_FIXED_REFINE: bool = False
    # Carry learned importance grids between related K2/Kf calculations.
    VEGAS_REUSE_GRID: bool = True
    # Refresh a reused grid with one quarter of the ordinary warmup.
    VEGAS_REUSE_WARMUP_FRACTION: float = 0.25

    def __post_init__(self) -> None:
        if self.NF not in {0, 3, 5}:
            raise ValueError("NF must be 0, 3, or 5")
        if self.MINR <= 0 or self.MAXR <= self.MINR or self.RPOINTS < 2:
            raise ValueError("the BK r grid requires 0 < MINR < MAXR and RPOINTS >= 2")
        if self.RINTPOINTS < 1 or self.THETAINTPOINTS < 1 or self.MCINTPOINTS < 1:
            raise ValueError("integration limits must be positive")
        if self.INTACCURACY <= 0 or self.MCINTACCURACY <= 0:
            raise ValueError("integration accuracies must be positive")
        if not 0 <= self.VEGAS_REUSE_WARMUP_FRACTION <= 1:
            raise ValueError("VEGAS_REUSE_WARMUP_FRACTION must lie between zero and one")
        if self.DE_SOLVER_STEP <= 0:
            raise ValueError("DE_SOLVER_STEP must be positive")
        if self.C2 <= 0 or self.KSUB <= 0:
            raise ValueError("C2 and KSUB must be positive")

        fixed_LO = self.RC_LO is RunningCouplingLO.FIXED_LO
        fixed_NLO = self.RC_NLO is RunningCouplingNLO.FIXED_NLO
        if fixed_LO != fixed_NLO:
            raise ValueError("fixed coupling must be selected for both RC_LO and RC_NLO")
        if self.KINEMATICAL_CONSTRAINT and not self.EULER_METHOD:
            raise ValueError("KINEMATICAL_CONSTRAINT requires EULER_METHOD")


__all__ = [
    "BKConfig",
    "BKOrder",
    "IntegrationMethod",
    "ResummationCoupling",
    "RunningCouplingLO",
    "RunningCouplingNLO",
]
