"""Constants, quarks, and fixed choices for DIS observables."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

LambdaQCD = 0.241
AlphaEM = 1 / 137.035999
NC = 3
CF = (NC**2 - 1) / (2 * NC)
Q0sqr = 1.0


class Polarization(Enum):
    T = auto()
    L = auto()


class DISOrder(Enum):
    LO = auto()
    NLO = auto()


class NcScheme(Enum):
    FiniteNC = auto()
    LargeNC = auto()


class RunningCouplingScheme(Enum):
    SMALLEST = auto()
    PARENT = auto()


class RunningCouplingIRScheme(Enum):
    FREEZE = auto()
    SMOOTH = auto()


class HeavyQuarkX(Enum):
    MassDependentX = auto()
    MassIndependentX = auto()


class QuarkType(Enum):
    U = auto()
    D = auto()
    S = auto()
    C = auto()
    B = auto()
    T = auto()
    LIGHT = auto()


_TYPE_NOT_GIVEN = object()


@dataclass(frozen=True, slots=True, init=False)
class Quark:
    type: QuarkType
    mass: float

    def __init__(
        self,
        type: QuarkType | object = _TYPE_NOT_GIVEN,
        mass: float | None = None,
    ) -> None:
        default_constructor = type is _TYPE_NOT_GIVEN
        quark_type = QuarkType.LIGHT if default_constructor else type
        if not isinstance(quark_type, QuarkType):
            raise TypeError("type must be a QuarkType")
        if mass is None:
            if default_constructor:
                mass = 0.001
            elif quark_type in {QuarkType.U, QuarkType.D, QuarkType.S, QuarkType.LIGHT}:
                mass = 0.002
            elif quark_type is QuarkType.C:
                mass = 1.4
            elif quark_type is QuarkType.B:
                mass = 4.18
            else:
                mass = 173.0
        object.__setattr__(self, "type", quark_type)
        object.__setattr__(self, "mass", float(mass))

    @property
    def charge(self) -> float:
        if self.type in {QuarkType.U, QuarkType.C}:
            return 2 / 3
        if self.type in {QuarkType.D, QuarkType.S, QuarkType.B}:
            return -1 / 3
        if self.type is QuarkType.LIGHT:
            return math.sqrt(2 / 3)
        raise ValueError("the source does not define an electric charge for the top quark")


@dataclass(frozen=True, slots=True)
class DISConfig:
    order: DISOrder = DISOrder.LO
    nc_scheme: NcScheme = NcScheme.FiniteNC
    rc_scheme: RunningCouplingScheme = RunningCouplingScheme.SMALLEST
    rc_ir_scheme: RunningCouplingIRScheme = RunningCouplingIRScheme.FREEZE
    maxr: float = 30.0
    C2_alpha: float = 1.0
    nf_alphas: int = -1
    max_alpha_s_freeze: float = 0.7
    heavy_quark_x_scheme: HeavyQuarkX = HeavyQuarkX.MassIndependentX
    transverse_area: float = 1.0
    maxeval: int = 2_000_000
    epsrel: float = 0.001
    cuda_fusion: bool = True
    cuda_nested: bool = True
    cuda_nested_points: int = 48
    quarks: tuple[Quark, ...] = field(
        default_factory=lambda: (Quark(QuarkType.LIGHT), Quark(QuarkType.C))
    )

    def __post_init__(self) -> None:
        if self.maxr <= 0 or self.C2_alpha <= 0:
            raise ValueError("maxr and C2_alpha must be positive")
        if self.max_alpha_s_freeze <= 0 or self.transverse_area <= 0:
            raise ValueError("coupling freeze and transverse area must be positive")
        if self.maxeval < 1 or self.epsrel <= 0:
            raise ValueError("integration budget and relative accuracy must be positive")
        if not 8 <= self.cuda_nested_points <= 128:
            raise ValueError("nested DIS integration requires 8 to 128 inner points")
        if not self.quarks:
            raise ValueError("at least one quark is required")

    @property
    def active_flavors(self) -> int:
        if self.nf_alphas >= 0:
            return self.nf_alphas
        return sum(3 if quark.type is QuarkType.LIGHT else 1 for quark in self.quarks)


__all__ = [
    "AlphaEM",
    "CF",
    "DISConfig",
    "DISOrder",
    "HeavyQuarkX",
    "LambdaQCD",
    "NC",
    "NcScheme",
    "Polarization",
    "Q0sqr",
    "Quark",
    "QuarkType",
    "RunningCouplingIRScheme",
    "RunningCouplingScheme",
]
