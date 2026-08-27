"""Buildable initial conditions for BK evolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from nlo_torch.numerics.interpolation import NaturalCubicSpline


@dataclass(frozen=True, slots=True)
class MV:
    qs0sqr: float = 0.10
    x0: float = 0.01
    ec: float = 1.0
    lambdaqcd: float = 0.241
    gamma: float = 1.0

    @property
    def min_r(self) -> float:
        return 1e-99

    @property
    def max_r(self) -> float:
        return 1e99

    def dipole_amplitude(self, r: torch.Tensor) -> torch.Tensor:
        exponent = (
            torch.pow(r.square() * self.qs0sqr, self.gamma)
            / 4
            * torch.log(1 / (r * self.lambdaqcd) + self.ec * 2.7182818)
        )
        return torch.where(exponent < 1e-5, exponent, 1 - torch.exp(-exponent))


class ICDataFile:
    """Natural-cubic initial condition loaded from a two-column ``r N`` file."""

    def __init__(
        self,
        path: str | Path,
        *,
        x0: float = 0.01,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        self.path = Path(path)
        self.x0 = x0

        r_values: list[float] = []
        N_values: list[float] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            columns = stripped.split()
            if len(columns) < 2:
                raise ValueError(f"{self.path}:{line_number} requires two columns: r N")
            try:
                r_values.append(float(columns[0]))
                N_values.append(float(columns[1]))
            except ValueError as error:
                raise ValueError(
                    f"{self.path}:{line_number} contains a non-numeric r or N"
                ) from error

        if len(N_values) < 10:
            raise ValueError(
                f"{self.path} contains {len(N_values)} points; at least 10 are required"
            )

        self.r = torch.tensor(r_values, dtype=dtype, device=device)
        self.N = torch.tensor(N_values, dtype=dtype, device=device)
        self._interpolator = NaturalCubicSpline(self.r, self.N)

    @property
    def min_r(self) -> float:
        return float(self.r[0].item())

    @property
    def max_r(self) -> float:
        return float(self.r[-1].item())

    def dipole_amplitude(self, r: torch.Tensor) -> torch.Tensor:
        value = self._interpolator(r)
        value = torch.where(r < self.r[0], torch.zeros_like(value), value)
        return torch.where(r > self.r[-1], torch.ones_like(value), value)


__all__ = ["ICDataFile", "MV"]
