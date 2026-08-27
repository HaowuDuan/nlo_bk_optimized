"""Concrete GBW and BK-table dipole amplitudes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import torch

from nlo_torch.dipole.table import DipoleTable
from nlo_torch.numerics.interpolation import NaturalCubicSpline


class AmplitudeInterpolation(Enum):
    SPLINE_LINEAR = auto()
    LINEAR_LINEAR = auto()


@dataclass(frozen=True, slots=True)
class GBW:
    Qs0sqr: float = 0.1
    lambda_: float = 0.3
    gamma: float = 1.0
    x0: float = 1.0

    @property
    def min_r(self) -> float:
        return 1e-30

    @property
    def max_r(self) -> float:
        return 1e3

    def X0(self) -> float:
        return self.x0

    def dipole_amplitude(self, r: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        Y0 = Y.new_tensor(math.log(1 / self.x0))
        Yeff = torch.maximum(Y, Y0)
        Qs2_Y = self.Qs0sqr * torch.exp(self.lambda_ * Yeff)
        arg = torch.pow(r.square() * Qs2_Y, self.gamma) / 4
        return torch.where(torch.abs(arg) < 1e-7, arg, 1 - torch.exp(-arg))

    def saturation_scale(self, Y: torch.Tensor, Ns: float) -> torch.Tensor:
        return _saturation_scale(self, Y, Ns)


class BKDipole:
    def __init__(
        self,
        table: DipoleTable,
        interpolation: AmplitudeInterpolation = AmplitudeInterpolation.SPLINE_LINEAR,
    ) -> None:
        if table.r.numel() < 9:
            raise ValueError("BKDipole requires at least nine r points")

        r_multiplier = table.r[1] / table.r[0]
        expected_r = table.r[0] * torch.pow(
            r_multiplier,
            torch.arange(table.r.numel(), dtype=table.r.dtype, device=table.r.device),
        )
        relative_grid_error = torch.abs(expected_r / table.r - 1)
        if not bool((relative_grid_error <= 0.01).all()):
            raise ValueError("BKDipole requires a logarithmic r grid")

        Y0 = max(0.0, math.log(1 / table.x0))
        self.table = table
        self.Y = table.y + table.y.new_tensor(Y0)
        self.interpolation = interpolation
        self._interpolator: NaturalCubicSpline | None = None
        self._interpolator_Y = -1.0
        self._maxr_interpolate = -1.0

    @property
    def min_r(self) -> float:
        return float(self.table.r[0].item())

    @property
    def max_r(self) -> float:
        return float(self.table.r[-1].item())

    @property
    def min_Y(self) -> float:
        return float(self.Y[0].item())

    @property
    def max_Y(self) -> float:
        return float(self.Y[-1].item())

    @property
    def maxr_interpolate(self) -> float:
        return self._maxr_interpolate

    def X0(self) -> float:
        return 1.0

    def interpolator_initialized(self, Y: torch.Tensor) -> torch.Tensor:
        near_zero = (torch.abs(Y) < 0.001) & (abs(self._interpolator_Y) < 0.001)
        denominator = torch.minimum(Y, Y.new_tensor(self._interpolator_Y))
        safe_denominator = torch.where(denominator != 0, denominator, torch.ones_like(Y))
        relative_match = (self._interpolator_Y >= 0) & (
            torch.abs(Y - self._interpolator_Y) / safe_denominator < 0.001
        )
        return near_zero | relative_match

    def initialize_interpolation(self, Y: float | torch.Tensor) -> None:
        Y_value = float(Y.item()) if isinstance(Y, torch.Tensor) else float(Y)
        if 0 <= Y_value <= self.min_Y:
            Y_value = self.min_Y
        if Y_value < 0:
            raise ValueError(f"cannot initialize interpolation at negative Y={Y_value}")
        if Y_value > self.max_Y:
            raise ValueError(
                f"cannot initialize interpolation at Y={Y_value}; maximum is {self.max_Y}"
            )

        Y_tensor = self.table.y.new_tensor(Y_value)
        if self._interpolator is not None and bool(self.interpolator_initialized(Y_tensor)):
            return

        self._interpolator = None
        self._interpolator_Y = -1.0
        self._maxr_interpolate = -1.0

        r = self.table.r.clone()
        evaluation_r = r.clone()
        evaluation_r[0] = evaluation_r[0] * 1.0001
        evaluation_r[-1] = evaluation_r[-1] * 0.9999
        evaluation_Y = torch.full_like(evaluation_r, Y_value)
        N = self.dipole_amplitude(evaluation_r, evaluation_Y)

        self._interpolator = NaturalCubicSpline(r, N)
        self._interpolator_Y = Y_value
        self._find_maxr_interpolate(Y_value)

    def dipole_amplitude(self, r: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        r, Y = torch.broadcast_tensors(r, Y)
        if r.device != self.table.r.device or r.dtype != self.table.r.dtype:
            raise ValueError("r and Y must have the BK table device and dtype")
        if not bool(torch.isfinite(r).all()) or not bool(torch.isfinite(Y).all()):
            raise ValueError("r and Y must contain only finite values")

        min_r = self.table.r[0]
        max_r = self.table.r[-1]
        r = torch.where(r < min_r, min_r * 1.000001, r)
        r = torch.where(r > max_r, max_r * 0.999999, r)

        min_Y = self.Y[0]
        max_Y = self.Y[-1]
        Y = torch.where((Y >= 0) & (Y < min_Y), min_Y, Y)
        if bool(((Y < min_Y) | (Y > max_Y)).any()):
            raise ValueError(
                f"Y must lie in [{self.min_Y}, {self.max_Y}] after initial-rapidity freezing"
            )

        initialized = self.interpolator_initialized(Y)
        if self._interpolator is not None and bool(initialized.all()):
            return self._initialized_amplitude(r)
        if self._interpolator is None or not bool(initialized.any()):
            return self._uninitialized_amplitude(r, Y)

        initialized_value = self._initialized_amplitude(r)
        uninitialized_value = self._uninitialized_amplitude(r, Y)
        return torch.where(initialized, initialized_value, uninitialized_value)

    def saturation_scale(self, Y: torch.Tensor, Ns: float) -> torch.Tensor:
        return _saturation_scale(self, Y, Ns)

    def _initialized_amplitude(self, r: torch.Tensor) -> torch.Tensor:
        if self._interpolator is None:
            raise RuntimeError("interpolation has not been initialized")
        value = self._interpolator(r)
        if self._maxr_interpolate > 0:
            value = torch.where(r >= self._maxr_interpolate, torch.ones_like(value), value)
        return value.clamp(0, 1)

    def _uninitialized_amplitude(self, r: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        shape = r.shape
        r = r.reshape(-1)
        Y = Y.reshape(-1)
        if self.interpolation is AmplitudeInterpolation.LINEAR_LINEAR:
            value = self._bilinear_amplitude(r, Y)
        else:
            value = self._local_spline_amplitude(r, Y)
        return value.reshape(shape).clamp(0, 1)

    def _bilinear_amplitude(self, r: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        rind = torch.searchsorted(self.table.r, r, right=True) - 1
        yind = torch.searchsorted(self.Y, Y, right=True) - 1
        rind2 = (rind + 1).clamp(max=self.table.r.numel() - 1)
        yind2 = (yind + 1).clamp(max=self.Y.numel() - 1)

        r1 = self.table.r[rind]
        r2 = self.table.r[rind2]
        Y1 = self.Y[yind]
        Y2 = self.Y[yind2]
        safe_dr = torch.where(rind2 != rind, r2 - r1, torch.ones_like(r))
        safe_dY = torch.where(yind2 != yind, Y2 - Y1, torch.ones_like(Y))

        N11 = self.table.N[yind, rind]
        N12 = self.table.N[yind, rind2]
        N21 = self.table.N[yind2, rind]
        N22 = self.table.N[yind2, rind2]
        value = (
            N11 * (r2 - r) * (Y2 - Y)
            + N12 * (r - r1) * (Y2 - Y)
            + N21 * (r2 - r) * (Y - Y1)
            + N22 * (r - r1) * (Y - Y1)
        ) / (safe_dr * safe_dY)
        value_at_last_Y = N11 + (r - r1) / safe_dr * (N12 - N11)
        value = torch.where(yind2 == yind, value_at_last_Y, value)
        return torch.where(rind2 == rind, torch.ones_like(value), value)

    def _local_spline_amplitude(self, r: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        rind = torch.searchsorted(self.table.r, r, right=True) - 1
        yind = torch.searchsorted(self.Y, Y, right=True) - 1
        upper_window = rind + 3 > self.table.r.numel() - 1

        value = torch.empty_like(r)
        regular = ~upper_window
        if bool(regular.any()):
            start = torch.where(rind[regular] - 3 < 0, 0, rind[regular] - 3)
            value[regular] = self._evaluate_local_splines(
                r[regular], Y[regular], yind[regular], start, 7
            )
        if bool(upper_window.any()):
            start = torch.full_like(rind[upper_window], self.table.r.numel() - 9)
            value[upper_window] = self._evaluate_local_splines(
                r[upper_window], Y[upper_window], yind[upper_window], start, 9
            )
        return value

    def _evaluate_local_splines(
        self,
        r: torch.Tensor,
        Y: torch.Tensor,
        yind: torch.Tensor,
        start: torch.Tensor,
        points: int,
    ) -> torch.Tensor:
        rind = start.unsqueeze(-1) + torch.arange(points, device=start.device, dtype=start.dtype)
        yind2 = (yind + 1).clamp(max=self.Y.numel() - 1)
        N_lower = self.table.N[yind.unsqueeze(-1), rind]
        N_upper = self.table.N[yind2.unsqueeze(-1), rind]
        Y_lower = self.Y[yind]
        Y_upper = self.Y[yind2]
        safe_dY = torch.where(yind2 != yind, Y_upper - Y_lower, torch.ones_like(Y))
        fraction = torch.where(yind2 != yind, (Y - Y_lower) / safe_dY, 0)
        N = N_lower + fraction.unsqueeze(-1) * (N_upper - N_lower)
        r_nodes = self.table.r[rind]
        return NaturalCubicSpline(r_nodes, N)(r)

    def _find_maxr_interpolate(self, Y: float) -> None:
        step = 2.0
        previous_r = 0.1
        r = 0.01
        for _ in range(41):
            if r > self.max_r:
                break
            value = self.dipole_amplitude(self.table.r.new_tensor(r), self.table.y.new_tensor(Y))
            if float(value.item()) >= 0.99999:
                if step < 1e-2:
                    self._maxr_interpolate = r
                    return
                step /= 1.5
                r = previous_r
            previous_r = r
            r += step
        self._maxr_interpolate = -1.0


def _saturation_scale(dipole: GBW | BKDipole, Y: torch.Tensor, Ns: float) -> torch.Tensor:
    if Y.numel() != 1:
        raise ValueError("saturation_scale requires a scalar Y")

    lower = Y.new_tensor(dipole.min_r * 1.0001)
    upper = Y.new_tensor(dipole.max_r * 0.999)
    Ns_tensor = Y.new_tensor(Ns)
    lower_value = dipole.dipole_amplitude(lower, Y) - Ns_tensor
    upper_value = dipole.dipole_amplitude(upper, Y) - Ns_tensor
    if bool(lower_value * upper_value > 0):
        raise ValueError("the saturation-scale bracket does not contain a root")

    root = (lower + upper) / 2
    for iteration in range(1, 1001):
        root = (lower + upper) / 2
        value = dipole.dipole_amplitude(root, Y) - Ns_tensor
        if bool(lower_value * value <= 0):
            upper = root
        else:
            lower = root
            lower_value = value

        tolerance = 1e-5 * torch.minimum(torch.abs(lower), torch.abs(upper))
        if bool(torch.abs(upper - lower) < tolerance) and iteration < 1000:
            return 2 / root.square()

    raise RuntimeError(f"saturation-scale root finding failed at Y={Y.item()}")


__all__ = ["AmplitudeInterpolation", "BKDipole", "GBW"]
