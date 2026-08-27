"""One-dimensional cubic interpolation used by both calculations."""

from __future__ import annotations

import torch

MIN_LOG_VALUE = -40.0


class NaturalCubicSpline:
    """Natural cubic splines whose interpolation points occupy the last axis."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.ndim < 1 or y.ndim < 1:
            raise ValueError("x and y must have at least one dimension")
        if x.shape != y.shape:
            raise ValueError("x and y must have equal shapes")
        if x.shape[-1] < 3:
            raise ValueError("a cubic spline requires at least three points")
        if x.device != y.device or x.dtype != y.dtype:
            raise ValueError("x and y must have the same device and dtype")
        if not x.is_floating_point():
            raise TypeError("x and y must be floating-point tensors")
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
            raise ValueError("x and y must contain only finite values")

        h = x[..., 1:] - x[..., :-1]
        if not bool((h > 0).all()):
            raise ValueError("x must be strictly increasing")

        diagonal = 2 * (h[..., :-1] + h[..., 1:])
        off_diagonal = h[..., 1:-1]
        right_hand_side = 6 * (
            (y[..., 2:] - y[..., 1:-1]) / h[..., 1:] - (y[..., 1:-1] - y[..., :-2]) / h[..., :-1]
        )
        interior = _solve_symmetric_tridiagonal(off_diagonal, diagonal, right_hand_side)
        endpoint_shape = (*y.shape[:-1], 1)
        second_derivative = torch.cat(
            (y.new_zeros(endpoint_shape), interior, y.new_zeros(endpoint_shape)), dim=-1
        )

        self.x = x
        self.a = y[..., :-1]
        self.b = (y[..., 1:] - y[..., :-1]) / h - h * (
            2 * second_derivative[..., :-1] + second_derivative[..., 1:]
        ) / 6
        self.c = second_derivative[..., :-1] / 2
        self.d = (second_derivative[..., 1:] - second_derivative[..., :-1]) / (6 * h)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.x.device or x.dtype != self.x.dtype:
            raise ValueError("evaluation points must have the spline device and dtype")

        if self.x.ndim == 1:
            interval = torch.searchsorted(self.x, x, right=True) - 1
            interval = interval.clamp(0, self.x.shape[-1] - 2)
            dx = x - self.x[interval]
            return self.a[interval] + dx * (
                self.b[interval] + dx * (self.c[interval] + dx * self.d[interval])
            )

        if x.shape != self.x.shape[:-1]:
            raise ValueError("batched evaluation points must match the spline batch shape")
        interval = torch.searchsorted(self.x.contiguous(), x.unsqueeze(-1), right=True)
        interval = interval.squeeze(-1).sub(1).clamp(0, self.x.shape[-1] - 2)
        gather_index = interval.unsqueeze(-1)
        x_left = torch.take_along_dim(self.x, gather_index, dim=-1).squeeze(-1)
        a = torch.take_along_dim(self.a, gather_index, dim=-1).squeeze(-1)
        b = torch.take_along_dim(self.b, gather_index, dim=-1).squeeze(-1)
        c = torch.take_along_dim(self.c, gather_index, dim=-1).squeeze(-1)
        d = torch.take_along_dim(self.d, gather_index, dim=-1).squeeze(-1)
        dx = x - x_left
        return a + dx * (b + dx * (c + dx * d))

    @classmethod
    def from_coefficients(
        cls,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
    ) -> NaturalCubicSpline:
        """Construct a spline from coefficients produced by a validated backend."""

        spline = cls.__new__(cls)
        spline.x = x
        spline.a = a
        spline.b = b
        spline.c = c
        spline.d = d
        return spline


class LogLogSpline:
    """Natural cubic interpolation of ``log(y)`` as a function of ``log(x)``."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if not bool((x > 0).all()):
            raise ValueError("logarithmic interpolation requires x > 0")

        log_y = torch.where(y > 0, torch.log(y), y.new_tensor(MIN_LOG_VALUE))
        self._spline = NaturalCubicSpline(torch.log(x), log_y)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self._spline(torch.log(x)))

    @classmethod
    def from_coefficients(
        cls,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
    ) -> LogLogSpline:
        """Construct a log-log spline from precomputed natural-cubic coefficients."""

        spline = cls.__new__(cls)
        spline._spline = NaturalCubicSpline.from_coefficients(x, a, b, c, d)
        return spline


def _solve_symmetric_tridiagonal(
    off_diagonal: torch.Tensor,
    diagonal: torch.Tensor,
    right_hand_side: torch.Tensor,
) -> torch.Tensor:
    """Solve the natural-spline tridiagonal system with the Thomas algorithm."""

    points = diagonal.shape[-1]
    if points == 1:
        return right_hand_side / diagonal

    upper: list[torch.Tensor] = [off_diagonal[..., 0] / diagonal[..., 0]]
    solution: list[torch.Tensor] = [right_hand_side[..., 0] / diagonal[..., 0]]

    for i in range(1, points):
        denominator = diagonal[..., i] - off_diagonal[..., i - 1] * upper[i - 1]
        if i < points - 1:
            upper.append(off_diagonal[..., i] / denominator)
        solution.append(
            (right_hand_side[..., i] - off_diagonal[..., i - 1] * solution[i - 1]) / denominator
        )

    for i in range(points - 2, -1, -1):
        solution[i] = solution[i] - upper[i] * solution[i + 1]

    return torch.stack(solution, dim=-1)


__all__ = ["LogLogSpline", "MIN_LOG_VALUE", "NaturalCubicSpline"]
