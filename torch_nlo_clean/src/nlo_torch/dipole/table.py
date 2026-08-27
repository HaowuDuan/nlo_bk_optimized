"""Shared in-memory dipole table and legacy text-table I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True, slots=True)
class DipoleTable:
    """BK amplitudes ``N(y, r)`` stored at evolution rapidities ``y``."""

    r: torch.Tensor
    y: torch.Tensor
    N: torch.Tensor
    x0: float

    def __post_init__(self) -> None:
        if self.r.ndim != 1 or self.y.ndim != 1 or self.N.ndim != 2:
            raise ValueError("r and y must be one-dimensional and N must be two-dimensional")
        if self.r.numel() < 2 or self.y.numel() < 1:
            raise ValueError("a dipole table requires at least two r points and one y point")
        if self.N.shape != (self.y.numel(), self.r.numel()):
            raise ValueError("N must have shape (len(y), len(r))")
        if self.r.device != self.y.device or self.r.device != self.N.device:
            raise ValueError("r, y, and N must be on the same device")
        if self.r.dtype != self.y.dtype or self.r.dtype != self.N.dtype:
            raise ValueError("r, y, and N must have the same dtype")
        if not self.r.is_floating_point():
            raise TypeError("r, y, and N must be floating-point tensors")
        if not bool(torch.isfinite(self.r).all()):
            raise ValueError("r must contain only finite values")
        if not bool(torch.isfinite(self.y).all()):
            raise ValueError("y must contain only finite values")
        if not bool(torch.isfinite(self.N).all()):
            raise ValueError("N must contain only finite values")
        if not bool((self.r > 0).all()) or not bool((self.r[1:] > self.r[:-1]).all()):
            raise ValueError("r must be positive and strictly increasing")
        if self.y.numel() > 1 and not bool((self.y[1:] > self.y[:-1]).all()):
            raise ValueError("y must be strictly increasing")
        if not math.isfinite(self.x0) or self.x0 <= 0:
            raise ValueError("x0 must be positive and finite")


def load_bk_table(
    path: str | Path,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> DipoleTable:
    """Load the four-header BK text format without changing stored rapidities."""

    filename = Path(path)
    lines = filename.read_text(encoding="utf-8").splitlines()

    configuration: list[str] = []
    data_start = 0
    for line_number, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("###"):
            configuration.append(stripped[3:].strip())
            if len(configuration) == 4:
                data_start = line_number + 1
                break

    if len(configuration) != 4:
        raise ValueError(f"{filename} does not contain four BK configuration headers")

    try:
        minr = float(configuration[0])
        r_multiplier = float(configuration[1])
        rpoints = int(configuration[2])
        x0 = float(configuration[3])
    except ValueError as error:
        raise ValueError(f"{filename} has an invalid BK configuration header") from error

    if minr <= 0 or r_multiplier <= 1 or rpoints < 2 or x0 <= 0:
        raise ValueError(f"{filename} has invalid BK grid metadata")

    y_values: list[float] = []
    rows: list[list[float]] = []
    row: list[float] | None = None

    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped or (stripped.startswith("#") and not stripped.startswith("###")):
            continue
        if stripped.startswith("###"):
            if row is not None:
                _append_row(filename, rows, row, rpoints)
            try:
                y_values.append(float(stripped[3:].strip()))
            except ValueError as error:
                raise ValueError(f"{filename} has an invalid rapidity header") from error
            row = []
            continue
        if row is None:
            raise ValueError(f"{filename} contains amplitude data before a rapidity header")
        try:
            row.append(float(stripped))
        except ValueError as error:
            raise ValueError(f"{filename} contains an invalid amplitude value") from error

    if row is not None:
        _append_row(filename, rows, row, rpoints)
    if not rows or len(rows) != len(y_values):
        raise ValueError(f"{filename} contains no complete rapidity blocks")

    r = minr * torch.pow(
        torch.tensor(r_multiplier, dtype=dtype, device=device),
        torch.arange(rpoints, dtype=dtype, device=device),
    )
    y = torch.tensor(y_values, dtype=dtype, device=device)
    N = torch.tensor(rows, dtype=dtype, device=device)
    return DipoleTable(r=r, y=y, N=N, x0=x0)


def save_bk_table(path: str | Path, table: DipoleTable) -> None:
    """Write evolution rapidities and the table's actual ``x0``."""

    r = table.r.detach().cpu()
    y = table.y.detach().cpu()
    N = table.N.detach().cpu()
    r_multiplier = r[1] / r[0]
    expected_r = r[0] * torch.pow(r_multiplier, torch.arange(r.numel(), dtype=r.dtype))
    tolerance = 5e-6 if r.dtype == torch.float32 else 2e-12
    if not bool(torch.allclose(r, expected_r, rtol=tolerance, atol=0)):
        raise ValueError("the legacy BK format requires a logarithmically uniform r grid")

    output = [
        "# nlo_torch BK dipole table",
        f"###{r[0].item():.15e}",
        f"###{r_multiplier.item():.15e}",
        f"###{r.numel()}",
        f"###{table.x0:.15e}",
    ]
    for yind in range(y.numel()):
        output.append(f"###{y[yind].item():.15e}")
        output.extend(f"{value:.15e}" for value in N[yind].tolist())

    Path(path).write_text("\n".join(output) + "\n", encoding="utf-8")


def _append_row(filename: Path, rows: list[list[float]], row: list[float], rpoints: int) -> None:
    if len(row) != rpoints:
        raise ValueError(f"{filename} has {len(row)} amplitudes in a block; expected {rpoints}")
    rows.append(row)


__all__ = ["DipoleTable", "load_bk_table", "save_bk_table"]
