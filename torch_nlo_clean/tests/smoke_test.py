"""Small CPU checks for a clean nlo-torch installation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from nlo_torch import (
    F2,
    GBW,
    BKDipole,
    DISConfig,
    DISOrder,
    Quark,
    QuarkType,
    load_bk_table,
    save_bk_table,
)
from nlo_torch.dipole import DipoleTable


def main() -> None:
    dtype = torch.float64
    r = torch.logspace(-4, 1, 16, dtype=dtype)
    y = torch.tensor([0.0, 0.2], dtype=dtype)
    analytic = GBW(Qs0sqr=0.2, x0=0.01)
    amplitude = torch.stack([analytic.dipole_amplitude(r, rapidity.expand_as(r)) for rapidity in y])
    table = DipoleTable(r=r, y=y, N=amplitude, x0=0.01)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "dipole.dat"
        save_bk_table(path, table)
        loaded = load_bk_table(path, dtype=dtype, device="cpu")
    assert torch.allclose(loaded.r, table.r)
    assert torch.allclose(loaded.y, table.y)
    assert torch.allclose(loaded.N, table.N)
    assert loaded.x0 == table.x0

    table_dipole = BKDipole(loaded)
    probe_r = torch.tensor([1e-3, 0.1, 1.0], dtype=dtype)
    probe_y = torch.full_like(probe_r, table_dipole.min_Y)
    values = table_dipole.dipole_amplitude(probe_r, probe_y)
    assert bool(torch.isfinite(values).all())
    assert bool(((values >= 0) & (values <= 1)).all())

    config = DISConfig(
        order=DISOrder.LO,
        quarks=(Quark(QuarkType.C, 1.4),),
        maxr=5.0,
    )
    result = F2(
        torch.tensor(9.0, dtype=dtype),
        torch.tensor(1e-3, dtype=dtype),
        analytic,
        config,
        quadrature_points=12,
        seed=7,
    )
    assert bool(torch.isfinite(result.value))
    assert bool(torch.isfinite(result.error))
    print("nlo-torch smoke test passed")


if __name__ == "__main__":
    main()
