"""Compiled DIS I2 expression."""

from __future__ import annotations

import torch

from nlo_torch.dis.transverse import ITNLOqg_massive_tripole_part_I2

_compiled_i2 = torch.compile(
    ITNLOqg_massive_tripole_part_I2,
    fullgraph=True,
    dynamic=True,
)


def ITNLOqg_massive_tripole_part_I2_fused(
    Q2: torch.Tensor,
    mf: float,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x01sq: torch.Tensor,
    x02sq: torch.Tensor,
    x21sq: torch.Tensor,
    y_t1: torch.Tensor,
) -> torch.Tensor:
    arguments = (Q2, mf, z1, z2, x01sq, x02sq, x21sq, y_t1)
    if not Q2.is_cuda:
        return ITNLOqg_massive_tripole_part_I2(*arguments)
    return _compiled_i2(*arguments)
