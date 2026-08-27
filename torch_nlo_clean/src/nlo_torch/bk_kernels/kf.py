"""Pointwise fermion-loop BK kernel Kf."""

from __future__ import annotations

import torch

from nlo_torch.bk.config import BKConfig


def Kernel_nlo_fermion(
    r: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    X2: torch.Tensor,
    Y2: torch.Tensor,
    z_m_z2: torch.Tensor,
    config: BKConfig,
) -> torch.Tensor:
    """Evaluate ``Kernel_nlo_fermion(r, X, Y, X2, Y2, z_m_z2)`` elementwise."""

    r, X, Y, X2, Y2, z_m_z2 = torch.broadcast_tensors(r, X, Y, X2, Y2, z_m_z2)
    invalid_distance = (X < 1e-20) | (X2 < 1e-20) | (Y < 1e-20) | (Y2 < 1e-20) | (z_m_z2 < 1e-20)
    safe_X = torch.where(invalid_distance, torch.ones_like(X), X)
    safe_Y = torch.where(invalid_distance, torch.ones_like(Y), Y)
    safe_X2 = torch.where(invalid_distance, torch.ones_like(X2), X2)
    safe_Y2 = torch.where(invalid_distance, torch.ones_like(Y2), Y2)
    safe_z_m_z2 = torch.where(invalid_distance, torch.ones_like(z_m_z2), z_m_z2)

    XY2sq = (safe_X * safe_Y2).square()
    X2Ysq = (safe_X2 * safe_Y).square()
    kernel = 2 / safe_z_m_z2.pow(4)
    kernel = kernel - (
        (XY2sq + X2Ysq - (r * safe_z_m_z2).square())
        / (safe_z_m_z2.pow(4) * (XY2sq - X2Ysq))
        * 2
        * torch.log(safe_X * safe_Y2 / (safe_X2 * safe_Y))
    )
    kernel = kernel * config.NF / config.NC

    valid = ~invalid_distance & torch.isfinite(kernel)
    return torch.where(valid, kernel, torch.zeros_like(kernel))


__all__ = ["Kernel_nlo_fermion"]
