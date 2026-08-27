"""Coordinate-space running couplings used by BK evolution and DIS."""

from __future__ import annotations

import math

import torch


def bk_alpha_s(
    r: torch.Tensor,
    *,
    NC: float = 3.0,
    NF: int = 3,
    LambdaQCD: float = 0.241,
    C2: float = 1.0,
    fixed: bool = False,
    fixed_alpha_s: float = 0.2,
) -> torch.Tensor:
    """The coupling in ``BKSolver::Alphas``, including heavy-flavor thresholds."""

    if fixed:
        return torch.full_like(r, fixed_alpha_s)

    if NF <= 3:
        return _smooth_alpha_s(r, NC=NC, Nf=NF, LambdaQCD=LambdaQCD, C2=C2)

    dipolescale = 4 * C2 / r.square()
    Nf = torch.where(
        dipolescale < 1.3**2,
        torch.full_like(r, 3.0),
        torch.where(
            dipolescale < 4.5**2,
            torch.full_like(r, 4.0),
            torch.full_like(r, 5.0),
        ),
    )
    Lambda = torch.where(
        Nf == 3,
        torch.full_like(r, 0.146159),
        torch.where(Nf == 4, torch.full_like(r, 0.122944), torch.full_like(r, 0.0904389)),
    )
    b0 = 11 - 2 * Nf / 3
    log_arg = 4 * C2 / (r.square() * Lambda.square())
    alpha_s = 4 * math.pi / (b0 * torch.log(log_arg))
    return torch.where((log_arg < 1) | (alpha_s > 1), torch.ones_like(r), alpha_s)


def dis_alpha_s_freeze(
    r: torch.Tensor,
    *,
    Nf: int,
    C2: float = 1.0,
    LambdaQCD: float = 0.241,
    NC: float = 3.0,
    max_alpha_s: float = 0.7,
) -> torch.Tensor:
    """DIS coordinate-space coupling with the source's hard infrared cap."""

    b0 = (11 * NC - 2 * Nf) / (12 * math.pi)
    log_arg = 4 * C2 / (r.square() * LambdaQCD**2)
    alpha_s = 1 / (b0 * torch.log(log_arg))
    return torch.where(
        (alpha_s > max_alpha_s) | (log_arg < 1),
        torch.full_like(r, max_alpha_s),
        alpha_s,
    )


def dis_alpha_s_smooth(
    r: torch.Tensor,
    *,
    Nf: int,
    C2: float = 1.0,
    LambdaQCD: float = 0.241,
    NC: float = 3.0,
) -> torch.Tensor:
    return _smooth_alpha_s(r, NC=NC, Nf=Nf, LambdaQCD=LambdaQCD, C2=C2)


def _smooth_alpha_s(
    r: torch.Tensor,
    *,
    NC: float,
    Nf: int,
    LambdaQCD: float,
    C2: float,
) -> torch.Tensor:
    alphas_mu0 = 2.5
    alphas_freeze_c = 0.2
    b0 = (11 * NC - 2 * Nf) / (12 * math.pi)

    log_mu0_term = r.new_tensor(2 / alphas_freeze_c * math.log(alphas_mu0))
    scale = 4 * C2 / (r.square() * LambdaQCD**2)
    log_scale_term = torch.log(scale) / alphas_freeze_c
    log_arg = alphas_freeze_c * torch.logaddexp(log_mu0_term, log_scale_term)
    return 1 / (b0 * log_arg)


__all__ = ["bk_alpha_s", "dis_alpha_s_freeze", "dis_alpha_s_smooth"]
