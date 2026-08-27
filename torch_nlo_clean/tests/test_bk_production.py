from __future__ import annotations

import pytest
import torch

from nlo_torch.bk.config import BKConfig
from nlo_torch.bk.evolution import _fixed_rk23_step, _fixed_rk23_supported


def test_fast_bk_settings_are_the_defaults() -> None:
    config = BKConfig()

    assert config.CUDA_FUSION
    assert config.K1_FIXED
    assert not config.K1_FIXED_REFINE
    assert not _fixed_rk23_supported(torch.tensor([0.1, 0.2]), config)


def test_fixed_rk23_uses_three_derivatives(monkeypatch: pytest.MonkeyPatch) -> None:
    rapidities: list[float] = []

    def derivative(N, r, rapidity, *args, **kwargs):
        rapidities.append(float(rapidity))
        return torch.full_like(N, 2.0)

    monkeypatch.setattr("nlo_torch.bk.evolution._evolve_derivative", derivative)
    r = torch.tensor([0.1, 0.2])
    N = torch.tensor([0.3, 0.4])
    result = _fixed_rk23_step(
        N,
        1.0,
        0.2,
        r,
        torch.tensor([0.0]),
        N.unsqueeze(0),
        BKConfig(),
        121,
    )

    assert rapidities == pytest.approx([1.0, 1.1, 1.2])
    torch.testing.assert_close(result, N + 0.4)
