from __future__ import annotations

import pytest
import torch

from nlo_torch.bk.config import BKConfig
from nlo_torch.bk.evolution import _fixed_rk23_step, _fixed_rk23_supported
from nlo_torch.numerics.integration import VegasState, vegas


def test_fast_bk_settings_are_the_defaults() -> None:
    config = BKConfig()

    assert config.CUDA_FUSION
    assert config.K1_FIXED
    assert not config.K1_FIXED_REFINE
    assert config.VEGAS_REUSE_GRID
    assert config.VEGAS_REUSE_WARMUP_FRACTION == 0.25
    assert not _fixed_rk23_supported(torch.tensor([0.1, 0.2]), config)


def test_fixed_rk23_uses_three_derivatives(monkeypatch: pytest.MonkeyPatch) -> None:
    rapidities: list[float] = []
    observed_vegas_states = []

    def derivative(N, r, rapidity, *args, **kwargs):
        rapidities.append(float(rapidity))
        observed_vegas_states.append(kwargs.get("vegas_states"))
        return torch.full_like(N, 2.0)

    monkeypatch.setattr("nlo_torch.bk.evolution._evolve_derivative", derivative)
    r = torch.tensor([0.1, 0.2])
    N = torch.tensor([0.3, 0.4])
    vegas_states = {}
    result = _fixed_rk23_step(
        N,
        1.0,
        0.2,
        r,
        torch.tensor([0.0]),
        N.unsqueeze(0),
        BKConfig(),
        121,
        vegas_states=vegas_states,
    )

    assert rapidities == pytest.approx([1.0, 1.1, 1.2])
    assert all(state is vegas_states for state in observed_vegas_states)
    torch.testing.assert_close(result, N + 0.4)


def test_vegas_reuses_grid_with_quarter_warmup() -> None:
    bounds = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    state = VegasState()
    arguments = {
        "samples_per_iteration": 2_000,
        "warmup_samples": 400,
        "max_iterations": 2,
        "min_iterations": 2,
        "epsrel": 1.0,
        "seed": 73,
        "state": state,
        "reuse_warmup_fraction": 0.25,
    }

    first = vegas(lambda x: x[:, 0].square() + x[:, 1], bounds, **arguments)
    assert first.n_eval == 4_400
    assert state.edges is not None
    first_edges = state.edges.clone()

    second = vegas(lambda x: x[:, 0].square() + x[:, 1], bounds, **arguments)
    assert second.n_eval == 4_100
    assert state.edges is not None
    assert not torch.equal(state.edges, first_edges)
