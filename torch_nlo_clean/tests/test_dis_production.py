from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from nlo_torch.dipole import GBW
from nlo_torch.dis import DISConfig, Polarization, Quark, QuarkType
from nlo_torch.dis.config import NC, AlphaEM
from nlo_torch.dis.fixed_sobol import (
    FixedSobolResult,
    integrate_triple_fixed_sobol,
    make_sobol_points,
    map_importance_grid,
    paired_endpoint_inputs,
)
from nlo_torch.dis.observables import (
    _custom_dis_supported,
    _fixed_sobol_supported,
    integrand_dip_massive,
    photon_proton_cross_section_LO_d2b,
    sigma_qg_d2b,
)
from nlo_torch.numerics.integration import IntegralResult


def test_fast_dis_settings_are_the_defaults() -> None:
    config = DISConfig()

    assert config.cuda_fusion
    assert config.cuda_nested
    assert config.cuda_nested_points == 48


def test_custom_dis_support_requires_enabled_cuda_float32() -> None:
    cuda_float32 = SimpleNamespace(is_cuda=True, dtype=torch.float32)

    assert _custom_dis_supported(cuda_float32, DISConfig())
    assert not _custom_dis_supported(cuda_float32, DISConfig(cuda_fusion=False))
    assert not _custom_dis_supported(torch.tensor(1.0), DISConfig())


def test_dipole_integrand_dispatches_to_custom_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = torch.tensor([1.0, 2.0])

    monkeypatch.setattr(
        "nlo_torch.dis.observables._custom_dis_supported",
        lambda reference, config: True,
    )
    monkeypatch.setattr(
        "nlo_torch.custom_kernels.dis.lo_dipole.extension.dis_dipole_integrand_cuda",
        lambda *args, **kwargs: expected,
    )
    actual = integrand_dip_massive(
        torch.zeros((2, 3)),
        torch.tensor(9.0),
        torch.tensor(1e-3),
        Polarization.L,
        Quark(QuarkType.C, 1.4),
        "ab",
        GBW(),
        DISConfig(),
    )

    assert actual is expected


def test_lo_cross_section_dispatches_to_custom_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def integrand(*args, **kwargs):
        nonlocal calls
        calls += 1
        return torch.tensor([2.0, 3.0])

    def quadrature(function, reference, dimensions, points):
        values = function(torch.zeros((2, dimensions)))
        return values.sum(), torch.tensor(0.25), values.numel()

    monkeypatch.setattr(
        "nlo_torch.dis.observables._custom_dis_supported",
        lambda reference, config: True,
    )
    monkeypatch.setattr(
        "nlo_torch.custom_kernels.dis.lo_dipole.extension.dis_lo_integrand_cuda",
        integrand,
    )
    monkeypatch.setattr(
        "nlo_torch.custom_kernels.quadrature.unit_tensor_gauss_legendre_cuda",
        quadrature,
    )
    result = photon_proton_cross_section_LO_d2b(
        torch.tensor(9.0),
        torch.tensor(1e-3),
        Polarization.T,
        GBW(),
        DISConfig(epsrel=0.1),
    )

    factor = 4 * AlphaEM * NC / (2 * math.pi) ** 2
    torch.testing.assert_close(result.value, torch.tensor(5.0 * factor))
    torch.testing.assert_close(result.error, torch.tensor(0.25 * factor))
    assert calls == 1
    assert result.n_eval == 2
    assert result.converged


def test_fixed_sobol_support_is_limited_to_validated_workload() -> None:
    dipole = GBW()
    cuda_float32 = SimpleNamespace(is_cuda=True, dtype=torch.float32)

    assert _fixed_sobol_supported(cuda_float32, dipole, DISConfig())
    assert not _fixed_sobol_supported(cuda_float32, dipole, DISConfig(maxeval=100_000))
    assert not _fixed_sobol_supported(torch.tensor(1.0), dipole, DISConfig())


def test_fixed_sobol_sampling_is_reproducible_and_covers_the_gluon_disk() -> None:
    arguments = {
        "dimensions": 5,
        "training_samples": 256,
        "training_rounds": 2,
        "samples_per_replicate": 512,
        "replicates": 3,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
        "seed": 11,
    }
    first = make_sobol_points(**arguments)
    second = make_sobol_points(**arguments)
    for actual, expected in zip(
        first.training + first.replicates,
        second.training + second.replicates,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    points = first.replicates[0]
    edges = torch.linspace(0, 1, 33).expand(5, -1).clone()
    mapped, inverse_density, _ = map_importance_grid(points, edges)
    torch.testing.assert_close(mapped, points)
    torch.testing.assert_close(inverse_density, torch.ones_like(inverse_density))
    _, first_weight, _, second_weight = paired_endpoint_inputs(points, 5.0)
    torch.testing.assert_close(
        (first_weight + second_weight).mean(),
        torch.tensor(1.0),
        rtol=0,
        atol=1e-2,
    )


def test_fixed_sobol_total_uses_correlated_replicates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nlo_torch.dis.fixed_sobol.dis_sobol_endpoint_map_cuda",
        lambda points, edges, maxr, fold_angle: (
            points,
            torch.ones(points.shape[0]),
            points,
            torch.zeros(points.shape[0]),
        ),
    )

    def i1_sums(x: torch.Tensor, weights: torch.Tensor):
        return x[:, 0] * weights, torch.zeros(1, dtype=torch.float64)

    def i2_i3(x: torch.Tensor):
        return x[:, 1], -0.5 * x[:, 0]

    replicates = (
        torch.tensor([[0.1, 0.8], [0.3, 0.4]]),
        torch.tensor([[0.5, 0.2], [0.7, 0.6]]),
        torch.tensor([[0.2, 0.9], [0.4, 0.1]]),
    )
    result = integrate_triple_fixed_sobol(
        i1_sums,
        i2_i3,
        torch.empty(0),
        replicates,
        maxr=5.0,
        epsrel=1.0,
        seed=13,
        training_evaluations=10,
    )
    replicate_totals = torch.tensor([0.7, 0.7, 0.65], dtype=torch.float64)

    torch.testing.assert_close(result.total.value, replicate_totals.mean())
    torch.testing.assert_close(
        result.total.error,
        replicate_totals.std(unbiased=True) / math.sqrt(3),
    )
    assert result.total.n_eval == 22


def test_qg_cross_section_dispatches_to_fixed_sobol(monkeypatch: pytest.MonkeyPatch) -> None:
    quark = Quark(QuarkType.C, 1.4)
    config = DISConfig(quarks=(quark,))
    estimate = IntegralResult(
        value=torch.tensor(2.0, dtype=torch.float64),
        error=torch.tensor(0.1, dtype=torch.float64),
        n_eval=17,
        converged=True,
        seed=11,
    )

    monkeypatch.setattr(
        "nlo_torch.dis.observables._fixed_sobol_supported",
        lambda Q2, dipole, config: True,
    )
    monkeypatch.setattr(
        "nlo_torch.dis.fixed_sobol.gbw_qg_fixed_sobol",
        lambda *args, **kwargs: FixedSobolResult((estimate, estimate, estimate), estimate),
    )
    monkeypatch.setattr(
        "nlo_torch.dis.observables._vegas_with_budget",
        lambda *args, **kwargs: pytest.fail("fixed-Sobol dispatch fell through to Vegas"),
    )

    result = sigma_qg_d2b(
        torch.tensor(9.0, dtype=torch.float64),
        torch.tensor(1e-3, dtype=torch.float64),
        Polarization.T,
        GBW(),
        config,
        seed=11,
    )

    factor = quark.charge**2 * 4 * NC * AlphaEM / (2 * math.pi) ** 3 * 2 * math.pi
    torch.testing.assert_close(result.value, estimate.value * factor)
    torch.testing.assert_close(result.error, estimate.error * factor)
    assert result.n_eval == estimate.n_eval
    assert result.converged
