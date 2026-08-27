"""Local build and Python boundary for custom CUDA integration kernels."""

from __future__ import annotations

from pathlib import Path

import torch

from nlo_torch.custom_kernels.extension import load_cuda_extension


def _load_integration_extension():
    return load_cuda_extension(
        "nlo_torch_integration_cuda",
        Path(__file__).with_name("integration.cu"),
    )


def vegas_histogram_cuda(
    bin_index: torch.Tensor,
    absolute_weight: torch.Tensor,
    histogram: torch.Tensor,
    bin_count: torch.Tensor,
    *,
    semaphores: torch.Tensor | None = None,
    initialize: bool = False,
) -> None:
    """Accumulate one four-dimensional, 32-bin Vegas batch in place."""

    if semaphores is None:
        semaphores = torch.zeros(4, dtype=torch.int32, device=bin_index.device)
    _load_integration_extension().vegas_histogram(
        bin_index,
        absolute_weight,
        histogram,
        bin_count,
        semaphores,
        initialize,
    )


def vegas_random_cuda(
    samples: int,
    reference: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate four Vegas bin labels and uniforms per sample in one launch."""

    bin_index, random = _load_integration_extension().vegas_random(samples, reference, generator)
    return bin_index, random


def vegas_weighted_histogram_cuda(
    bin_index: torch.Tensor,
    weighted_value: torch.Tensor,
    total: torch.Tensor,
    total_square: torch.Tensor,
    histogram: torch.Tensor,
    bin_count: torch.Tensor,
    semaphores: torch.Tensor,
    *,
    initialize: bool = False,
) -> None:
    """Accumulate weighted moments and the four Vegas histograms together."""

    _load_integration_extension().vegas_weighted_histogram(
        bin_index,
        weighted_value,
        total,
        total_square,
        histogram,
        bin_count,
        semaphores,
        initialize,
    )


def vegas_initialize_cuda(
    bounds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize one four-dimensional, 32-bin Vegas state."""

    edges, bounds_lower, bounds_width, volume, semaphores = (
        _load_integration_extension().vegas_initialize(bounds)
    )
    return edges, bounds_lower, bounds_width, volume, semaphores


def vegas_accumulate_moments_cuda(
    value: torch.Tensor,
    width: torch.Tensor,
    volume: torch.Tensor,
    total: torch.Tensor,
    total_square: torch.Tensor,
    *,
    initialize: bool = False,
) -> torch.Tensor:
    """Accumulate four-dimensional Vegas moments and return histogram weights."""

    return _load_integration_extension().vegas_accumulate_moments(
        value, width, volume, total, total_square, initialize
    )


def vegas_accumulate_weighted_moments_cuda(
    weighted_value: torch.Tensor,
    total: torch.Tensor,
    total_square: torch.Tensor,
    *,
    initialize: bool = False,
) -> torch.Tensor:
    """Accumulate moments for final per-sample Vegas weights."""

    return _load_integration_extension().vegas_accumulate_weighted_moments(
        weighted_value, total, total_square, initialize
    )


def vegas_adapt_edges_cuda(
    edges: torch.Tensor,
    histogram: torch.Tensor,
    bin_count: torch.Tensor,
) -> torch.Tensor:
    """Adapt all four 32-bin Vegas grids in one CUDA launch."""

    return _load_integration_extension().vegas_adapt_edges(edges, histogram, bin_count)


def vegas_estimate_variance_cuda(
    total: torch.Tensor,
    total_square: torch.Tensor,
    samples: int,
) -> torch.Tensor:
    """Calculate one Vegas estimate and variance in one CUDA launch."""

    return _load_integration_extension().vegas_estimate_variance(total, total_square, samples)


def vegas_store_estimate_variance_cuda(
    total: torch.Tensor,
    total_square: torch.Tensor,
    samples: int,
    estimates: torch.Tensor,
    variances: torch.Tensor,
    output_index: int,
) -> None:
    """Store one Vegas estimate and variance in reusable device buffers."""

    _load_integration_extension().vegas_store_estimate_variance(
        total,
        total_square,
        samples,
        estimates,
        variances,
        output_index,
    )


def vegas_combine_estimates_cuda(
    estimates: torch.Tensor,
    variances: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    """Combine one to four Vegas estimates using inverse-variance weights."""

    return _load_integration_extension().vegas_combine_estimates(estimates, variances, iterations)


def vegas_store_and_combine_cuda(
    total: torch.Tensor,
    total_square: torch.Tensor,
    samples: int,
    estimates: torch.Tensor,
    variances: torch.Tensor,
    output_index: int,
) -> torch.Tensor:
    """Store one estimate and return the combined Vegas result."""

    return _load_integration_extension().vegas_store_and_combine(
        total,
        total_square,
        samples,
        estimates,
        variances,
        output_index,
    )


__all__ = [
    "vegas_accumulate_moments_cuda",
    "vegas_accumulate_weighted_moments_cuda",
    "vegas_adapt_edges_cuda",
    "vegas_combine_estimates_cuda",
    "vegas_estimate_variance_cuda",
    "vegas_histogram_cuda",
    "vegas_initialize_cuda",
    "vegas_random_cuda",
    "vegas_store_and_combine_cuda",
    "vegas_store_estimate_variance_cuda",
    "vegas_weighted_histogram_cuda",
]
