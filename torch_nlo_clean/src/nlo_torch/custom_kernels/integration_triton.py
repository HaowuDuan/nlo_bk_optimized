"""Handwritten Triton producers for CUDA Monte Carlo integration."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_rn(left, right):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", "=f,f,f", [left, right], tl.float32, True, 1
    )


@triton.jit
def _sub_rn(left, right):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;", "=f,f,f", [left, right], tl.float32, True, 1
    )


@triton.jit
def _mul_rn(left, right):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [left, right], tl.float32, True, 1
    )


@triton.jit
def _div_rn(left, right):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;", "=f,f,f", [left, right], tl.float32, True, 1
    )


@triton.jit
def _div_full(left, right):
    return tl.inline_asm_elementwise(
        "div.full.f32 $0, $1, $2;", "=f,f,f", [left, right], tl.float32, True, 1
    )


@triton.jit
def _fma_rn(left, middle, right):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        [left, middle, right],
        tl.float32,
        True,
        1,
    )


@triton.jit
def _vegas_samples_kernel(
    edges_ptr,
    lower_ptr,
    bounds_width_ptr,
    bin_index_ptr,
    random_ptr,
    x_ptr,
    width_ptr,
    elements,
    dimensions: tl.constexpr,
    edge_dimension_stride: tl.constexpr,
    edge_bin_stride: tl.constexpr,
    output_sample_stride: tl.constexpr,
    output_dimension_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    dimension = offsets % dimensions
    bin_index = tl.load(bin_index_ptr + offsets, mask=mask, other=0)
    edge_offset = dimension * edge_dimension_stride + bin_index * edge_bin_stride
    left = tl.load(edges_ptr + edge_offset, mask=mask)
    right = tl.load(edges_ptr + edge_offset + edge_bin_stride, mask=mask)
    width = _sub_rn(right, left)
    random = tl.load(random_ptr + offsets, mask=mask)
    u = _add_rn(left, _mul_rn(width, random))
    x = _add_rn(
        tl.load(lower_ptr + dimension, mask=mask),
        _mul_rn(tl.load(bounds_width_ptr + dimension, mask=mask), u),
    )
    sample = offsets // dimensions
    output_offset = sample * output_sample_stride + dimension * output_dimension_stride
    tl.store(x_ptr + output_offset, x, mask=mask)
    tl.store(width_ptr + output_offset, width, mask=mask)


@triton.jit
def _vegas_masks_kernel(
    bin_index_ptr,
    absolute_weight_ptr,
    weighted_mask_ptr,
    count_mask_ptr,
    elements,
    samples: tl.constexpr,
    dimensions: tl.constexpr,
    bins: tl.constexpr,
    sample_stride: tl.constexpr,
    dimension_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    bin_number = offsets % bins
    sample = (offsets // bins) % samples
    dimension = offsets // (samples * bins)
    labels = tl.load(
        bin_index_ptr + sample * sample_stride + dimension * dimension_stride,
        mask=mask,
        other=-1,
    )
    matches = labels == bin_number
    weights = tl.load(absolute_weight_ptr + sample, mask=mask, other=0.0)
    tl.store(weighted_mask_ptr + offsets, tl.where(matches, weights, 0.0), mask=mask)
    tl.store(count_mask_ptr + offsets, matches.to(tl.float32), mask=mask)


@triton.jit
def _vegas_smoothed_kernel(
    histogram_ptr,
    bin_count_ptr,
    output_ptr,
    elements,
    bins: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    position = offsets % bins
    row = offsets - position
    left_position = tl.maximum(position - 1, 0)
    right_position = tl.minimum(position + 1, bins - 1)

    left_count = tl.maximum(tl.load(bin_count_ptr + row + left_position, mask=mask), 1.0)
    center_count = tl.maximum(tl.load(bin_count_ptr + offsets, mask=mask), 1.0)
    right_count = tl.maximum(tl.load(bin_count_ptr + row + right_position, mask=mask), 1.0)
    left = _div_rn(tl.load(histogram_ptr + row + left_position, mask=mask), left_count)
    center = _div_rn(tl.load(histogram_ptr + offsets, mask=mask), center_count)
    right = _div_rn(tl.load(histogram_ptr + row + right_position, mask=mask), right_count)

    interior = _div_full(_add_rn(_add_rn(left, center), right), 3.0)
    first = _div_full(_add_rn(center, right), 2.0)
    last = _div_full(_add_rn(left, center), 2.0)
    smoothed = tl.where(position == 0, first, interior)
    smoothed = tl.where(position == bins - 1, last, smoothed)
    tl.store(output_ptr + offsets, _add_rn(smoothed, epsilon), mask=mask)


def vegas_smoothed_fused(histogram: torch.Tensor, bin_count: torch.Tensor) -> torch.Tensor:
    """Build the smoothed four-dimensional Vegas importance matrix."""

    if not histogram.is_cuda or histogram.dtype is not torch.float32:
        raise ValueError("fused Vegas smoothing requires CUDA float32 tensors")
    if histogram.ndim != 2 or not histogram.is_contiguous():
        raise ValueError("Vegas histogram must be a contiguous matrix")
    if (
        bin_count.shape != histogram.shape
        or bin_count.device != histogram.device
        or bin_count.dtype != histogram.dtype
        or not bin_count.is_contiguous()
    ):
        raise ValueError("Vegas bin counts must match the histogram")

    output = torch.empty_like(histogram)
    elements = output.numel()
    _vegas_smoothed_kernel[(triton.cdiv(elements, 128),)](
        histogram,
        bin_count,
        output,
        elements,
        bins=histogram.shape[1],
        epsilon=torch.finfo(histogram.dtype).eps,
        BLOCK_SIZE=128,
        num_warps=4,
    )
    return output


@triton.jit
def _linspace_value(total, position, intervals: tl.constexpr, values: tl.constexpr):
    step = _div_rn(total, tl.full(total.shape, intervals, tl.float32))
    from_start = _mul_rn(step, position.to(tl.float32))
    from_end = _fma_rn(-step, (intervals - position).to(tl.float32), total)
    return tl.where(position < values // 2, from_start, from_end)


@triton.jit
def _vegas_edges_kernel(
    edges_ptr,
    smoothed_ptr,
    cumulative_ptr,
    output_ptr,
    elements,
    bins: tl.constexpr,
    values_per_dimension: tl.constexpr,
    search_steps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    dimension = offsets // values_per_dimension
    position = offsets % values_per_dimension
    row_offset = dimension * values_per_dimension
    total = tl.load(cumulative_ptr + row_offset + bins, mask=mask)
    target = _linspace_value(total, position, bins, values_per_dimension)

    lower = tl.zeros_like(offsets)
    upper = tl.full(offsets.shape, values_per_dimension, tl.int32)
    for _ in range(search_steps):
        middle = (lower + upper) // 2
        middle_value = tl.load(cumulative_ptr + row_offset + middle, mask=mask)
        move_lower = middle_value <= target
        lower = tl.where(move_lower, middle + 1, lower)
        upper = tl.where(move_lower, upper, middle)
    interval = tl.maximum(0, tl.minimum(bins - 1, lower - 1))
    edge_row = dimension * values_per_dimension
    smooth_row = dimension * bins
    cumulative_left = tl.load(cumulative_ptr + row_offset + interval, mask=mask)
    smooth = tl.load(smoothed_ptr + smooth_row + interval, mask=mask)
    fraction = _div_rn(_sub_rn(target, cumulative_left), smooth)
    edge_left = tl.load(edges_ptr + edge_row + interval, mask=mask)
    edge_right = tl.load(edges_ptr + edge_row + interval + 1, mask=mask)
    updated = _add_rn(edge_left, _mul_rn(fraction, _sub_rn(edge_right, edge_left)))
    updated = tl.where(position == 0, 0.0, updated)
    updated = tl.where(position == bins, 1.0, updated)
    tl.store(output_ptr + offsets, updated, mask=mask)


@triton.jit
def _vegas_targets_kernel(
    total_ptr,
    output_ptr,
    elements,
    values_per_dimension: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    dimension = offsets // values_per_dimension
    position = offsets % values_per_dimension
    intervals = values_per_dimension - 1
    total = tl.load(total_ptr + dimension, mask=mask)
    value = _linspace_value(total, position, intervals, values_per_dimension)
    tl.store(output_ptr + offsets, value, mask=mask)


def vegas_targets_fused(total: torch.Tensor, values_per_dimension: int) -> torch.Tensor:
    """Build CUDA Vegas target rows without a host scalar round trip."""

    if not total.is_cuda or total.dtype is not torch.float32 or total.ndim != 1:
        raise ValueError("fused Vegas targets require a CUDA float32 vector")
    output = torch.empty(
        (total.numel(), values_per_dimension), device=total.device, dtype=total.dtype
    )
    elements = output.numel()
    _vegas_targets_kernel[(triton.cdiv(elements, 128),)](
        total,
        output,
        elements,
        values_per_dimension=values_per_dimension,
        BLOCK_SIZE=128,
        num_warps=4,
    )
    return output


def vegas_edges_fused(
    edges: torch.Tensor,
    smoothed: torch.Tensor,
    cumulative: torch.Tensor,
) -> torch.Tensor:
    """Finish all four Vegas edge updates in one CUDA launch."""

    tensors = (edges, smoothed, cumulative)
    if not edges.is_cuda or edges.dtype is not torch.float32:
        raise ValueError("fused Vegas edges require CUDA float32 tensors")
    if any(tensor.device != edges.device or tensor.dtype != edges.dtype for tensor in tensors):
        raise ValueError("Vegas edge tensors must have matching devices and dtypes")
    if any(tensor.ndim != 2 or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Vegas edge tensors must be contiguous matrices")
    dimensions, values_per_dimension = edges.shape
    bins = values_per_dimension - 1
    if smoothed.shape != (dimensions, bins):
        raise ValueError("smoothed importance must contain one value per bin")
    if cumulative.shape != edges.shape:
        raise ValueError("cumulative values must match the edge matrix")

    output = torch.empty_like(edges)
    elements = output.numel()
    _vegas_edges_kernel[(triton.cdiv(elements, 128),)](
        edges,
        smoothed,
        cumulative,
        output,
        elements,
        bins=bins,
        values_per_dimension=values_per_dimension,
        search_steps=values_per_dimension.bit_length(),
        BLOCK_SIZE=128,
        num_warps=4,
    )
    return output


def vegas_samples_fused(
    edges: torch.Tensor,
    lower: torch.Tensor,
    bounds_width: torch.Tensor,
    bin_index: torch.Tensor,
    random: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map Vegas random numbers to coordinates in one CUDA launch."""

    if not random.is_cuda or random.dtype is not torch.float32:
        raise ValueError("fused Vegas samples require CUDA float32 random numbers")
    if random.ndim != 2 or not random.is_contiguous():
        raise ValueError("Vegas random numbers must be a contiguous matrix")
    if bin_index.shape != random.shape or not bin_index.is_contiguous():
        raise ValueError("Vegas bin indices must match the random-number matrix")
    if bin_index.device != random.device or bin_index.dtype is not torch.int64:
        raise ValueError("Vegas bin indices must be matching CUDA int64 values")
    dimensions = random.shape[1]
    if edges.ndim != 2 or edges.shape[0] != dimensions or not edges.is_contiguous():
        raise ValueError("Vegas edges must be one contiguous row per dimension")
    vectors = (edges, lower, bounds_width)
    if any(tensor.device != random.device or tensor.dtype != random.dtype for tensor in vectors):
        raise ValueError("Vegas sampling tensors must have matching devices and dtypes")
    if lower.shape != (dimensions,) or bounds_width.shape != (dimensions,):
        raise ValueError("Vegas bounds must contain one value per dimension")
    if not lower.is_contiguous() or not bounds_width.is_contiguous():
        raise ValueError("Vegas bounds must be contiguous")

    # Native gather returns the transpose of a contiguous [dimension, sample]
    # matrix. Keep that layout because it also fixes the later product's
    # floating-point reduction order.
    samples = random.shape[0]
    x = torch.empty((dimensions, samples), device=random.device, dtype=random.dtype).T
    width = torch.empty_like(x)
    elements = random.numel()
    _vegas_samples_kernel[(triton.cdiv(elements, 256),)](
        edges,
        lower,
        bounds_width,
        bin_index,
        random,
        x,
        width,
        elements,
        dimensions=dimensions,
        edge_dimension_stride=edges.stride(0),
        edge_bin_stride=edges.stride(1),
        output_sample_stride=x.stride(0),
        output_dimension_stride=x.stride(1),
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return x, width


def vegas_masks_fused(
    bin_index: torch.Tensor,
    absolute_weight: torch.Tensor,
    bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build all deterministic Vegas reduction inputs in one CUDA launch."""

    if not absolute_weight.is_cuda or absolute_weight.dtype is not torch.float32:
        raise ValueError("fused Vegas masks require CUDA float32 weights")
    if absolute_weight.ndim != 1 or not absolute_weight.is_contiguous():
        raise ValueError("Vegas weights must be contiguous and one-dimensional")
    if bin_index.ndim != 2 or bin_index.shape[0] != absolute_weight.shape[0]:
        raise ValueError("Vegas bin indices must have one row per weight")
    if bin_index.device != absolute_weight.device or bin_index.dtype is not torch.int64:
        raise ValueError("Vegas bin indices must be matching CUDA int64 values")
    if bins < 2:
        raise ValueError("Vegas requires at least two bins")

    samples, dimensions = bin_index.shape
    output_shape = (dimensions, samples, bins)
    weighted_mask = torch.empty(output_shape, device=absolute_weight.device, dtype=torch.float32)
    count_mask = torch.empty_like(weighted_mask)
    elements = weighted_mask.numel()
    _vegas_masks_kernel[(triton.cdiv(elements, 256),)](
        bin_index,
        absolute_weight,
        weighted_mask,
        count_mask,
        elements,
        samples=samples,
        dimensions=dimensions,
        bins=bins,
        sample_stride=bin_index.stride(0),
        dimension_stride=bin_index.stride(1),
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return weighted_mask, count_mask


__all__ = [
    "vegas_edges_fused",
    "vegas_masks_fused",
    "vegas_samples_fused",
    "vegas_smoothed_fused",
    "vegas_targets_fused",
]
