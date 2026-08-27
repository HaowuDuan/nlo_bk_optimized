"""Handwritten Triton fusion for the four NLO BK K2/Kf evaluations."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from nlo_torch.bk.config import BKConfig, RunningCouplingNLO
from nlo_torch.bk_kernels import Kernel_nlo, Kernel_nlo_fermion
from nlo_torch.numerics.interpolation import LogLogSpline


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
    # Match PyTorch's CUDA elementwise division outside the singular kernel core.
    return tl.inline_asm_elementwise(
        "div.full.f32 $0, $1, $2;", "=f,f,f", [left, right], tl.float32, True, 1
    )


@triton.jit
def _logaddexp(left, right):
    maximum = tl.maximum(left, right)
    difference = tl.abs(_sub_rn(left, right))
    return _add_rn(maximum, libdevice.log1p(libdevice.exp(-difference)))


@triton.jit
def _smooth_alpha_s_value(
    r,
    b0: tl.constexpr,
    lambda_squared: tl.constexpr,
    scale_numerator: tl.constexpr,
    log_mu0_term: tl.constexpr,
):
    r_square = _mul_rn(r, r)
    scale_denominator = _mul_rn(r_square, lambda_squared)
    scale = _mul_rn(libdevice.rcp_rn(scale_denominator), scale_numerator)
    log_scale_term = _mul_rn(libdevice.log(scale), 5.0)
    log_argument = _mul_rn(_logaddexp(log_mu0_term, log_scale_term), 0.2)
    return libdevice.rcp_rn(_mul_rn(b0, log_argument))


@triton.jit
def _smooth_alpha_s_kernel(
    r_ptr,
    output_ptr,
    elements,
    b0: tl.constexpr,
    lambda_squared: tl.constexpr,
    scale_numerator: tl.constexpr,
    log_mu0_term: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    r = tl.load(r_ptr + offsets, mask=mask)
    result = _smooth_alpha_s_value(
        r,
        b0,
        lambda_squared,
        scale_numerator,
        log_mu0_term,
    )
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _nlo_geometry_values(r, z, theta_z, z2, theta_z2):
    r_square = _mul_rn(r, r)
    z_square = _mul_rn(z, z)
    z2_square = _mul_rn(z2, z2)
    X_square = _sub_rn(
        _add_rn(r_square, z_square),
        _mul_rn(_mul_rn(_mul_rn(2.0, r), z), libdevice.cos(theta_z)),
    )
    X2_square = _sub_rn(
        _add_rn(r_square, z2_square),
        _mul_rn(_mul_rn(_mul_rn(2.0, r), z2), libdevice.cos(theta_z2)),
    )
    z_m_z2_square = _sub_rn(
        _add_rn(z_square, z2_square),
        _mul_rn(
            _mul_rn(_mul_rn(2.0, z), z2),
            libdevice.cos(_sub_rn(theta_z, theta_z2)),
        ),
    )

    negative_square = (X_square < 0.0) | (X2_square < 0.0) | (z_m_z2_square < 0.0)
    X = libdevice.sqrt_rn(tl.maximum(X_square, 0.0))
    X2 = libdevice.sqrt_rn(tl.maximum(X2_square, 0.0))
    z_m_z2 = libdevice.sqrt_rn(tl.maximum(z_m_z2_square, 0.0))
    invalid = negative_square | (
        (X < 1e-20) | (z < 1e-20) | (X2 < 1e-20) | (z2 < 1e-20) | (z_m_z2 < 1e-20)
    )

    safe_X = tl.where(invalid, 1.0, X)
    safe_Y = tl.where(invalid, 1.0, z)
    safe_X2 = tl.where(invalid, 1.0, X2)
    safe_Y2 = tl.where(invalid, 1.0, z2)
    safe_z_m_z2 = tl.where(invalid, 1.0, z_m_z2)
    smallest_distance = tl.minimum(
        tl.minimum(tl.minimum(r, safe_X), safe_Y),
        tl.minimum(tl.minimum(safe_X2, safe_Y2), safe_z_m_z2),
    )
    numerator = _mul_rn(safe_X, safe_Y2)
    denominator = _mul_rn(safe_X2, safe_Y)

    return (
        safe_X,
        safe_Y,
        safe_X2,
        safe_Y2,
        safe_z_m_z2,
        invalid,
        smallest_distance,
        _div_rn(numerator, denominator),
        _div_rn(denominator, numerator),
    )


@triton.jit
def _nlo_geometry_kernel(
    r_ptr,
    z_ptr,
    theta_z_ptr,
    z2_ptr,
    theta_z2_ptr,
    X_ptr,
    Y_ptr,
    X2_ptr,
    Y2_ptr,
    z_m_z2_ptr,
    invalid_ptr,
    minimum_ptr,
    ratio_ptr,
    ratio_swap_ptr,
    elements,
    z_stride: tl.constexpr,
    theta_z_stride: tl.constexpr,
    z2_stride: tl.constexpr,
    theta_z2_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    (
        safe_X,
        safe_Y,
        safe_X2,
        safe_Y2,
        safe_z_m_z2,
        invalid,
        smallest_distance,
        ratio,
        ratio_swap,
    ) = _nlo_geometry_values(
        tl.load(r_ptr),
        tl.load(z_ptr + offsets * z_stride, mask=mask),
        tl.load(theta_z_ptr + offsets * theta_z_stride, mask=mask),
        tl.load(z2_ptr + offsets * z2_stride, mask=mask),
        tl.load(theta_z2_ptr + offsets * theta_z2_stride, mask=mask),
    )
    tl.store(X_ptr + offsets, safe_X, mask=mask)
    tl.store(Y_ptr + offsets, safe_Y, mask=mask)
    tl.store(X2_ptr + offsets, safe_X2, mask=mask)
    tl.store(Y2_ptr + offsets, safe_Y2, mask=mask)
    tl.store(z_m_z2_ptr + offsets, safe_z_m_z2, mask=mask)
    tl.store(invalid_ptr + offsets, invalid, mask=mask)
    tl.store(minimum_ptr + offsets, smallest_distance, mask=mask)
    tl.store(ratio_ptr + offsets, ratio, mask=mask)
    tl.store(ratio_swap_ptr + offsets, ratio_swap, mask=mask)


@triton.jit
def _loglog_spline_value(
    value,
    r_grid_ptr,
    log_grid_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    grid_points: tl.constexpr,
    search_steps: tl.constexpr,
):
    minimum_r = tl.load(r_grid_ptr)
    maximum_r = tl.load(r_grid_ptr + grid_points - 1)
    evaluation_r = tl.minimum(tl.maximum(value, minimum_r), maximum_r)
    log_r = libdevice.log(evaluation_r)

    lower = tl.zeros(value.shape, tl.int32)
    upper = lower + grid_points
    for _ in tl.static_range(search_steps):
        middle = (lower + upper) // 2
        middle_x = tl.load(log_grid_ptr + middle)
        move_right = log_r >= middle_x
        lower = tl.where(move_right, middle + 1, lower)
        upper = tl.where(move_right, upper, middle)

    interval = tl.minimum(tl.maximum(lower - 1, 0), grid_points - 2)
    dx = _sub_rn(log_r, tl.load(log_grid_ptr + interval))
    polynomial = _add_rn(
        tl.load(a_ptr + interval),
        _mul_rn(
            dx,
            _add_rn(
                tl.load(b_ptr + interval),
                _mul_rn(
                    dx,
                    _add_rn(tl.load(c_ptr + interval), _mul_rn(dx, tl.load(d_ptr + interval))),
                ),
            ),
        ),
    )
    result = libdevice.exp(polynomial)
    result = tl.where(libdevice.finitef(result) != 0, result, 0.0)
    result = tl.where(value < minimum_r, 1.0, result)
    return tl.where(value > maximum_r, 0.0, result)


@triton.jit
def _nlo_spline_values(
    X,
    Y,
    X2,
    Y2,
    z_m_z2,
    r_grid_ptr,
    log_grid_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    grid_points: tl.constexpr,
    search_steps: tl.constexpr,
):
    return (
        _loglog_spline_value(
            X,
            r_grid_ptr,
            log_grid_ptr,
            a_ptr,
            b_ptr,
            c_ptr,
            d_ptr,
            grid_points,
            search_steps,
        ),
        _loglog_spline_value(
            Y,
            r_grid_ptr,
            log_grid_ptr,
            a_ptr,
            b_ptr,
            c_ptr,
            d_ptr,
            grid_points,
            search_steps,
        ),
        _loglog_spline_value(
            X2,
            r_grid_ptr,
            log_grid_ptr,
            a_ptr,
            b_ptr,
            c_ptr,
            d_ptr,
            grid_points,
            search_steps,
        ),
        _loglog_spline_value(
            Y2,
            r_grid_ptr,
            log_grid_ptr,
            a_ptr,
            b_ptr,
            c_ptr,
            d_ptr,
            grid_points,
            search_steps,
        ),
        _loglog_spline_value(
            z_m_z2,
            r_grid_ptr,
            log_grid_ptr,
            a_ptr,
            b_ptr,
            c_ptr,
            d_ptr,
            grid_points,
            search_steps,
        ),
    )


@triton.jit
def _nlo_spline_kernel(
    X_ptr,
    Y_ptr,
    X2_ptr,
    Y2_ptr,
    z_m_z2_ptr,
    r_grid_ptr,
    log_grid_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    S_X_ptr,
    S_Y_ptr,
    S_X2_ptr,
    S_Y2_ptr,
    S_z_m_z2_ptr,
    elements,
    grid_points: tl.constexpr,
    search_steps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    S_X, S_Y, S_X2, S_Y2, S_z_m_z2 = _nlo_spline_values(
        tl.load(X_ptr + offsets, mask=mask),
        tl.load(Y_ptr + offsets, mask=mask),
        tl.load(X2_ptr + offsets, mask=mask),
        tl.load(Y2_ptr + offsets, mask=mask),
        tl.load(z_m_z2_ptr + offsets, mask=mask),
        r_grid_ptr,
        log_grid_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        d_ptr,
        grid_points,
        search_steps,
    )
    tl.store(S_X_ptr + offsets, S_X, mask=mask)
    tl.store(S_Y_ptr + offsets, S_Y, mask=mask)
    tl.store(S_X2_ptr + offsets, S_X2, mask=mask)
    tl.store(S_Y2_ptr + offsets, S_Y2, mask=mask)
    tl.store(S_z_m_z2_ptr + offsets, S_z_m_z2, mask=mask)


@triton.jit
def _nlo_geometry_spline_kernel(
    r_ptr,
    z_ptr,
    theta_z_ptr,
    z2_ptr,
    theta_z2_ptr,
    r_grid_ptr,
    log_grid_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    X_ptr,
    Y_ptr,
    X2_ptr,
    Y2_ptr,
    z_m_z2_ptr,
    invalid_ptr,
    minimum_ptr,
    ratio_ptr,
    ratio_swap_ptr,
    S_X_ptr,
    S_Y_ptr,
    S_X2_ptr,
    S_Y2_ptr,
    S_z_m_z2_ptr,
    elements,
    z_stride: tl.constexpr,
    theta_z_stride: tl.constexpr,
    z2_stride: tl.constexpr,
    theta_z2_stride: tl.constexpr,
    grid_points: tl.constexpr,
    search_steps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    (
        safe_X,
        safe_Y,
        safe_X2,
        safe_Y2,
        safe_z_m_z2,
        invalid,
        smallest_distance,
        ratio,
        ratio_swap,
    ) = _nlo_geometry_values(
        tl.load(r_ptr),
        tl.load(z_ptr + offsets * z_stride, mask=mask),
        tl.load(theta_z_ptr + offsets * theta_z_stride, mask=mask),
        tl.load(z2_ptr + offsets * z2_stride, mask=mask),
        tl.load(theta_z2_ptr + offsets * theta_z2_stride, mask=mask),
    )
    S_X, S_Y, S_X2, S_Y2, S_z_m_z2 = _nlo_spline_values(
        safe_X,
        safe_Y,
        safe_X2,
        safe_Y2,
        safe_z_m_z2,
        r_grid_ptr,
        log_grid_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        d_ptr,
        grid_points,
        search_steps,
    )

    tl.store(X_ptr + offsets, safe_X, mask=mask)
    tl.store(Y_ptr + offsets, safe_Y, mask=mask)
    tl.store(X2_ptr + offsets, safe_X2, mask=mask)
    tl.store(Y2_ptr + offsets, safe_Y2, mask=mask)
    tl.store(z_m_z2_ptr + offsets, safe_z_m_z2, mask=mask)
    tl.store(invalid_ptr + offsets, invalid, mask=mask)
    tl.store(minimum_ptr + offsets, smallest_distance, mask=mask)
    tl.store(ratio_ptr + offsets, ratio, mask=mask)
    tl.store(ratio_swap_ptr + offsets, ratio_swap, mask=mask)
    tl.store(S_X_ptr + offsets, S_X, mask=mask)
    tl.store(S_Y_ptr + offsets, S_Y, mask=mask)
    tl.store(S_X2_ptr + offsets, S_X2, mask=mask)
    tl.store(S_Y2_ptr + offsets, S_Y2, mask=mask)
    tl.store(S_z_m_z2_ptr + offsets, S_z_m_z2, mask=mask)


@triton.jit
def _nlo_k2_kf_values(
    r,
    X,
    Y,
    X2,
    Y2,
    z_m_z2,
    z_m_z2_fourth,
    r_fourth,
    log_ratio,
    log_ratio_swap,
    nf: tl.constexpr,
    nc: tl.constexpr,
):
    invalid = (X < 1e-20) | (X2 < 1e-20) | (Y < 1e-20) | (Y2 < 1e-20) | (z_m_z2 < 1e-20)
    safe_X = tl.where(invalid, 1.0, X)
    safe_Y = tl.where(invalid, 1.0, Y)
    safe_X2 = tl.where(invalid, 1.0, X2)
    safe_Y2 = tl.where(invalid, 1.0, Y2)
    safe_z_m_z2 = tl.where(invalid, 1.0, z_m_z2)

    XY2 = _mul_rn(safe_X, safe_Y2)
    X2Y = _mul_rn(safe_X2, safe_Y)
    XY2sq = _mul_rn(XY2, XY2)
    X2Ysq = _mul_rn(X2Y, X2Y)
    difference = _sub_rn(XY2sq, X2Ysq)
    difference_swap = _sub_rn(X2Ysq, XY2sq)
    r_z = _mul_rn(r, safe_z_m_z2)
    r_z_sq = _mul_rn(r_z, r_z)
    r_sq = _mul_rn(r, r)

    products_sum = _add_rn(XY2sq, X2Ysq)
    shared_numerator = _sub_rn(products_sum, _mul_rn(4.0, r_z_sq))
    XY2_z = _mul_rn(XY2, safe_z_m_z2)
    X2Y_z = _mul_rn(X2Y, safe_z_m_z2)

    k2_term_a = _div_rn(shared_numerator, _mul_rn(z_m_z2_fourth, difference))
    k2_term_b = _div_rn(r_fourth, _mul_rn(XY2sq, difference))
    k2_term_c = _div_rn(r_sq, _mul_rn(XY2_z, XY2_z))
    k2_sum = _add_rn(_add_rn(k2_term_a, k2_term_b), k2_term_c)
    k2_correction = _mul_rn(_mul_rn(k2_sum, 2.0), log_ratio)
    k2 = _add_rn(_div_rn(-2.0, z_m_z2_fourth), k2_correction)

    k2_swap_term_a = _div_rn(shared_numerator, _mul_rn(z_m_z2_fourth, difference_swap))
    k2_swap_term_b = _div_rn(r_fourth, _mul_rn(X2Ysq, difference_swap))
    k2_swap_term_c = _div_rn(r_sq, _mul_rn(X2Y_z, X2Y_z))
    k2_swap_sum = _add_rn(_add_rn(k2_swap_term_a, k2_swap_term_b), k2_swap_term_c)
    k2_swap_correction = _mul_rn(_mul_rn(k2_swap_sum, 2.0), log_ratio_swap)
    k2_swap = _add_rn(_div_rn(-2.0, z_m_z2_fourth), k2_swap_correction)

    fermion_numerator = _sub_rn(products_sum, r_z_sq)
    kf_ratio = _div_rn(fermion_numerator, _mul_rn(z_m_z2_fourth, difference))
    kf_correction = _mul_rn(_mul_rn(kf_ratio, 2.0), log_ratio)
    kf = _div_rn(_mul_rn(_sub_rn(_div_rn(2.0, z_m_z2_fourth), kf_correction), nf), nc)

    kf_swap_ratio = _div_rn(fermion_numerator, _mul_rn(z_m_z2_fourth, difference_swap))
    kf_swap_correction = _mul_rn(_mul_rn(kf_swap_ratio, 2.0), log_ratio_swap)
    kf_swap = _div_rn(_mul_rn(_sub_rn(_div_rn(2.0, z_m_z2_fourth), kf_swap_correction), nf), nc)

    valid_k2 = ~invalid & (libdevice.finitef(k2) != 0)
    valid_k2_swap = ~invalid & (libdevice.finitef(k2_swap) != 0)
    valid_kf = ~invalid & (libdevice.finitef(kf) != 0)
    valid_kf_swap = ~invalid & (libdevice.finitef(kf_swap) != 0)
    return (
        tl.where(valid_k2, k2, 0.0),
        tl.where(valid_k2_swap, k2_swap, 0.0),
        tl.where(valid_kf, kf, 0.0),
        tl.where(valid_kf_swap, kf_swap, 0.0),
    )


@triton.jit
def _nlo_k2_kf_kernel(
    r_ptr,
    X_ptr,
    Y_ptr,
    X2_ptr,
    Y2_ptr,
    z_m_z2_ptr,
    z_m_z2_fourth_ptr,
    r_fourth_ptr,
    log_ratio_ptr,
    log_ratio_swap_ptr,
    k2_ptr,
    k2_swap_ptr,
    kf_ptr,
    kf_swap_ptr,
    elements,
    nf: tl.constexpr,
    nc: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    values = _nlo_k2_kf_values(
        tl.load(r_ptr),
        tl.load(X_ptr + offsets, mask=mask),
        tl.load(Y_ptr + offsets, mask=mask),
        tl.load(X2_ptr + offsets, mask=mask),
        tl.load(Y2_ptr + offsets, mask=mask),
        tl.load(z_m_z2_ptr + offsets, mask=mask),
        tl.load(z_m_z2_fourth_ptr + offsets, mask=mask),
        tl.load(r_fourth_ptr),
        tl.load(log_ratio_ptr + offsets, mask=mask),
        tl.load(log_ratio_swap_ptr + offsets, mask=mask),
        nf,
        nc,
    )
    tl.store(k2_ptr + offsets, values[0], mask=mask)
    tl.store(k2_swap_ptr + offsets, values[1], mask=mask)
    tl.store(kf_ptr + offsets, values[2], mask=mask)
    tl.store(kf_swap_ptr + offsets, values[3], mask=mask)


@triton.jit
def _nlo_integrand_kernel(
    r_ptr,
    X_ptr,
    Y_ptr,
    X2_ptr,
    Y2_ptr,
    z_m_z2_ptr,
    z_m_z2_fourth_ptr,
    r_fourth_ptr,
    ratio_ptr,
    ratio_swap_ptr,
    S_X_ptr,
    S_Y_ptr,
    S_X2_ptr,
    S_Y2_ptr,
    S_z_m_z2_ptr,
    alpha_s_ptr,
    smallest_distance_ptr,
    invalid_ptr,
    result_ptr,
    elements,
    nf: tl.constexpr,
    nc: tl.constexpr,
    symmetrize: tl.constexpr,
    normalization_denominator: tl.constexpr,
    fuse_smooth_alpha: tl.constexpr,
    coupling_b0: tl.constexpr,
    coupling_lambda_squared: tl.constexpr,
    coupling_scale_numerator: tl.constexpr,
    coupling_log_mu0_term: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    r = tl.load(r_ptr)
    X = tl.load(X_ptr + offsets, mask=mask)
    Y = tl.load(Y_ptr + offsets, mask=mask)
    X2 = tl.load(X2_ptr + offsets, mask=mask)
    Y2 = tl.load(Y2_ptr + offsets, mask=mask)
    z_m_z2 = tl.load(z_m_z2_ptr + offsets, mask=mask)
    log_ratio = libdevice.log(tl.load(ratio_ptr + offsets, mask=mask))
    log_ratio_swap = libdevice.log(tl.load(ratio_swap_ptr + offsets, mask=mask))
    values = _nlo_k2_kf_values(
        r,
        X,
        Y,
        X2,
        Y2,
        z_m_z2,
        tl.load(z_m_z2_fourth_ptr + offsets, mask=mask),
        tl.load(r_fourth_ptr),
        log_ratio,
        log_ratio_swap,
        nf,
        nc,
    )
    k2, k2_swap, kf, kf_swap = values

    S_X = tl.load(S_X_ptr + offsets, mask=mask)
    S_Y = tl.load(S_Y_ptr + offsets, mask=mask)
    S_X2 = tl.load(S_X2_ptr + offsets, mask=mask)
    S_Y2 = tl.load(S_Y2_ptr + offsets, mask=mask)
    S_z_m_z2 = tl.load(S_z_m_z2_ptr + offsets, mask=mask)

    dipole = -_sub_rn(_mul_rn(_mul_rn(S_X, S_z_m_z2), S_Y2), _mul_rn(S_X, S_Y))
    dipole_swap = -_sub_rn(_mul_rn(_mul_rn(S_X2, S_z_m_z2), S_Y), _mul_rn(S_X2, S_Y2))
    cut = (tl.abs(k2) > 1e10) & (tl.abs(dipole) < 1e-10)
    cut_swap = (tl.abs(k2_swap) > 1e10) & (tl.abs(dipole_swap) < 1e-10)
    k2 = tl.where(cut, 0.0, k2)
    dipole = tl.where(cut, 0.0, dipole)
    k2_swap = tl.where(cut_swap, 0.0, k2_swap)
    dipole_swap = tl.where(cut_swap, 0.0, dipole_swap)

    if symmetrize:
        result = _div_full(_add_rn(_mul_rn(k2, dipole), _mul_rn(k2_swap, dipole_swap)), 2.0)
    else:
        result = _mul_rn(k2, dipole)

    if nf > 0:
        dipole_f = _mul_rn(S_Y, _sub_rn(S_X2, S_X))
        dipole_f_swap = _mul_rn(S_Y2, _sub_rn(S_X, S_X2))
        if symmetrize:
            fermion = _div_full(
                _add_rn(_mul_rn(kf, dipole_f), _mul_rn(kf_swap, dipole_f_swap)), 2.0
            )
        else:
            fermion = _mul_rn(kf, dipole_f)
        result = _sub_rn(result, fermion)

    if fuse_smooth_alpha:
        alpha_s = _smooth_alpha_s_value(
            tl.load(smallest_distance_ptr + offsets, mask=mask),
            coupling_b0,
            coupling_lambda_squared,
            coupling_scale_numerator,
            coupling_log_mu0_term,
        )
    else:
        alpha_s = tl.load(alpha_s_ptr + offsets, mask=mask)
    scaled_alpha_s = _mul_rn(alpha_s, nc)
    result = _mul_rn(result, _mul_rn(scaled_alpha_s, scaled_alpha_s))
    result = _div_full(result, normalization_denominator)
    invalid = tl.load(invalid_ptr + offsets, mask=mask)
    valid = ~invalid & (libdevice.finitef(result) != 0)
    tl.store(result_ptr + offsets, tl.where(valid, result, 0.0), mask=mask)


def Kernel_nlo_pair_fused(
    r: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    X2: torch.Tensor,
    Y2: torch.Tensor,
    z_m_z2: torch.Tensor,
    config: BKConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return K2, swapped K2, Kf, and swapped Kf from one CUDA launch."""

    if not X.is_cuda or X.dtype is not torch.float32:
        k2 = Kernel_nlo(r, X, Y, X2, Y2, z_m_z2)
        k2_swap = Kernel_nlo(r, X2, Y2, X, Y, z_m_z2)
        kf = Kernel_nlo_fermion(r, X, Y, X2, Y2, z_m_z2, config)
        kf_swap = Kernel_nlo_fermion(r, X2, Y2, X, Y, z_m_z2, config)
        return k2, k2_swap, kf, kf_swap

    tensors = (X, Y, X2, Y2, z_m_z2)
    if any(tensor.shape != X.shape for tensor in tensors):
        raise ValueError("NLO BK kernel inputs must have equal shapes")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("NLO BK kernel inputs must be contiguous")
    if r.numel() != 1 or r.device != X.device or r.dtype != X.dtype:
        raise ValueError("r must be a matching scalar tensor")

    outputs = tuple(torch.empty_like(X) for _ in range(4))
    z_m_z2_fourth = z_m_z2.pow(4)
    r_fourth = r.pow(4)
    log_ratio = torch.log(X * Y2 / (X2 * Y))
    log_ratio_swap = torch.log(X2 * Y / (X * Y2))
    elements = X.numel()
    grid = (triton.cdiv(elements, 256),)
    _nlo_k2_kf_kernel[grid](
        r,
        X,
        Y,
        X2,
        Y2,
        z_m_z2,
        z_m_z2_fourth,
        r_fourth,
        log_ratio,
        log_ratio_swap,
        *outputs,
        elements=elements,
        nf=float(config.NF),
        nc=config.NC,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return outputs


def smooth_alpha_s_fused(r: torch.Tensor, config: BKConfig) -> torch.Tensor:
    """Evaluate the smooth BK coupling in one CUDA launch."""

    if not r.is_cuda or r.dtype is not torch.float32 or r.ndim != 1 or not r.is_contiguous():
        raise ValueError("the fused smooth BK coupling requires contiguous CUDA float32 values")
    if config.NF > 3:
        raise ValueError("the fused smooth BK coupling supports NF <= 3")

    output = torch.empty_like(r)
    elements = r.numel()
    grid = (triton.cdiv(elements, 256),)
    _smooth_alpha_s_kernel[grid](
        r,
        output,
        elements=elements,
        b0=(11 * config.NC - 2 * config.NF) / (12 * math.pi),
        lambda_squared=config.LambdaQCD**2,
        scale_numerator=4 * config.C2,
        log_mu0_term=10 * math.log(2.5),
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return output


def nlo_geometry_fused(
    r: torch.Tensor,
    z: torch.Tensor,
    theta_z: torch.Tensor,
    z2: torch.Tensor,
    theta_z2: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Return safe NLO geometry and its directly derived shared values."""

    tensors = (z, theta_z, z2, theta_z2)
    if not z.is_cuda or z.dtype is not torch.float32:
        raise ValueError("the fused NLO BK geometry requires CUDA float32 tensors")
    if any(tensor.shape != z.shape for tensor in tensors):
        raise ValueError("NLO BK geometry inputs must have equal shapes")
    if any(tensor.device != z.device or tensor.dtype != z.dtype for tensor in tensors):
        raise ValueError("NLO BK geometry inputs must have matching devices and dtypes")
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("NLO BK geometry inputs must be one-dimensional")
    if r.numel() != 1 or r.device != z.device or r.dtype != z.dtype:
        raise ValueError("r must be a matching scalar tensor")

    distances = tuple(torch.empty_like(z, memory_format=torch.contiguous_format) for _ in range(5))
    invalid = torch.empty_like(z, dtype=torch.bool, memory_format=torch.contiguous_format)
    smallest_distance = torch.empty_like(z, memory_format=torch.contiguous_format)
    ratio_arguments = tuple(
        torch.empty_like(z, memory_format=torch.contiguous_format) for _ in range(2)
    )
    elements = z.numel()
    grid = (triton.cdiv(elements, 256),)
    _nlo_geometry_kernel[grid](
        r,
        z,
        theta_z,
        z2,
        theta_z2,
        *distances,
        invalid,
        smallest_distance,
        *ratio_arguments,
        elements=elements,
        z_stride=z.stride(0),
        theta_z_stride=theta_z.stride(0),
        z2_stride=z2.stride(0),
        theta_z2_stride=theta_z2.stride(0),
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return *distances, invalid, smallest_distance, *ratio_arguments


def nlo_spline_fused(
    interpolator: LogLogSpline,
    r_grid: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    X2: torch.Tensor,
    Y2: torch.Tensor,
    z_m_z2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the five NLO S-matrix distances in one CUDA launch."""

    tensors = (X, Y, X2, Y2, z_m_z2)
    if not X.is_cuda or X.dtype is not torch.float32:
        raise ValueError("the fused NLO BK spline requires CUDA float32 tensors")
    if any(tensor.shape != X.shape for tensor in tensors):
        raise ValueError("NLO BK spline inputs must have equal shapes")
    if any(tensor.device != X.device or tensor.dtype != X.dtype for tensor in tensors):
        raise ValueError("NLO BK spline inputs must have matching devices and dtypes")
    if any(tensor.ndim != 1 or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("NLO BK spline inputs must be contiguous and one-dimensional")

    spline = interpolator._spline
    coefficients = (spline.x, spline.a, spline.b, spline.c, spline.d)
    if spline.x.ndim != 1 or r_grid.ndim != 1 or r_grid.shape != spline.x.shape:
        raise ValueError("the fused NLO BK spline requires one shared one-dimensional grid")
    if any(tensor.device != X.device or tensor.dtype != X.dtype for tensor in coefficients):
        raise ValueError("spline coefficients must match the evaluation device and dtype")
    if r_grid.device != X.device or r_grid.dtype != X.dtype:
        raise ValueError("r_grid must match the evaluation device and dtype")

    outputs = tuple(torch.empty_like(X) for _ in range(5))
    elements = X.numel()
    grid_points = r_grid.numel()
    grid = (triton.cdiv(elements, 256),)
    _nlo_spline_kernel[grid](
        *tensors,
        r_grid,
        *coefficients,
        *outputs,
        elements=elements,
        grid_points=grid_points,
        search_steps=(grid_points - 1).bit_length(),
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return outputs


def nlo_geometry_spline_fused(
    interpolator: LogLogSpline,
    r_grid: torch.Tensor,
    r: torch.Tensor,
    z: torch.Tensor,
    theta_z: torch.Tensor,
    z2: torch.Tensor,
    theta_z2: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Evaluate NLO geometry and its five S-matrix splines in one CUDA launch."""

    tensors = (z, theta_z, z2, theta_z2)
    if not z.is_cuda or z.dtype is not torch.float32:
        raise ValueError("the fused NLO BK geometry-spline path requires CUDA float32 tensors")
    if any(tensor.shape != z.shape for tensor in tensors):
        raise ValueError("NLO BK geometry-spline inputs must have equal shapes")
    if any(tensor.device != z.device or tensor.dtype != z.dtype for tensor in tensors):
        raise ValueError("NLO BK geometry-spline inputs must have matching devices and dtypes")
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("NLO BK geometry-spline inputs must be one-dimensional")
    if r.numel() != 1 or r.device != z.device or r.dtype != z.dtype:
        raise ValueError("r must be a matching scalar tensor")

    spline = interpolator._spline
    coefficients = (spline.x, spline.a, spline.b, spline.c, spline.d)
    if spline.x.ndim != 1 or r_grid.ndim != 1 or r_grid.shape != spline.x.shape:
        raise ValueError("the fused NLO BK spline requires one shared one-dimensional grid")
    if any(tensor.device != z.device or tensor.dtype != z.dtype for tensor in coefficients):
        raise ValueError("spline coefficients must match the evaluation device and dtype")
    if r_grid.device != z.device or r_grid.dtype != z.dtype:
        raise ValueError("r_grid must match the evaluation device and dtype")

    distances = tuple(torch.empty_like(z, memory_format=torch.contiguous_format) for _ in range(5))
    invalid = torch.empty_like(z, dtype=torch.bool, memory_format=torch.contiguous_format)
    smallest_distance = torch.empty_like(z, memory_format=torch.contiguous_format)
    ratio_arguments = tuple(
        torch.empty_like(z, memory_format=torch.contiguous_format) for _ in range(2)
    )
    spline_outputs = tuple(
        torch.empty_like(z, memory_format=torch.contiguous_format) for _ in range(5)
    )
    elements = z.numel()
    grid_points = r_grid.numel()
    grid = (triton.cdiv(elements, 64),)
    _nlo_geometry_spline_kernel[grid](
        r,
        z,
        theta_z,
        z2,
        theta_z2,
        r_grid,
        *coefficients,
        *distances,
        invalid,
        smallest_distance,
        *ratio_arguments,
        *spline_outputs,
        elements=elements,
        z_stride=z.stride(0),
        theta_z_stride=theta_z.stride(0),
        z2_stride=z2.stride(0),
        theta_z2_stride=theta_z2.stride(0),
        grid_points=grid_points,
        search_steps=(grid_points - 1).bit_length(),
        BLOCK_SIZE=64,
        num_warps=2,
    )
    return *distances, invalid, smallest_distance, *ratio_arguments, *spline_outputs


def nlo_integrand_fused(
    r: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    X2: torch.Tensor,
    Y2: torch.Tensor,
    z_m_z2: torch.Tensor,
    S_X: torch.Tensor,
    S_Y: torch.Tensor,
    S_X2: torch.Tensor,
    S_Y2: torch.Tensor,
    S_z_m_z2: torch.Tensor,
    alpha_s: torch.Tensor | None,
    smallest_distance: torch.Tensor,
    invalid: torch.Tensor,
    ratio_argument: torch.Tensor,
    ratio_argument_swap: torch.Tensor,
    config: BKConfig,
) -> torch.Tensor:
    """Fuse K2/Kf with its dipole contractions and normalization."""

    tensors = (
        X,
        Y,
        X2,
        Y2,
        z_m_z2,
        S_X,
        S_Y,
        S_X2,
        S_Y2,
        S_z_m_z2,
        ratio_argument,
        ratio_argument_swap,
        smallest_distance,
    )
    if not X.is_cuda or X.dtype is not torch.float32:
        raise ValueError("the fused NLO BK integrand requires CUDA float32 tensors")
    if any(tensor.shape != X.shape for tensor in tensors) or invalid.shape != X.shape:
        raise ValueError("NLO BK integrand inputs must have equal shapes")
    if any(tensor.device != X.device or tensor.dtype != X.dtype for tensor in tensors):
        raise ValueError("NLO BK integrand inputs must have matching devices and dtypes")
    if any(not tensor.is_contiguous() for tensor in tensors) or not invalid.is_contiguous():
        raise ValueError("NLO BK integrand inputs must be contiguous")
    fuse_smooth_alpha = alpha_s is None
    if fuse_smooth_alpha:
        if config.RC_NLO is not RunningCouplingNLO.SMALLEST_NLO or config.NF > 3:
            raise ValueError(
                "inline smooth coupling requires smallest-distance running with NF <= 3"
            )
    elif (
        alpha_s.shape != X.shape
        or alpha_s.device != X.device
        or alpha_s.dtype != X.dtype
        or not alpha_s.is_contiguous()
    ):
        raise ValueError("alpha_s must match the NLO BK integrand inputs")
    if invalid.device != X.device or invalid.dtype is not torch.bool:
        raise ValueError("invalid must be a matching CUDA boolean tensor")
    if r.numel() != 1 or r.device != X.device or r.dtype != X.dtype:
        raise ValueError("r must be a matching scalar tensor")

    z_m_z2_fourth = z_m_z2.pow(4)
    r_fourth = r.pow(4)
    result = torch.empty_like(X)
    elements = X.numel()
    grid = (triton.cdiv(elements, 256),)
    _nlo_integrand_kernel[grid](
        r,
        X,
        Y,
        X2,
        Y2,
        z_m_z2,
        z_m_z2_fourth,
        r_fourth,
        ratio_argument,
        ratio_argument_swap,
        S_X,
        S_Y,
        S_X2,
        S_Y2,
        S_z_m_z2,
        alpha_s if alpha_s is not None else X,
        smallest_distance,
        invalid,
        result,
        elements=elements,
        nf=float(config.NF),
        nc=config.NC,
        symmetrize=config.SYMMETRIZE_Z_Z2_INTEGRATION,
        normalization_denominator=8 * torch.pi**4,
        fuse_smooth_alpha=fuse_smooth_alpha,
        coupling_b0=(11 * config.NC - 2 * config.NF) / (12 * math.pi),
        coupling_lambda_squared=config.LambdaQCD**2,
        coupling_scale_numerator=4 * config.C2,
        coupling_log_mu0_term=10 * math.log(2.5),
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return result


__all__ = [
    "Kernel_nlo_pair_fused",
    "nlo_geometry_fused",
    "nlo_geometry_spline_fused",
    "nlo_integrand_fused",
    "nlo_spline_fused",
    "smooth_alpha_s_fused",
]
