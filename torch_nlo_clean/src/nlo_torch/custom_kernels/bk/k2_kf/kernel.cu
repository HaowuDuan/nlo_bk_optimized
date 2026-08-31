#include <cmath>
#include <cfloat>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

__device__ __forceinline__ float add_rn(float left, float right) {
    return __fadd_rn(left, right);
}

__device__ __forceinline__ float sub_rn(float left, float right) {
    return __fsub_rn(left, right);
}

__device__ __forceinline__ float multiply_rn(float left, float right) {
    return __fmul_rn(left, right);
}

__device__ __forceinline__ float divide_rn(float numerator, float denominator) {
    float result;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(numerator), "f"(denominator));
    return result;
}

__device__ __forceinline__ float divide_full(float numerator, float denominator) {
    float result;
    asm("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(numerator), "f"(denominator));
    return result;
}

__device__ __forceinline__ float reciprocal_rn(float value) {
    float result;
    asm("rcp.rn.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ float logaddexp(float left, float right) {
    const float maximum = fmaxf(left, right);
    const float difference = fabsf(sub_rn(left, right));
    return add_rn(maximum, log1pf(expf(-difference)));
}

__device__ __forceinline__ float smooth_alpha_s(
    float r,
    float b0,
    float lambda_squared,
    float scale_numerator,
    float log_mu0_term
) {
    const float r_square = multiply_rn(r, r);
    const float scale_denominator = multiply_rn(r_square, lambda_squared);
    const float scale = multiply_rn(reciprocal_rn(scale_denominator), scale_numerator);
    const float log_scale_term = multiply_rn(logf(scale), 5.0F);
    const float log_argument = multiply_rn(logaddexp(log_mu0_term, log_scale_term), 0.2F);
    return reciprocal_rn(multiply_rn(b0, log_argument));
}

struct Geometry {
    float X;
    float Y;
    float X2;
    float Y2;
    float z_m_z2;
    float smallest_distance;
    float ratio;
    float ratio_swap;
    bool invalid;
};

__device__ __forceinline__ Geometry nlo_geometry(
    float r,
    float z,
    float theta_z,
    float z2,
    float theta_z2
) {
    const float r_square = multiply_rn(r, r);
    const float z_square = multiply_rn(z, z);
    const float z2_square = multiply_rn(z2, z2);
    const float X_square = sub_rn(
        add_rn(r_square, z_square),
        multiply_rn(multiply_rn(multiply_rn(2.0F, r), z), cosf(theta_z))
    );
    const float X2_square = sub_rn(
        add_rn(r_square, z2_square),
        multiply_rn(multiply_rn(multiply_rn(2.0F, r), z2), cosf(theta_z2))
    );
    const float z_m_z2_square = sub_rn(
        add_rn(z_square, z2_square),
        multiply_rn(
            multiply_rn(multiply_rn(2.0F, z), z2), cosf(sub_rn(theta_z, theta_z2))
        )
    );
    const bool negative_square = X_square < 0.0F || X2_square < 0.0F || z_m_z2_square < 0.0F;
    const float X = __fsqrt_rn(fmaxf(X_square, 0.0F));
    const float X2 = __fsqrt_rn(fmaxf(X2_square, 0.0F));
    const float z_m_z2 = __fsqrt_rn(fmaxf(z_m_z2_square, 0.0F));
    const bool invalid = negative_square || X < 1e-20F || z < 1e-20F || X2 < 1e-20F ||
        z2 < 1e-20F || z_m_z2 < 1e-20F;
    const float safe_X = invalid ? 1.0F : X;
    const float safe_Y = invalid ? 1.0F : z;
    const float safe_X2 = invalid ? 1.0F : X2;
    const float safe_Y2 = invalid ? 1.0F : z2;
    const float safe_z_m_z2 = invalid ? 1.0F : z_m_z2;
    const float smallest_distance = fminf(
        fminf(fminf(r, safe_X), safe_Y),
        fminf(fminf(safe_X2, safe_Y2), safe_z_m_z2)
    );
    const float numerator = multiply_rn(safe_X, safe_Y2);
    const float denominator = multiply_rn(safe_X2, safe_Y);
    return {
        safe_X,
        safe_Y,
        safe_X2,
        safe_Y2,
        safe_z_m_z2,
        smallest_distance,
        divide_rn(numerator, denominator),
        divide_rn(denominator, numerator),
        invalid,
    };
}

__device__ __forceinline__ bool is_sensitive_phase_space(
    float r,
    float log_z,
    float log_z2,
    float theta_z,
    float theta_z2,
    float sensitive_ratio
) {
    const float z = expf(log_z);
    const float z2 = expf(log_z2);
    const Geometry geometry = nlo_geometry(r, z, theta_z, z2, theta_z2);
    if (geometry.invalid) {
        return false;
    }
    const float XY2 = multiply_rn(geometry.X, geometry.Y2);
    const float X2Y = multiply_rn(geometry.X2, geometry.Y);
    const float XY2_square = multiply_rn(XY2, XY2);
    const float X2Y_square = multiply_rn(X2Y, X2Y);
    const float scale = add_rn(XY2_square, X2Y_square);
    const float relative_difference = divide_rn(
        fabsf(sub_rn(XY2_square, X2Y_square)),
        fmaxf(scale, FLT_MIN)
    );
    return relative_difference < sensitive_ratio;
}

__device__ __forceinline__ float n_loglog_spline(
    float value,
    const float* __restrict__ r_grid,
    const float* __restrict__ log_grid,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    const float* __restrict__ d,
    int grid_points
) {
    const float minimum_r = r_grid[0];
    const float maximum_r = r_grid[grid_points - 1];
    const float evaluation_r = fminf(fmaxf(value, minimum_r), maximum_r);
    const float log_r = logf(evaluation_r);
    int lower = 0;
    int upper = grid_points;
    while (lower < upper) {
        const int middle = (lower + upper) / 2;
        if (log_r >= log_grid[middle]) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    const int interval = max(0, min(grid_points - 2, lower - 1));
    const float dx = sub_rn(log_r, log_grid[interval]);
    const float polynomial = add_rn(
        a[interval],
        multiply_rn(
            dx,
            add_rn(b[interval], multiply_rn(dx, add_rn(c[interval], multiply_rn(dx, d[interval]))))
        )
    );
    float result = expf(polynomial);
    result = isfinite(result) ? result : 0.0F;
    result = value < minimum_r ? 0.0F : result;
    return value > maximum_r ? 1.0F : result;
}

struct NLOKernels {
    float k2;
    float k2_swap;
    float kf;
    float kf_swap;
};

__device__ __forceinline__ NLOKernels nlo_kernels(
    float r,
    const Geometry& geometry,
    float log_ratio,
    float log_ratio_swap,
    float nf,
    float nc
) {
    const float XY2 = multiply_rn(geometry.X, geometry.Y2);
    const float X2Y = multiply_rn(geometry.X2, geometry.Y);
    const float XY2_square = multiply_rn(XY2, XY2);
    const float X2Y_square = multiply_rn(X2Y, X2Y);
    const float difference = sub_rn(XY2_square, X2Y_square);
    const float difference_swap = sub_rn(X2Y_square, XY2_square);
    const float r_z = multiply_rn(r, geometry.z_m_z2);
    const float r_z_square = multiply_rn(r_z, r_z);
    const float r_square = multiply_rn(r, r);
    const float r_fourth = powf(r, 4.0F);
    const float z_m_z2_fourth = powf(geometry.z_m_z2, 4.0F);
    const float products_sum = add_rn(XY2_square, X2Y_square);
    const float shared_numerator = sub_rn(products_sum, multiply_rn(4.0F, r_z_square));
    const float XY2_z = multiply_rn(XY2, geometry.z_m_z2);
    const float X2Y_z = multiply_rn(X2Y, geometry.z_m_z2);

    const float k2_term_a = divide_rn(
        shared_numerator, multiply_rn(z_m_z2_fourth, difference)
    );
    const float k2_term_b = divide_rn(r_fourth, multiply_rn(XY2_square, difference));
    const float k2_term_c = divide_rn(r_square, multiply_rn(XY2_z, XY2_z));
    const float k2_sum = add_rn(add_rn(k2_term_a, k2_term_b), k2_term_c);
    const float k2 = add_rn(
        divide_rn(-2.0F, z_m_z2_fourth), multiply_rn(multiply_rn(k2_sum, 2.0F), log_ratio)
    );

    const float k2_swap_term_a = divide_rn(
        shared_numerator, multiply_rn(z_m_z2_fourth, difference_swap)
    );
    const float k2_swap_term_b = divide_rn(r_fourth, multiply_rn(X2Y_square, difference_swap));
    const float k2_swap_term_c = divide_rn(r_square, multiply_rn(X2Y_z, X2Y_z));
    const float k2_swap_sum = add_rn(
        add_rn(k2_swap_term_a, k2_swap_term_b), k2_swap_term_c
    );
    const float k2_swap = add_rn(
        divide_rn(-2.0F, z_m_z2_fourth),
        multiply_rn(multiply_rn(k2_swap_sum, 2.0F), log_ratio_swap)
    );

    const float fermion_numerator = sub_rn(products_sum, r_z_square);
    const float kf_ratio = divide_rn(
        fermion_numerator, multiply_rn(z_m_z2_fourth, difference)
    );
    const float kf = divide_rn(
        multiply_rn(
            sub_rn(
                divide_rn(2.0F, z_m_z2_fourth),
                multiply_rn(multiply_rn(kf_ratio, 2.0F), log_ratio)
            ),
            nf
        ),
        nc
    );
    const float kf_swap_ratio = divide_rn(
        fermion_numerator, multiply_rn(z_m_z2_fourth, difference_swap)
    );
    const float kf_swap = divide_rn(
        multiply_rn(
            sub_rn(
                divide_rn(2.0F, z_m_z2_fourth),
                multiply_rn(multiply_rn(kf_swap_ratio, 2.0F), log_ratio_swap)
            ),
            nf
        ),
        nc
    );
    return {
        !geometry.invalid && isfinite(k2) ? k2 : 0.0F,
        !geometry.invalid && isfinite(k2_swap) ? k2_swap : 0.0F,
        !geometry.invalid && isfinite(kf) ? kf : 0.0F,
        !geometry.invalid && isfinite(kf_swap) ? kf_swap : 0.0F,
    };
}

struct NLOIntegrand32 {
    float total;
    float k2;
    float kf;
};

__device__ __forceinline__ NLOIntegrand32 bk_nlo_regular_value32(
    float r,
    float log_z,
    float log_z2,
    float theta_z,
    float theta_z2,
    const float* __restrict__ r_grid,
    const float* __restrict__ log_grid,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    const float* __restrict__ d,
    int grid_points,
    float nf,
    float nc,
    bool symmetrize,
    float normalization_denominator,
    float coupling_b0,
    float coupling_lambda_squared,
    float coupling_scale_numerator,
    float coupling_log_mu0_term
) {
    const float z = expf(log_z);
    const float z2 = expf(log_z2);
    const Geometry geometry = nlo_geometry(r, z, theta_z, z2, theta_z2);
    const float N_X = n_loglog_spline(
        geometry.X, r_grid, log_grid, a, b, c, d, grid_points
    );
    const float N_Y = n_loglog_spline(
        geometry.Y, r_grid, log_grid, a, b, c, d, grid_points
    );
    const float N_X2 = n_loglog_spline(
        geometry.X2, r_grid, log_grid, a, b, c, d, grid_points
    );
    const float N_Y2 = n_loglog_spline(
        geometry.Y2, r_grid, log_grid, a, b, c, d, grid_points
    );
    const float N_z_m_z2 = n_loglog_spline(
        geometry.z_m_z2, r_grid, log_grid, a, b, c, d, grid_points
    );
    const NLOKernels kernels = nlo_kernels(
        r, geometry, logf(geometry.ratio), logf(geometry.ratio_swap), nf, nc
    );

    const float S_X = sub_rn(1.0F, N_X);
    const float S_X2 = sub_rn(1.0F, N_X2);
    float dipole_difference = sub_rn(add_rn(N_z_m_z2, N_Y2), N_Y);
    dipole_difference = sub_rn(dipole_difference, multiply_rn(N_z_m_z2, N_Y2));
    float dipole_swap_difference = sub_rn(add_rn(N_z_m_z2, N_Y), N_Y2);
    dipole_swap_difference = sub_rn(
        dipole_swap_difference, multiply_rn(N_z_m_z2, N_Y)
    );
    float dipole = multiply_rn(S_X, dipole_difference);
    float dipole_swap = multiply_rn(S_X2, dipole_swap_difference);
    float k2 = kernels.k2;
    float k2_swap = kernels.k2_swap;
    const bool cut = fabsf(k2) > 1e10F && fabsf(dipole) < 1e-10F;
    const bool cut_swap = fabsf(k2_swap) > 1e10F && fabsf(dipole_swap) < 1e-10F;
    k2 = cut ? 0.0F : k2;
    dipole = cut ? 0.0F : dipole;
    k2_swap = cut_swap ? 0.0F : k2_swap;
    dipole_swap = cut_swap ? 0.0F : dipole_swap;

    float k2_result = symmetrize
        ? divide_full(
              add_rn(multiply_rn(k2, dipole), multiply_rn(k2_swap, dipole_swap)), 2.0F
          )
        : multiply_rn(k2, dipole);
    float kf_result = 0.0F;
    if (nf > 0.0F) {
        const float dipole_f = multiply_rn(sub_rn(1.0F, N_Y), sub_rn(N_X, N_X2));
        const float dipole_f_swap = multiply_rn(
            sub_rn(1.0F, N_Y2), sub_rn(N_X2, N_X)
        );
        const float fermion = symmetrize
            ? divide_full(
                  add_rn(
                      multiply_rn(kernels.kf, dipole_f),
                      multiply_rn(kernels.kf_swap, dipole_f_swap)
                  ),
                  2.0F
              )
            : multiply_rn(kernels.kf, dipole_f);
        kf_result = -fermion;
    }
    const float alpha_s = smooth_alpha_s(
        geometry.smallest_distance,
        coupling_b0,
        coupling_lambda_squared,
        coupling_scale_numerator,
        coupling_log_mu0_term
    );
    const float scaled_alpha_s = multiply_rn(alpha_s, nc);
    float scale = divide_full(
        multiply_rn(scaled_alpha_s, scaled_alpha_s), normalization_denominator
    );
    const float jacobian_exponent = add_rn(
        multiply_rn(2.0F, log_z), multiply_rn(2.0F, log_z2)
    );
    scale = multiply_rn(scale, expf(jacobian_exponent));
    k2_result = multiply_rn(k2_result, scale);
    kf_result = multiply_rn(kf_result, scale);
    k2_result = !geometry.invalid && isfinite(k2_result) ? k2_result : 0.0F;
    kf_result = !geometry.invalid && isfinite(kf_result) ? kf_result : 0.0F;
    return {add_rn(k2_result, kf_result), k2_result, kf_result};
}

struct Geometry64 {
    double X;
    double Y;
    double X2;
    double Y2;
    double z_m_z2;
    double smallest_distance;
    double ratio;
    double ratio_swap;
    bool invalid;
};

__device__ __forceinline__ Geometry64 nlo_geometry64(
    double r,
    double z,
    double theta_z,
    double z2,
    double theta_z2
) {
    const double r_square = r * r;
    const double z_square = z * z;
    const double z2_square = z2 * z2;
    const double X_square = r_square + z_square - 2.0 * r * z * cos(theta_z);
    const double X2_square = r_square + z2_square - 2.0 * r * z2 * cos(theta_z2);
    const double z_m_z2_square =
        z_square + z2_square - 2.0 * z * z2 * cos(theta_z - theta_z2);
    const bool negative_square = X_square < 0.0 || X2_square < 0.0 || z_m_z2_square < 0.0;
    const double X = sqrt(fmax(X_square, 0.0));
    const double X2 = sqrt(fmax(X2_square, 0.0));
    const double z_m_z2 = sqrt(fmax(z_m_z2_square, 0.0));
    const bool invalid = negative_square || X < 1e-20 || z < 1e-20 || X2 < 1e-20 ||
        z2 < 1e-20 || z_m_z2 < 1e-20;
    const double safe_X = invalid ? 1.0 : X;
    const double safe_Y = invalid ? 1.0 : z;
    const double safe_X2 = invalid ? 1.0 : X2;
    const double safe_Y2 = invalid ? 1.0 : z2;
    const double safe_z_m_z2 = invalid ? 1.0 : z_m_z2;
    const double smallest_distance = fmin(
        fmin(fmin(r, safe_X), safe_Y),
        fmin(fmin(safe_X2, safe_Y2), safe_z_m_z2)
    );
    const double numerator = safe_X * safe_Y2;
    const double denominator = safe_X2 * safe_Y;
    return {
        safe_X,
        safe_Y,
        safe_X2,
        safe_Y2,
        safe_z_m_z2,
        smallest_distance,
        numerator / denominator,
        denominator / numerator,
        invalid,
    };
}

__device__ __forceinline__ double loglog_spline64(
    double value,
    const double* __restrict__ r_grid,
    const double* __restrict__ log_grid,
    const double* __restrict__ a,
    const double* __restrict__ b,
    const double* __restrict__ c,
    const double* __restrict__ d,
    int grid_points
) {
    const double minimum_r = r_grid[0];
    const double maximum_r = r_grid[grid_points - 1];
    const double evaluation_r = fmin(fmax(value, minimum_r), maximum_r);
    const double log_r = log(evaluation_r);
    int lower = 0;
    int upper = grid_points;
    while (lower < upper) {
        const int middle = (lower + upper) / 2;
        if (log_r >= log_grid[middle]) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    const int interval = max(0, min(grid_points - 2, lower - 1));
    const double dx = log_r - log_grid[interval];
    const double polynomial =
        a[interval] + dx * (b[interval] + dx * (c[interval] + dx * d[interval]));
    double result = exp(polynomial);
    result = isfinite(result) ? result : 0.0;
    result = value < minimum_r ? 1.0 : result;
    return value > maximum_r ? 0.0 : result;
}

struct NLOKernels64 {
    double k2;
    double k2_swap;
    double kf;
    double kf_swap;
};

__device__ __forceinline__ NLOKernels64 nlo_kernels64(
    double r,
    const Geometry64& geometry,
    double log_ratio,
    double log_ratio_swap,
    double nf,
    double nc
) {
    const double XY2 = geometry.X * geometry.Y2;
    const double X2Y = geometry.X2 * geometry.Y;
    const double XY2_square = XY2 * XY2;
    const double X2Y_square = X2Y * X2Y;
    const double difference = XY2_square - X2Y_square;
    const double difference_swap = X2Y_square - XY2_square;
    const double r_z = r * geometry.z_m_z2;
    const double r_z_square = r_z * r_z;
    const double r_square = r * r;
    const double r_fourth = pow(r, 4.0);
    const double z_m_z2_fourth = pow(geometry.z_m_z2, 4.0);
    const double products_sum = XY2_square + X2Y_square;
    const double shared_numerator = products_sum - 4.0 * r_z_square;
    const double XY2_z = XY2 * geometry.z_m_z2;
    const double X2Y_z = X2Y * geometry.z_m_z2;

    const double k2_sum =
        shared_numerator / (z_m_z2_fourth * difference) +
        r_fourth / (XY2_square * difference) +
        r_square / (XY2_z * XY2_z);
    const double k2 = -2.0 / z_m_z2_fourth + 2.0 * k2_sum * log_ratio;
    const double k2_swap_sum =
        shared_numerator / (z_m_z2_fourth * difference_swap) +
        r_fourth / (X2Y_square * difference_swap) +
        r_square / (X2Y_z * X2Y_z);
    const double k2_swap = -2.0 / z_m_z2_fourth + 2.0 * k2_swap_sum * log_ratio_swap;

    const double fermion_numerator = products_sum - r_z_square;
    const double kf_ratio = fermion_numerator / (z_m_z2_fourth * difference);
    const double kf = (2.0 / z_m_z2_fourth - 2.0 * kf_ratio * log_ratio) * nf / nc;
    const double kf_swap_ratio = fermion_numerator / (z_m_z2_fourth * difference_swap);
    const double kf_swap =
        (2.0 / z_m_z2_fourth - 2.0 * kf_swap_ratio * log_ratio_swap) * nf / nc;
    return {
        !geometry.invalid && isfinite(k2) ? k2 : 0.0,
        !geometry.invalid && isfinite(k2_swap) ? k2_swap : 0.0,
        !geometry.invalid && isfinite(kf) ? kf : 0.0,
        !geometry.invalid && isfinite(kf_swap) ? kf_swap : 0.0,
    };
}

__device__ __forceinline__ double smooth_alpha_s64(
    double r,
    double b0,
    double lambda_squared,
    double scale_numerator,
    double log_mu0_term
) {
    const double scale = scale_numerator / (r * r * lambda_squared);
    const double left = log_mu0_term;
    const double right = 5.0 * log(scale);
    const double maximum = fmax(left, right);
    const double log_argument = 0.2 * (maximum + log1p(exp(-fabs(left - right))));
    return 1.0 / (b0 * log_argument);
}

struct NLOIntegrand64 {
    double total;
    double k2;
    double kf;
};

__device__ __forceinline__ NLOIntegrand64 bk_nlo_log_integrand_value64(
    double r,
    double log_z,
    double log_z2,
    double theta_z,
    double theta_z2,
    const double* __restrict__ r_grid,
    const double* __restrict__ log_grid,
    const double* __restrict__ a,
    const double* __restrict__ b,
    const double* __restrict__ c,
    const double* __restrict__ d,
    int grid_points,
    double nf,
    double nc,
    bool symmetrize,
    double normalization_denominator,
    double coupling_b0,
    double coupling_lambda_squared,
    double coupling_scale_numerator,
    double coupling_log_mu0_term
) {
    const double z = exp(log_z);
    const double z2 = exp(log_z2);
    const Geometry64 geometry = nlo_geometry64(r, z, theta_z, z2, theta_z2);
    const double S_X = loglog_spline64(
        geometry.X, r_grid, log_grid, a, b, c, d, grid_points
    );
    const double S_Y = loglog_spline64(
        geometry.Y, r_grid, log_grid, a, b, c, d, grid_points
    );
    const double S_X2 = loglog_spline64(
        geometry.X2, r_grid, log_grid, a, b, c, d, grid_points
    );
    const double S_Y2 = loglog_spline64(
        geometry.Y2, r_grid, log_grid, a, b, c, d, grid_points
    );
    const double S_z_m_z2 = loglog_spline64(
        geometry.z_m_z2, r_grid, log_grid, a, b, c, d, grid_points
    );
    const NLOKernels64 kernels = nlo_kernels64(
        r, geometry, log(geometry.ratio), log(geometry.ratio_swap), nf, nc
    );

    double dipole = -(S_X * S_z_m_z2 * S_Y2 - S_X * S_Y);
    double dipole_swap = -(S_X2 * S_z_m_z2 * S_Y - S_X2 * S_Y2);
    double k2 = kernels.k2;
    double k2_swap = kernels.k2_swap;
    const bool cut = fabs(k2) > 1e10 && fabs(dipole) < 1e-10;
    const bool cut_swap = fabs(k2_swap) > 1e10 && fabs(dipole_swap) < 1e-10;
    k2 = cut ? 0.0 : k2;
    dipole = cut ? 0.0 : dipole;
    k2_swap = cut_swap ? 0.0 : k2_swap;
    dipole_swap = cut_swap ? 0.0 : dipole_swap;

    double k2_result = symmetrize
        ? (k2 * dipole + k2_swap * dipole_swap) / 2.0
        : k2 * dipole;
    double kf_result = 0.0;
    if (nf > 0.0) {
        const double dipole_f = S_Y * (S_X2 - S_X);
        const double dipole_f_swap = S_Y2 * (S_X - S_X2);
        const double fermion = symmetrize
            ? (kernels.kf * dipole_f + kernels.kf_swap * dipole_f_swap) / 2.0
            : kernels.kf * dipole_f;
        kf_result = -fermion;
    }
    const double alpha_s = smooth_alpha_s64(
        geometry.smallest_distance,
        coupling_b0,
        coupling_lambda_squared,
        coupling_scale_numerator,
        coupling_log_mu0_term
    );
    const double scale = (alpha_s * nc) * (alpha_s * nc) /
        normalization_denominator * exp(2.0 * log_z + 2.0 * log_z2);
    k2_result *= scale;
    kf_result *= scale;
    k2_result = !geometry.invalid && isfinite(k2_result) ? k2_result : 0.0;
    kf_result = !geometry.invalid && isfinite(kf_result) ? kf_result : 0.0;
    return {k2_result + kf_result, k2_result, kf_result};
}

__global__ void bk_nlo_vegas_region_summaries_kernel(
    const float* __restrict__ r_ptr,
    const float* __restrict__ edges,
    const float* __restrict__ bounds_lower,
    const float* __restrict__ bounds_width,
    const int64_t* __restrict__ bin_index,
    const float* __restrict__ random,
    const float* __restrict__ volume,
    const float* __restrict__ n_r_grid,
    const float* __restrict__ n_log_grid,
    const float* __restrict__ n_a,
    const float* __restrict__ n_b,
    const float* __restrict__ n_c,
    const float* __restrict__ n_d,
    const double* __restrict__ s_r_grid,
    const double* __restrict__ s_log_grid,
    const double* __restrict__ s_a,
    const double* __restrict__ s_b,
    const double* __restrict__ s_c,
    const double* __restrict__ s_d,
    float* __restrict__ absolute_weight,
    float* __restrict__ regular_block_sum,
    float* __restrict__ regular_block_square,
    double* __restrict__ sensitive_block_sum,
    double* __restrict__ sensitive_block_square,
    double* __restrict__ k2_block_sum,
    double* __restrict__ kf_block_sum,
    int samples,
    int grid_points,
    float sensitive_ratio,
    double nf,
    double nc,
    bool symmetrize,
    double normalization_denominator,
    double coupling_b0,
    double coupling_lambda_squared,
    double coupling_scale_numerator,
    double coupling_log_mu0_term
) {
    const int sample = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float regular_sum[128];
    __shared__ float regular_square[128];
    __shared__ double sensitive_sum[128];
    __shared__ double sensitive_square[128];
    __shared__ double k2_sum[128];
    __shared__ double kf_sum[128];
    float regular_value = 0.0F;
    double sensitive_value = 0.0;
    double k2_value = 0.0;
    double kf_value = 0.0;
    float x[4];
    float width[4];
    if (sample < samples) {
#pragma unroll
        for (int dimension = 0; dimension < 4; ++dimension) {
            const int64_t label = bin_index[static_cast<int64_t>(sample) * 4 + dimension];
            const float left = edges[dimension * 33 + label];
            const float right = edges[dimension * 33 + label + 1];
            width[dimension] = sub_rn(right, left);
            const float u = add_rn(
                left,
                multiply_rn(
                    width[dimension], random[static_cast<int64_t>(sample) * 4 + dimension]
                )
            );
            x[dimension] = add_rn(
                bounds_lower[dimension], multiply_rn(bounds_width[dimension], u)
            );
        }
        const bool sensitive = is_sensitive_phase_space(
            r_ptr[0], x[0], x[1], x[2], x[3], sensitive_ratio
        );
        if (sensitive) {
            const NLOIntegrand64 integrand = bk_nlo_log_integrand_value64(
                static_cast<double>(r_ptr[0]),
                static_cast<double>(x[0]),
                static_cast<double>(x[1]),
                static_cast<double>(x[2]),
                static_cast<double>(x[3]),
                s_r_grid,
                s_log_grid,
                s_a,
                s_b,
                s_c,
                s_d,
                grid_points,
                nf,
                nc,
                symmetrize,
                normalization_denominator,
                coupling_b0,
                coupling_lambda_squared,
                coupling_scale_numerator,
                coupling_log_mu0_term
            );
            const double inverse_density =
                (32.0 * static_cast<double>(width[0])) *
                (32.0 * static_cast<double>(width[1])) *
                (32.0 * static_cast<double>(width[2])) *
                (32.0 * static_cast<double>(width[3]));
            const double weight = static_cast<double>(volume[0]) * inverse_density;
            sensitive_value = integrand.total * weight;
            k2_value = integrand.k2 * weight;
            kf_value = integrand.kf * weight;
            absolute_weight[sample] = static_cast<float>(fabs(sensitive_value));
        } else {
            const NLOIntegrand32 integrand = bk_nlo_regular_value32(
                r_ptr[0],
                x[0],
                x[1],
                x[2],
                x[3],
                n_r_grid,
                n_log_grid,
                n_a,
                n_b,
                n_c,
                n_d,
                grid_points,
                static_cast<float>(nf),
                static_cast<float>(nc),
                symmetrize,
                static_cast<float>(normalization_denominator),
                static_cast<float>(coupling_b0),
                static_cast<float>(coupling_lambda_squared),
                static_cast<float>(coupling_scale_numerator),
                static_cast<float>(coupling_log_mu0_term)
            );
            const float density_0 = multiply_rn(32.0F, width[0]);
            const float density_1 = multiply_rn(32.0F, width[1]);
            const float density_2 = multiply_rn(32.0F, width[2]);
            const float density_3 = multiply_rn(32.0F, width[3]);
            const float inverse_density = multiply_rn(
                multiply_rn(multiply_rn(density_0, density_1), density_2), density_3
            );
            regular_value = multiply_rn(
                multiply_rn(integrand.total, volume[0]), inverse_density
            );
            const float weight = multiply_rn(volume[0], inverse_density);
            k2_value = static_cast<double>(multiply_rn(integrand.k2, weight));
            kf_value = static_cast<double>(multiply_rn(integrand.kf, weight));
            absolute_weight[sample] = fabsf(regular_value);
        }
    }
    regular_sum[threadIdx.x] = regular_value;
    regular_square[threadIdx.x] = multiply_rn(regular_value, regular_value);
    sensitive_sum[threadIdx.x] = sensitive_value;
    sensitive_square[threadIdx.x] = sensitive_value * sensitive_value;
    k2_sum[threadIdx.x] = k2_value;
    kf_sum[threadIdx.x] = kf_value;
    __syncthreads();

#pragma unroll
    for (int offset = 64; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            regular_sum[threadIdx.x] = add_rn(
                regular_sum[threadIdx.x], regular_sum[threadIdx.x + offset]
            );
            regular_square[threadIdx.x] = add_rn(
                regular_square[threadIdx.x], regular_square[threadIdx.x + offset]
            );
            sensitive_sum[threadIdx.x] += sensitive_sum[threadIdx.x + offset];
            sensitive_square[threadIdx.x] += sensitive_square[threadIdx.x + offset];
            k2_sum[threadIdx.x] += k2_sum[threadIdx.x + offset];
            kf_sum[threadIdx.x] += kf_sum[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        regular_block_sum[blockIdx.x] = regular_sum[0];
        regular_block_square[blockIdx.x] = regular_square[0];
        sensitive_block_sum[blockIdx.x] = sensitive_sum[0];
        sensitive_block_square[blockIdx.x] = sensitive_square[0];
        k2_block_sum[blockIdx.x] = k2_sum[0];
        kf_block_sum[blockIdx.x] = kf_sum[0];
    }
}

__global__ void bk_nlo_vegas_region_totals_kernel(
    const float* __restrict__ regular_block_sum,
    const float* __restrict__ regular_block_square,
    const double* __restrict__ sensitive_block_sum,
    const double* __restrict__ sensitive_block_square,
    const double* __restrict__ k2_block_sum,
    const double* __restrict__ kf_block_sum,
    float* __restrict__ regular_total,
    float* __restrict__ regular_total_square,
    double* __restrict__ sensitive_total,
    double* __restrict__ sensitive_total_square,
    double* __restrict__ k2_total,
    double* __restrict__ kf_total,
    int blocks
) {
    __shared__ float regular_sum[512];
    __shared__ float regular_square[512];
    __shared__ double sensitive_sum[512];
    __shared__ double sensitive_square[512];
    __shared__ double k2_sum[512];
    __shared__ double kf_sum[512];
    const int index = threadIdx.x;
    regular_sum[index] = index < blocks ? regular_block_sum[index] : 0.0F;
    regular_square[index] = index < blocks ? regular_block_square[index] : 0.0F;
    sensitive_sum[index] = index < blocks ? sensitive_block_sum[index] : 0.0;
    sensitive_square[index] = index < blocks ? sensitive_block_square[index] : 0.0;
    k2_sum[index] = index < blocks ? k2_block_sum[index] : 0.0;
    kf_sum[index] = index < blocks ? kf_block_sum[index] : 0.0;
    __syncthreads();

#pragma unroll
    for (int offset = 256; offset > 0; offset /= 2) {
        if (index < offset) {
            regular_sum[index] = add_rn(regular_sum[index], regular_sum[index + offset]);
            regular_square[index] = add_rn(
                regular_square[index], regular_square[index + offset]
            );
            sensitive_sum[index] += sensitive_sum[index + offset];
            sensitive_square[index] += sensitive_square[index + offset];
            k2_sum[index] += k2_sum[index + offset];
            kf_sum[index] += kf_sum[index + offset];
        }
        __syncthreads();
    }
    if (index == 0) {
        regular_total[0] = regular_sum[0];
        regular_total_square[0] = regular_square[0];
        sensitive_total[0] = sensitive_sum[0];
        sensitive_total_square[0] = sensitive_square[0];
        k2_total[0] = k2_sum[0];
        kf_total[0] = kf_sum[0];
    }
}

std::vector<torch::Tensor> bk_nlo_mixed_vegas_summaries(
    torch::Tensor r,
    torch::Tensor edges,
    torch::Tensor bounds_lower,
    torch::Tensor bounds_width,
    torch::Tensor bin_index,
    torch::Tensor random,
    torch::Tensor volume,
    torch::Tensor r_grid,
    torch::Tensor log_grid,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d,
    torch::Tensor r_grid64,
    torch::Tensor log_grid64,
    torch::Tensor a64,
    torch::Tensor b64,
    torch::Tensor c64,
    torch::Tensor d64,
    double sensitive_ratio,
    int64_t nf,
    double nc,
    bool symmetrize,
    double normalization_denominator,
    double coupling_b0,
    double coupling_lambda_squared,
    double coupling_scale_numerator,
    double coupling_log_mu0_term
) {
    TORCH_CHECK(
        random.is_cuda() && random.scalar_type() == torch::kFloat32 && random.dim() == 2 &&
            random.size(1) == 4 && random.is_contiguous(),
        "random must be a contiguous CUDA float32 [samples, 4] matrix"
    );
    TORCH_CHECK(
        bin_index.is_cuda() && bin_index.scalar_type() == torch::kInt64 &&
            bin_index.sizes() == random.sizes() && bin_index.is_contiguous(),
        "bin_index must be a matching contiguous CUDA int64 matrix"
    );
    TORCH_CHECK(
        edges.is_cuda() && edges.scalar_type() == torch::kFloat32 &&
            edges.sizes() == torch::IntArrayRef({4, 33}) && edges.is_contiguous(),
        "edges must be a contiguous CUDA float32 [4, 33] matrix"
    );
    for (const auto& vector : {bounds_lower, bounds_width}) {
        TORCH_CHECK(
            vector.is_cuda() && vector.scalar_type() == torch::kFloat32 && vector.numel() == 4 &&
                vector.is_contiguous(),
            "Vegas bounds must be contiguous four-value CUDA float32 vectors"
        );
    }
    TORCH_CHECK(
        volume.is_cuda() && volume.scalar_type() == torch::kFloat32 && volume.numel() == 1 &&
            volume.is_contiguous(),
        "volume must be one contiguous CUDA float32 value"
    );
    TORCH_CHECK(
        r.is_cuda() && r.scalar_type() == torch::kFloat32 && r.numel() == 1,
        "r must be one CUDA float32 value"
    );
    for (const auto& tensor : {r_grid, log_grid, a, b, c, d}) {
        TORCH_CHECK(
            tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32 && tensor.dim() == 1 &&
                tensor.is_contiguous(),
            "regular spline tensors must be contiguous CUDA float32 vectors"
        );
    }
    for (const auto& tensor : {r_grid64, log_grid64, a64, b64, c64, d64}) {
        TORCH_CHECK(
            tensor.is_cuda() && tensor.scalar_type() == torch::kFloat64 && tensor.dim() == 1 &&
                tensor.is_contiguous(),
            "sensitive spline tensors must be contiguous CUDA float64 vectors"
        );
    }
    const std::vector<torch::Tensor> all_tensors = {
        r,          edges,       bounds_lower, bounds_width, bin_index,
        volume,     r_grid,      log_grid,     a,            b,
        c,          d,           r_grid64,     log_grid64,   a64,
        b64,        c64,         d64,
    };
    for (const auto& tensor : all_tensors) {
        TORCH_CHECK(tensor.device() == random.device(), "all tensors must use one CUDA device");
    }
    TORCH_CHECK(r_grid.numel() == log_grid.numel(), "r and log grids must have equal sizes");
    TORCH_CHECK(
        r_grid64.numel() == r_grid.numel() && log_grid64.numel() == r_grid.numel(),
        "regular and sensitive spline grids must have equal sizes"
    );
    TORCH_CHECK(
        a.numel() == r_grid.numel() - 1 && b.numel() == a.numel() && c.numel() == a.numel() &&
            d.numel() == a.numel() && a64.numel() == a.numel() && b64.numel() == a.numel() &&
            c64.numel() == a.numel() && d64.numel() == a.numel(),
        "spline coefficients must contain one value per interval"
    );
    TORCH_CHECK(sensitive_ratio > 0.0 && sensitive_ratio < 1.0, "invalid sensitive ratio");
    TORCH_CHECK(nf == 0 || nf == 3, "mixed Vegas BK requires NF=0 or NF=3");
    TORCH_CHECK(
        random.size(0) > 0 && random.size(0) <= 65536,
        "mixed Vegas BK batches must contain 1-65536 samples"
    );

    constexpr int threads = 128;
    const int samples = static_cast<int>(random.size(0));
    const int blocks = (samples + threads - 1) / threads;
    auto absolute_weight = torch::empty({samples}, random.options());
    auto regular_block_sum = torch::empty({blocks}, random.options());
    auto regular_block_square = torch::empty({blocks}, random.options());
    auto sensitive_block_sum = torch::empty(
        {blocks}, random.options().dtype(torch::kFloat64)
    );
    auto sensitive_block_square = torch::empty_like(sensitive_block_sum);
    auto k2_block_sum = torch::empty_like(sensitive_block_sum);
    auto kf_block_sum = torch::empty_like(sensitive_block_sum);
    auto regular_sum = torch::empty({}, random.options());
    auto regular_square = torch::empty({}, random.options());
    auto sensitive_sum = torch::empty({}, random.options().dtype(torch::kFloat64));
    auto sensitive_square = torch::empty_like(sensitive_sum);
    auto k2_sum = torch::empty_like(sensitive_sum);
    auto kf_sum = torch::empty_like(sensitive_sum);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bk_nlo_vegas_region_summaries_kernel<<<blocks, threads, 0, stream>>>(
        r.data_ptr<float>(),
        edges.data_ptr<float>(),
        bounds_lower.data_ptr<float>(),
        bounds_width.data_ptr<float>(),
        bin_index.data_ptr<int64_t>(),
        random.data_ptr<float>(),
        volume.data_ptr<float>(),
        r_grid.data_ptr<float>(),
        log_grid.data_ptr<float>(),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        d.data_ptr<float>(),
        r_grid64.data_ptr<double>(),
        log_grid64.data_ptr<double>(),
        a64.data_ptr<double>(),
        b64.data_ptr<double>(),
        c64.data_ptr<double>(),
        d64.data_ptr<double>(),
        absolute_weight.data_ptr<float>(),
        regular_block_sum.data_ptr<float>(),
        regular_block_square.data_ptr<float>(),
        sensitive_block_sum.data_ptr<double>(),
        sensitive_block_square.data_ptr<double>(),
        k2_block_sum.data_ptr<double>(),
        kf_block_sum.data_ptr<double>(),
        samples,
        r_grid64.numel(),
        static_cast<float>(sensitive_ratio),
        static_cast<double>(nf),
        nc,
        symmetrize,
        normalization_denominator,
        coupling_b0,
        coupling_lambda_squared,
        coupling_scale_numerator,
        coupling_log_mu0_term
    );
    bk_nlo_vegas_region_totals_kernel<<<1, 512, 0, stream>>>(
        regular_block_sum.data_ptr<float>(),
        regular_block_square.data_ptr<float>(),
        sensitive_block_sum.data_ptr<double>(),
        sensitive_block_square.data_ptr<double>(),
        k2_block_sum.data_ptr<double>(),
        kf_block_sum.data_ptr<double>(),
        regular_sum.data_ptr<float>(),
        regular_square.data_ptr<float>(),
        sensitive_sum.data_ptr<double>(),
        sensitive_square.data_ptr<double>(),
        k2_sum.data_ptr<double>(),
        kf_sum.data_ptr<double>(),
        blocks
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        absolute_weight,
        regular_sum,
        regular_square,
        sensitive_sum,
        sensitive_square,
        k2_sum,
        kf_sum,
    };
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "bk_nlo_mixed_vegas_summaries",
        &bk_nlo_mixed_vegas_summaries,
        "Disjoint float32 regular and float64 sensitive NLO BK Vegas summaries"
    );
}
