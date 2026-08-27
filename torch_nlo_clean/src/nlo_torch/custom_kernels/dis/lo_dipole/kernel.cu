#include <cmath>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Math.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int threads = 256;
constexpr float pi = 3.14159265358979323846F;
constexpr float pi_squared_over_six = 1.64493406684822643647F;
constexpr float lambda_qcd = 0.241F;
constexpr float color_factor = 4.0F / 3.0F;
constexpr float z_min = 1.0e-6F;

__device__ __forceinline__ float square(float value) {
    return value * value;
}

__device__ __forceinline__ float K0(float value) {
    return modified_bessel_k0_forward<float>(value);
}

__device__ __forceinline__ float K1(float value) {
    return modified_bessel_k1_forward<float>(value);
}

__device__ __forceinline__ float li2_series(float x) {
    float term = x;
    float result = x;
#pragma unroll 8
    for (int k = 2; k <= 64; ++k) {
        term *= x;
        result += term / static_cast<float>(k * k);
    }
    return result;
}

// Real dilogarithm on the x <= 1 domain used by the dipole impact factors.
__device__ __forceinline__ float li2(float x) {
    if (x == 0.0F) {
        return x;
    }
    if (x == 1.0F) {
        return pi_squared_over_six;
    }
    if (x < -1.0F) {
        const float inverse = 1.0F / x;
        const float core = inverse / (inverse - 1.0F);
        return li2_series(core) + 0.5F * square(log1pf(-inverse)) -
            pi_squared_over_six - 0.5F * square(logf(-x));
    }
    if (x < 0.0F) {
        const float core = x / (x - 1.0F);
        return -li2_series(core) - 0.5F * square(log1pf(-x));
    }
    if (x > 0.5F) {
        return pi_squared_over_six - logf(x) * log1pf(-x) - li2_series(1.0F - x);
    }
    return li2_series(x);
}

__device__ __forceinline__ float L_dip(float Q2, float z, float mf) {
    const float gamma = sqrtf(1.0F + 4.0F * square(mf) / Q2);
    const float two_z = 2.0F * z;
    const float two_one_minus_z = 2.0F * (1.0F - z);
    const float value =
        li2(1.0F / (1.0F - (1.0F - gamma) / two_z)) +
        li2(1.0F / (1.0F - (1.0F + gamma) / two_z)) +
        li2(1.0F / (1.0F - (1.0F - gamma) / two_one_minus_z)) +
        li2(1.0F / (1.0F - (1.0F + gamma) / two_one_minus_z));
    return value - (pi_squared_over_six - 0.5F * square(logf(z / (1.0F - z))));
}

__device__ __forceinline__ float spline_amplitude(
    float r,
    const float* nodes,
    const float* a,
    const float* b,
    const float* c,
    const float* d,
    int64_t node_count,
    float maxr_interpolate
) {
    const float minimum = nodes[0];
    const float maximum = nodes[node_count - 1];
    r = fminf(fmaxf(r, minimum), maximum);
    if (maxr_interpolate > 0.0F && r >= maxr_interpolate) {
        return 1.0F;
    }
    const float inverse_log_step = 1.0F / logf(nodes[1] / minimum);
    int64_t interval = static_cast<int64_t>(floorf(logf(r / minimum) * inverse_log_step));
    interval = max(static_cast<int64_t>(0), min(interval, node_count - 2));
    if (interval + 1 < node_count - 1 && nodes[interval + 1] <= r) {
        ++interval;
    }
    if (interval > 0 && nodes[interval] > r) {
        --interval;
    }
    const float dx = r - nodes[interval];
    const float value = a[interval] + dx * (b[interval] + dx * (c[interval] + dx * d[interval]));
    return fminf(fmaxf(value, 0.0F), 1.0F);
}

__device__ __forceinline__ float gbw_amplitude(
    float r,
    float rapidity,
    float Qs0_square,
    float lambda,
    float gamma,
    float x0
) {
    const float initial_rapidity = logf(1.0F / x0);
    const float effective_rapidity = fmaxf(rapidity, initial_rapidity);
    const float Qs_square = Qs0_square * expf(lambda * effective_rapidity);
    const float argument = 0.25F * powf(square(r) * Qs_square, gamma);
    return fabsf(argument) < 1.0e-7F ? argument : 1.0F - expf(-argument);
}

__device__ __forceinline__ float amplitude(
    float r,
    float rapidity,
    int amplitude_mode,
    const float* nodes,
    const float* a,
    const float* b,
    const float* c,
    const float* d,
    int64_t node_count,
    float maxr_interpolate,
    float Qs0_square,
    float lambda,
    float gamma,
    float x0
) {
    if (amplitude_mode == 0) {
        return gbw_amplitude(r, rapidity, Qs0_square, lambda, gamma, x0);
    }
    return spline_amplitude(r, nodes, a, b, c, d, node_count, maxr_interpolate);
}

__device__ __forceinline__ float running_coupling(
    float r_square,
    float C2,
    int active_flavors,
    float maximum_alpha,
    bool smooth
) {
    const float b0 = (33.0F - 2.0F * active_flavors) / (12.0F * pi);
    const float scale = 4.0F * C2 / (r_square * square(lambda_qcd));
    if (!smooth) {
        const float alpha = 1.0F / (b0 * logf(scale));
        return (alpha > maximum_alpha || scale < 1.0F) ? maximum_alpha : alpha;
    }
    constexpr float freeze_c = 0.2F;
    const float first = 2.0F / freeze_c * logf(2.5F);
    const float second = logf(scale) / freeze_c;
    const float larger = fmaxf(first, second);
    const float smaller = fminf(first, second);
    return 1.0F / (b0 * freeze_c * (larger + log1pf(expf(smaller - larger))));
}

__device__ __forceinline__ float omega_L_V(float Q2, float z, float mf) {
    const float gamma = sqrtf(1.0F + 4.0F * square(mf) / Q2);
    const float one_minus_z = 1.0F - z;
    return
        0.5F / z * (logf(one_minus_z) + gamma * logf((1.0F + gamma) / (1.0F + gamma - 2.0F * z))) +
        0.5F / one_minus_z * (logf(z) + gamma * logf((1.0F + gamma) / (1.0F + gamma - 2.0F * one_minus_z))) +
        0.25F / (z * one_minus_z) * (gamma - 1.0F + 2.0F * square(mf) / Q2) *
            logf((z * one_minus_z * Q2 + square(mf)) / square(mf));
}

__device__ __forceinline__ float omega_T_V_unsymmetric(float Q, float z, float mf) {
    const float Q2 = square(Q);
    const float gamma = sqrtf(1.0F + 4.0F * square(mf) / Q2);
    return (1.0F + 0.5F / z) *
        (logf(1.0F - z) + gamma * logf((1.0F + gamma) / (1.0F + gamma - 2.0F * z))) -
        0.5F / z * ((z + 0.5F) * (1.0F - gamma) + square(mf) / Q2) *
        logf((z * (1.0F - z) * Q2 + square(mf)) / square(mf));
}

__device__ __forceinline__ float omega_T_N_unsymmetric(float Q, float z, float mf) {
    const float Q2 = square(Q);
    const float gamma = sqrtf(1.0F + 4.0F * square(mf) / Q2);
    return (1.0F + z - 2.0F * square(z)) / z *
        (logf(1.0F - z) + gamma * logf((1.0F + gamma) / (1.0F + gamma - 2.0F * z))) +
        (1.0F - z) / z * ((0.5F + z) * (gamma - 1.0F) - square(mf) / Q2) *
        logf((z * (1.0F - z) * Q2 + square(mf)) / square(mf));
}

__device__ __forceinline__ float longitudinal_omega(float Q2, float z, float r, float mf) {
    const float front = 4.0F * Q2 * square(z * (1.0F - z));
    const float argument = sqrtf(Q2 * z * (1.0F - z) + square(mf)) * r;
    if (argument < 1.0e-7F) {
        return 0.0F;
    }
    const float log_ratio = logf(z / (1.0F - z));
    const float factor = 2.5F - pi_squared_over_six + 0.5F * square(log_ratio) +
        omega_L_V(Q2, z, mf) + L_dip(Q2, z, mf);
    return front * square(K0(argument)) * factor;
}

__device__ __forceinline__ float longitudinal_ab(
    float Q2, float z, float r, float mf, float xi
) {
    // For light quarks b2 and b3 approach b1. Evaluate only this cancellation
    // region in float64, then return to the float32 production accumulation.
    const double Q2d = Q2;
    const double zd = z;
    const double rd = r;
    const double mfd = mf;
    const double xid = xi;
    const double mf2 = mfd * mfd;
    const double kappa2 = zd * (1.0 - zd) * Q2d + mf2;
    const double b1 = modified_bessel_k0_forward<double>(sqrt(kappa2) * rd);
    const double b2 = modified_bessel_k0_forward<double>(
        sqrt(kappa2 + (1.0 - zd) * xid / (1.0 - xid) * mf2) * rd
    );
    const double b3 = modified_bessel_k0_forward<double>(
        sqrt(kappa2 + zd * xid / (1.0 - xid) * mf2) * rd
    );
    const double core = b1 / xid *
        (-2.0 * log(xid) / (1.0 - xid) + 0.5 * (1.0 + xid)) *
        (2.0 * b1 - b2 - b3);
    return static_cast<float>(4.0 * Q2d * pow(zd * (1.0 - zd), 2) * core);
}

__device__ __forceinline__ float longitudinal_cd(
    float Q2, float z, float r, float mf, float xi, float x
) {
    const float one_minus_z = 1.0F - z;
    const float one_minus_xi = 1.0F - xi;
    const float one_minus_x = 1.0F - x;
    const float mf2 = square(mf);
    const float kappa_z2 = z * one_minus_z * Q2 + mf2;
    const float argument = sqrtf(kappa_z2) * r;
    if (argument < 1.0e-7F) {
        return 0.0F;
    }
    const float denominator1 = x * one_minus_xi + xi / one_minus_z;
    const float denominator2 = x * one_minus_xi + xi / z;
    const float CLm1 = square(z) * one_minus_xi / one_minus_z *
        (-square(xi) + x * one_minus_xi *
            (1.0F + one_minus_xi * (1.0F + z * xi / one_minus_z)) / denominator1);
    const float CLm2 = square(one_minus_z) * one_minus_xi / z *
        (-square(xi) + x * one_minus_xi *
            (1.0F + one_minus_xi * (1.0F + one_minus_z * xi / z)) / denominator2);
    const float kappa1 = xi * mf2 / (one_minus_xi * one_minus_x * denominator1) *
        (xi * one_minus_x + x * (1.0F - z * one_minus_xi / one_minus_z));
    const float kappa2 = xi * mf2 / (one_minus_xi * one_minus_x * denominator2) *
        (xi * one_minus_x + x * (1.0F - one_minus_z * one_minus_xi / z));
    const float b1 = K0(argument);
    const float first = (b1 - K0(r * sqrtf(kappa_z2 / one_minus_x + kappa1))) * CLm1 /
        (one_minus_xi * one_minus_x * denominator1 * (x / one_minus_x * kappa_z2 + kappa1));
    const float second = (b1 - K0(r * sqrtf(kappa_z2 / one_minus_x + kappa2))) * CLm2 /
        (one_minus_xi * one_minus_x * denominator2 * (x / one_minus_x * kappa_z2 + kappa2));
    return 4.0F * Q2 * square(z * one_minus_z) * b1 * mf2 * (first + second);
}

__device__ __forceinline__ float transverse_zero(float Q2, float z, float r, float mf) {
    const float Q = sqrtf(Q2);
    const float kappa = sqrtf(z * (1.0F - z) * Q2 + square(mf));
    const float argument = r * kappa;
    const float log_ratio = logf(z / (1.0F - z));
    const float common = -pi_squared_over_six + 0.5F * square(log_ratio) +
        omega_T_V_unsymmetric(Q, z, mf) + omega_T_V_unsymmetric(Q, 1.0F - z, mf) +
        L_dip(Q2, z, mf);
    const float omega_N = omega_T_N_unsymmetric(Q, z, mf) -
        omega_T_N_unsymmetric(Q, 1.0F - z, mf);
    const float term1 = square(kappa * K1(argument)) *
        ((square(z) + square(1.0F - z)) * (2.5F + common) + 0.5F * (2.0F * z - 1.0F) * omega_N);
    const float term2 = square(mf * K0(argument)) * (3.0F + common);
    return term1 + term2;
}

__device__ __forceinline__ double transverse_V1_unsymmetric(
    float Q, float z, float mf, float r, float xi
) {
    const double Qd = Q;
    const double zd = z;
    const double mfd = mf;
    const double rd = r;
    const double xid = xi;
    const double kappa = sqrt(zd * (1.0 - zd) * Qd * Qd + mfd * mfd);
    const double shifted = sqrt(
        kappa * kappa + xid / (1.0 - xid) * (1.0 - zd) * mfd * mfd
    );
    const double difference = shifted * modified_bessel_k1_forward<double>(rd * shifted) -
        kappa * modified_bessel_k1_forward<double>(rd * kappa);
    const double term1 =
        (2.0 * log(xid) / (1.0 - xid) - 0.5 * (1.0 + xid)) / xid * difference;
    const double term2 =
        -(log(xid) / ((1.0 - xid) * (1.0 - xid)) + zd / (1.0 - xid) + 0.5 * zd) *
        (1.0 - zd) * mfd * mfd / shifted *
        modified_bessel_k1_forward<double>(rd * shifted);
    return term1 + term2;
}

__device__ __forceinline__ double transverse_VMS1_unsymmetric(
    float Q, float z, float mf, float r, float xi
) {
    const double Qd = Q;
    const double zd = z;
    const double mfd = mf;
    const double rd = r;
    const double xid = xi;
    const double kappa = sqrt(zd * (1.0 - zd) * Qd * Qd + mfd * mfd);
    const double shifted = sqrt(
        kappa * kappa + xid / (1.0 - xid) * (1.0 - zd) * mfd * mfd
    );
    const double term1 =
        (2.0 * log(xid) / (1.0 - xid) - 0.5 * (1.0 + xid)) / xid *
        (modified_bessel_k0_forward<double>(rd * shifted) -
            modified_bessel_k0_forward<double>(rd * kappa));
    const double term2 = (-1.5 * (1.0 - zd) / (1.0 - xid) + 0.5 * (1.0 - zd)) *
        modified_bessel_k0_forward<double>(rd * shifted);
    return term1 + term2;
}

__device__ __forceinline__ float transverse_one(
    float Q2, float z, float r, float mf, float xi
) {
    const float Q = sqrtf(Q2);
    const float kappa = sqrtf(z * (1.0F - z) * Q2 + square(mf));
    const double V1 = transverse_V1_unsymmetric(Q, z, mf, r, xi) +
        transverse_V1_unsymmetric(Q, 1.0F - z, mf, r, xi);
    const double VMS1 = transverse_VMS1_unsymmetric(Q, z, mf, r, xi) +
        transverse_VMS1_unsymmetric(Q, 1.0F - z, mf, r, xi);
    return static_cast<float>(
        kappa * K1(r * kappa) * (square(z) + square(1.0F - z)) * V1 +
        square(mf) * K0(r * kappa) * VMS1
    );
}

__device__ __forceinline__ float weighted_k1_difference(
    float r, float kappa, float delta_kappa_square
) {
    const float base_square = square(kappa);
    const float shifted_square = base_square + delta_kappa_square;
    if (!(shifted_square > 0.0F && kappa > 0.0F)) {
        return 0.0F;
    }
    const float relative_delta = fabsf(delta_kappa_square) /
        (base_square + fabsf(delta_kappa_square) + 1.0e-30F);
    if (relative_delta < 1.0e-5F) {
        return -0.5F * r * delta_kappa_square * K0(r * kappa);
    }
    const float shifted = sqrtf(shifted_square);
    return shifted * K1(r * shifted) - kappa * K1(r * kappa);
}

__device__ __forceinline__ float transverse_V2_unsymmetric(
    float Q, float z, float mf, float r, float y_chi, float y_u
) {
    const float chi = z * y_chi;
    const float u = (1.0F - y_u) / y_u;
    const float kappa = sqrtf(z * (1.0F - z) * square(Q) + square(mf));
    const float kappa_chi2 = chi * (1.0F - chi) * square(Q) + square(mf);
    const float delta = u * (1.0F - z) / (1.0F - chi) * kappa_chi2;
    const float shifted = sqrtf(square(kappa) + delta);
    const float u_ratio = u / (u + 1.0F);
    const float term1 = -1.0F / (1.0F - chi) / (u * (u + 1.0F)) * square(mf) / kappa_chi2 *
        (2.0F * chi + square(u_ratio) / z * (z - chi) * (1.0F - 2.0F * chi)) *
        weighted_k1_difference(r, kappa, delta);
    const float term2 = -1.0F / square(1.0F - chi) / (u + 1.0F) * (z - chi) *
        (1.0F - 2.0F * u_ratio * (z - chi) + square(u_ratio) / z * square(z - chi)) *
        square(mf) / shifted * K1(r * shifted);
    return z / square(y_u) * (term1 + term2);
}

__device__ __forceinline__ float transverse_VMS2_unsymmetric(
    float Q, float z, float mf, float r, float y_chi, float y_u
) {
    const float chi = z * y_chi;
    const float u = (1.0F - y_u) / y_u;
    const float kappa2 = z * (1.0F - z) * square(Q) + square(mf);
    const float kappa_chi2 = chi * (1.0F - chi) * square(Q) + square(mf);
    const float shifted = sqrtf(kappa2 + u * (1.0F - z) / (1.0F - chi) * kappa_chi2);
    const float term1 = 1.0F / (1.0F - chi) / square(u + 1.0F) *
        (-z - u / (1.0F + u) * (z + u * chi) / z * (chi - (1.0F - z))) * K0(r * shifted);
    const float term2 = 1.0F / ((u + 1.0F) * square(u + 1.0F)) *
        (kappa2 / kappa_chi2 * (1.0F + u * chi * (1.0F - chi) / (z * (1.0F - z))) -
            square(mf) / kappa_chi2 * chi / (1.0F - chi) *
            (2.0F * square(1.0F + u) / u + u / (z * (1.0F - z)) * square(z - chi))) *
        (K0(r * shifted) - K0(r * sqrtf(kappa2)));
    return z / square(y_u) * (term1 + term2);
}

__device__ __forceinline__ float transverse_N_unsymmetric(
    float Q, float z, float mf, float r, float y_chi, float y_u
) {
    const float chi = z * y_chi;
    const float u = (1.0F - y_u) / y_u;
    const float kappa = sqrtf(z * (1.0F - z) * square(Q) + square(mf));
    const float kappa_chi2 = chi * (1.0F - chi) * square(Q) + square(mf);
    const float shifted = sqrtf(square(kappa) + u * (1.0F - z) / (1.0F - chi) * kappa_chi2);
    const float denominator = (u + 1.0F) * square(u + 1.0F);
    const float term1 = 2.0F * (1.0F - z) / z / denominator *
        ((2.0F + u) * u * z + square(u) * chi) * shifted * K1(r * shifted);
    const float term2 = 2.0F * (1.0F - z) / z / denominator * square(mf) / kappa_chi2 *
        (z / (1.0F - z) + chi / (1.0F - chi) * (u - 2.0F * z - 2.0F * u * chi)) *
        (shifted * K1(r * shifted) - kappa * K1(r * kappa));
    return z / square(y_u) * (term1 + term2);
}

__device__ __forceinline__ float transverse_two(
    float Q2, float z, float r, float mf, float y_chi, float y_u
) {
    const float Q = sqrtf(Q2);
    const float kappa = sqrtf(z * (1.0F - z) * Q2 + square(mf));
    const float V2 = transverse_V2_unsymmetric(Q, z, mf, r, y_chi, y_u) +
        transverse_V2_unsymmetric(Q, 1.0F - z, mf, r, y_chi, y_u);
    const float VMS2 = transverse_VMS2_unsymmetric(Q, z, mf, r, y_chi, y_u) +
        transverse_VMS2_unsymmetric(Q, 1.0F - z, mf, r, y_chi, y_u);
    const float N = transverse_N_unsymmetric(Q, z, mf, r, y_chi, y_u) -
        transverse_N_unsymmetric(Q, 1.0F - z, mf, r, y_chi, y_u);
    return kappa * K1(r * kappa) *
        ((square(z) + square(1.0F - z)) * V2 + 0.5F * (2.0F * z - 1.0F) * N) +
        square(mf) * K0(r * kappa) * VMS2;
}

__global__ void lo_integrand_kernel(
    const float* samples,
    int64_t sample_stride,
    int64_t coordinate_stride,
    float* output,
    int64_t count,
    float Q2,
    float xbj,
    bool transverse,
    const float* masses,
    const float* charge_squares,
    int64_t quark_count,
    float maxr,
    int amplitude_mode,
    const float* nodes,
    const float* a,
    const float* b,
    const float* c,
    const float* d,
    int64_t node_count,
    float maxr_interpolate,
    float Qs0_square,
    float lambda,
    float gamma,
    float x0
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float* point = samples + index * sample_stride;
    const float z = point[0];
    const float r = maxr * point[coordinate_stride];
    float wavefunction = 0.0F;
    for (int64_t flavor = 0; flavor < quark_count; ++flavor) {
        const float mf = masses[flavor];
        const float epsilon2 = Q2 * z * (1.0F - z) + square(mf);
        const float argument = r * sqrtf(epsilon2);
        if (argument >= 1.0e-7F && argument <= 5.0e2F) {
            if (transverse) {
                wavefunction += charge_squares[flavor] *
                    ((1.0F - 2.0F * z + 2.0F * square(z)) * epsilon2 * square(K1(argument)) +
                        square(mf * K0(argument)));
            } else {
                wavefunction += charge_squares[flavor] * 4.0F * Q2 * square(z) * square(1.0F - z) *
                    square(K0(argument));
            }
        }
    }
    const float rapidity = fmaxf(logf(x0 / xbj), 0.0F);
    const float dipole = amplitude(
        r, rapidity, amplitude_mode, nodes, a, b, c, d, node_count, maxr_interpolate,
        Qs0_square, lambda, gamma, x0
    );
    const float result = 2.0F * pi * r * maxr * wavefunction * dipole;
    output[index] = isfinite(result) ? result : 0.0F;
}

__global__ void dipole_integrand_kernel(
    const float* samples,
    int64_t sample_stride,
    int64_t coordinate_stride,
    float* output,
    int64_t count,
    int dimensions,
    float Q2,
    float xbj,
    float mf,
    int contribution,
    float maxr,
    bool mass_dependent_x,
    bool smooth_coupling,
    float coupling_C2,
    int active_flavors,
    float maximum_alpha,
    int amplitude_mode,
    const float* nodes,
    const float* a,
    const float* b,
    const float* c,
    const float* d,
    int64_t node_count,
    float maxr_interpolate,
    float minimum_r,
    float maximum_r,
    float Qs0_square,
    float lambda,
    float gamma,
    float x0
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float* point = samples + index * sample_stride;
    const float z = z_min + point[0] * (1.0F - 2.0F * z_min);
    const float raw_r = maxr * point[coordinate_stride];
    if (raw_r < minimum_r || raw_r > maximum_r) {
        output[index] = 0.0F;
        return;
    }
    const float r = fminf(fmaxf(raw_r, minimum_r), maximum_r);
    const float W2 = (1.0F - xbj) * Q2 / xbj;
    const float X = mass_dependent_x
        ? (Q2 + 4.0F * square(mf)) / (Q2 + W2)
        : Q2 / (Q2 + W2);
    const float rapidity = logf(1.0F / X);
    const float N = amplitude(
        r, rapidity, amplitude_mode, nodes, a, b, c, d, node_count, maxr_interpolate,
        Qs0_square, lambda, gamma, x0
    );
    float impact = 0.0F;
    if (contribution == 0) {
        impact = longitudinal_omega(Q2, z, r, mf);
    } else if (contribution == 1) {
        impact = longitudinal_ab(Q2, z, r, mf, point[2 * coordinate_stride]);
    } else if (contribution == 2) {
        impact = longitudinal_cd(
            Q2, z, r, mf, point[2 * coordinate_stride], point[3 * coordinate_stride]
        );
    } else if (contribution == 3) {
        impact = transverse_zero(Q2, z, r, mf);
    } else if (contribution == 4) {
        impact = transverse_one(Q2, z, r, mf, point[2 * coordinate_stride]);
    } else {
        impact = transverse_two(
            Q2, z, r, mf, point[2 * coordinate_stride], point[3 * coordinate_stride]
        );
    }
    const float alpha_bar = running_coupling(
        square(r), coupling_C2, active_flavors, maximum_alpha, smooth_coupling
    ) * color_factor / pi;
    const float jacobian = raw_r * maxr * (1.0F - 2.0F * z_min);
    const float result = N * impact * jacobian * alpha_bar;
    output[index] = isfinite(result) ? result : 0.0F;
}

void validate_common(
    const torch::Tensor& samples,
    const torch::Tensor& nodes,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c,
    const torch::Tensor& d,
    int dimensions,
    int amplitude_mode
) {
    TORCH_CHECK(
        samples.is_cuda() && samples.scalar_type() == torch::kFloat32 && samples.dim() == 2 &&
            samples.size(1) == dimensions && samples.stride(0) > 0 && samples.stride(1) > 0,
        "DIS samples must be a positive-stride CUDA float32 matrix with the expected dimensions"
    );
    for (const auto& tensor : {nodes, a, b, c, d}) {
        TORCH_CHECK(
            tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32 && tensor.dim() == 1 &&
                tensor.is_contiguous() && tensor.device() == samples.device(),
            "DIS amplitude data must be contiguous CUDA float32 vectors on the sample device"
        );
    }
    TORCH_CHECK(amplitude_mode == 0 || amplitude_mode == 1, "invalid DIS amplitude mode");
    if (amplitude_mode == 1) {
        TORCH_CHECK(nodes.numel() >= 3, "a BK spline needs at least three nodes");
        TORCH_CHECK(
            a.numel() == nodes.numel() - 1 && b.numel() == a.numel() &&
                c.numel() == a.numel() && d.numel() == a.numel(),
            "invalid BK spline coefficient sizes"
        );
    }
}

torch::Tensor dis_lo_integrand(
    torch::Tensor samples,
    double Q2,
    double xbj,
    bool transverse,
    torch::Tensor masses,
    torch::Tensor charge_squares,
    double maxr,
    int64_t amplitude_mode,
    torch::Tensor nodes,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d,
    double maxr_interpolate,
    double Qs0_square,
    double lambda,
    double gamma,
    double x0
) {
    validate_common(samples, nodes, a, b, c, d, 2, amplitude_mode);
    TORCH_CHECK(
        masses.is_cuda() && charge_squares.is_cuda() && masses.device() == samples.device() &&
            charge_squares.device() == samples.device() && masses.scalar_type() == torch::kFloat32 &&
            charge_squares.scalar_type() == torch::kFloat32 && masses.dim() == 1 &&
            charge_squares.dim() == 1 && masses.numel() == charge_squares.numel() &&
            masses.numel() > 0 && masses.is_contiguous() && charge_squares.is_contiguous(),
        "LO quark masses and charges must be matching contiguous CUDA float32 vectors"
    );
    auto output = torch::empty({samples.size(0)}, samples.options());
    if (samples.size(0) == 0) {
        return output;
    }
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (samples.size(0) + threads - 1) / threads;
    lo_integrand_kernel<<<blocks, threads, 0, stream>>>(
        samples.data_ptr<float>(), samples.stride(0), samples.stride(1), output.data_ptr<float>(),
        samples.size(0), static_cast<float>(Q2), static_cast<float>(xbj), transverse,
        masses.data_ptr<float>(), charge_squares.data_ptr<float>(), masses.numel(),
        static_cast<float>(maxr), amplitude_mode, nodes.data_ptr<float>(), a.data_ptr<float>(),
        b.data_ptr<float>(), c.data_ptr<float>(), d.data_ptr<float>(), nodes.numel(),
        static_cast<float>(maxr_interpolate), static_cast<float>(Qs0_square),
        static_cast<float>(lambda), static_cast<float>(gamma), static_cast<float>(x0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor dis_dipole_integrand(
    torch::Tensor samples,
    double Q2,
    double xbj,
    double mf,
    int64_t contribution,
    double maxr,
    bool mass_dependent_x,
    bool smooth_coupling,
    double coupling_C2,
    int64_t active_flavors,
    double maximum_alpha,
    int64_t amplitude_mode,
    torch::Tensor nodes,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d,
    double maxr_interpolate,
    double minimum_r,
    double maximum_r,
    double Qs0_square,
    double lambda,
    double gamma,
    double x0
) {
    TORCH_CHECK(contribution >= 0 && contribution < 6, "invalid dipole contribution");
    const int dimensions = contribution == 0 || contribution == 3 ? 2 :
        (contribution == 1 || contribution == 4 ? 3 : 4);
    validate_common(samples, nodes, a, b, c, d, dimensions, amplitude_mode);
    auto output = torch::empty({samples.size(0)}, samples.options());
    if (samples.size(0) == 0) {
        return output;
    }
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (samples.size(0) + threads - 1) / threads;
    dipole_integrand_kernel<<<blocks, threads, 0, stream>>>(
        samples.data_ptr<float>(), samples.stride(0), samples.stride(1), output.data_ptr<float>(),
        samples.size(0), dimensions, static_cast<float>(Q2), static_cast<float>(xbj),
        static_cast<float>(mf), contribution, static_cast<float>(maxr), mass_dependent_x,
        smooth_coupling, static_cast<float>(coupling_C2), active_flavors,
        static_cast<float>(maximum_alpha), amplitude_mode, nodes.data_ptr<float>(),
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), d.data_ptr<float>(),
        nodes.numel(), static_cast<float>(maxr_interpolate), static_cast<float>(minimum_r),
        static_cast<float>(maximum_r), static_cast<float>(Qs0_square), static_cast<float>(lambda),
        static_cast<float>(gamma), static_cast<float>(x0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("dis_lo_integrand", &dis_lo_integrand, "Fused LO DIS integrand");
    module.def("dis_dipole_integrand", &dis_dipole_integrand, "Fused NLO dipole-sector integrand");
}
