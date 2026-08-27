#include <cmath>
#include <type_traits>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Math.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int threads = 128;
constexpr double pi = 3.14159265358979323846;
constexpr double lambda_qcd = 0.241;
constexpr double color_factor = 4.0 / 3.0;

template <typename scalar_t>
__device__ __forceinline__ scalar_t square(scalar_t value) {
    return value * value;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t device_sqrt(scalar_t value) {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return sqrtf(value);
    }
    return sqrt(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t device_exp(scalar_t value) {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return expf(value);
    }
    return exp(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t device_log(scalar_t value) {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return logf(value);
    }
    return log(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t device_log1p(scalar_t value) {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return log1pf(value);
    }
    return log1p(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t device_pow(scalar_t base, scalar_t exponent) {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return powf(base, exponent);
    }
    return pow(base, exponent);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t device_cos(scalar_t value) {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return cosf(value);
    }
    return cos(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t device_abs(scalar_t value) {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return fabsf(value);
    }
    return fabs(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t device_min(scalar_t first, scalar_t second) {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return fminf(first, second);
    }
    return fmin(first, second);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t device_max(scalar_t first, scalar_t second) {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return fmaxf(first, second);
    }
    return fmax(first, second);
}

template <typename scalar_t>
struct Geometry {
    scalar_t z0;
    scalar_t x20x21;
    scalar_t Qbar_j_square;
    scalar_t Qbar_k_square;
    scalar_t omega_j;
    scalar_t omega_k;
    scalar_t lambda_j;
    scalar_t lambda_k;
    scalar_t x2_j_square;
    scalar_t x2_k_square;
    scalar_t x3_j_square;
    scalar_t x3_k_square;
    scalar_t x2j_x3j;
    scalar_t x2k_x3k;
    scalar_t x2j_x3k;
    scalar_t x2k_x3j;
    scalar_t x3j_x3k;
};

template <bool transverse, typename scalar_t>
__device__ __forceinline__ Geometry<scalar_t> tripole_geometry(
    scalar_t Q2,
    scalar_t z1,
    scalar_t z2,
    scalar_t x01_square,
    scalar_t x02_square,
    scalar_t x21_square
) {
    const scalar_t z0 = scalar_t(1) - z1 - z2;
    const scalar_t x20x21 = scalar_t(-0.5) * (x01_square - x21_square - x02_square);
    const scalar_t z0_z2 = z0 + z2;
    const scalar_t z1_z2 = z1 + z2;
    const scalar_t Q = device_sqrt(Q2);
    const scalar_t Qbar_j = Q * device_sqrt(z1 * (scalar_t(1) - z1));
    const scalar_t Qbar_k = Q * device_sqrt(z0 * (scalar_t(1) - z0));
    const scalar_t omega_j = z0 * z2 / (z1 * square(z0_z2));
    const scalar_t omega_k = z1 * z2 / (z0 * square(z1_z2));
    const scalar_t lambda_j = z1 * z2 / z0;
    const scalar_t lambda_k = z0 * z2 / z1;
    const scalar_t x3_j_square = square(z0) / square(z0_z2) * x02_square +
        x21_square - scalar_t(2) * z0 / z0_z2 * x20x21;
    const scalar_t x3_k_square = square(z1) / square(z1_z2) * x21_square +
        x02_square - scalar_t(2) * z1 / z1_z2 * x20x21;
    scalar_t x2j_x3j = 0;
    scalar_t x2k_x3k = 0;
    scalar_t x2j_x3k = 0;
    scalar_t x2k_x3j = 0;
    scalar_t x3j_x3k = 0;
    if constexpr (transverse) {
        x2j_x3j = x20x21 - z0 / z0_z2 * x02_square;
        x2k_x3k = -x20x21 + z1 / z1_z2 * x21_square;
        x2j_x3k = -x02_square + z1 / z1_z2 * x20x21;
        x2k_x3j = x21_square - z0 / z0_z2 * x20x21;
        x3j_x3k = z0 / z0_z2 * x02_square + z1 / z1_z2 * x21_square -
            (scalar_t(1) + z0 * z1 / (z0_z2 * z1_z2)) * x20x21;
    }
    return {
        z0,
        x20x21,
        square(Qbar_j),
        square(Qbar_k),
        omega_j,
        omega_k,
        lambda_j,
        lambda_k,
        x02_square,
        x21_square,
        x3_j_square,
        x3_k_square,
        x2j_x3j,
        x2k_x3k,
        x2j_x3k,
        x2k_x3j,
        x3j_x3k,
    };
}

template <typename scalar_t>
struct OuterBesselValues {
    scalar_t K0;
    scalar_t K1;
    scalar_t G12_sing;
    scalar_t G22_sing;
    scalar_t H;
};

template <typename scalar_t>
__device__ __forceinline__ OuterBesselValues<scalar_t> outer_bessel_values(
    scalar_t Qbar_square,
    scalar_t mf_square,
    scalar_t x2_square,
    scalar_t x3_square,
    scalar_t omega,
    scalar_t lambda
) {
    const scalar_t A0 = Qbar_square + mf_square;
    const scalar_t B0 = x3_square + omega * x2_square;
    const scalar_t argument = device_sqrt(A0 * B0);
    const scalar_t K0 = modified_bessel_k0_forward<scalar_t>(argument);
    const scalar_t K1 = modified_bessel_k1_forward<scalar_t>(argument);
    const scalar_t root_ratio_K1 = device_sqrt(A0 / B0) * K1;
    const scalar_t AH = Qbar_square + mf_square * (scalar_t(1) + lambda);
    const scalar_t H = scalar_t(4) * device_sqrt(AH / B0) *
        modified_bessel_k1_forward<scalar_t>(device_sqrt(AH * B0));
    return {
        K0,
        K1,
        K0 / x2_square,
        root_ratio_K1 / x2_square,
        H,
    };
}

template <typename scalar_t>
struct Impacts {
    scalar_t dipole;
    scalar_t tripole;
};

template <typename scalar_t>
__device__ __forceinline__ Impacts<scalar_t> longitudinal_impacts(
    scalar_t Q2,
    scalar_t mf_square,
    scalar_t z1,
    scalar_t z2,
    scalar_t x01_square,
    const Geometry<scalar_t>& g
) {
    constexpr double exp_euler = 1.7810724179901979852;
    const scalar_t z0 = g.z0;
    const scalar_t z0_z2 = z0 + z2;
    const scalar_t z1_z2 = z1 + z2;
    const scalar_t coefficient_j = scalar_t(2) * z0 * z0_z2 + square(z2);
    const scalar_t coefficient_k = scalar_t(2) * z1 * z1_z2 + square(z2);
    const scalar_t A_j = g.Qbar_j_square + mf_square;
    const scalar_t A_k = g.Qbar_k_square + mf_square;
    const scalar_t tripole_K0_j = modified_bessel_k0_forward<scalar_t>(device_sqrt(
        A_j * (g.x3_j_square + g.omega_j * g.x2_j_square)
    ));
    const scalar_t tripole_K0_k = modified_bessel_k0_forward<scalar_t>(device_sqrt(
        A_k * (g.x3_k_square + g.omega_k * g.x2_k_square)
    ));
    const scalar_t parent_K0_j =
        modified_bessel_k0_forward<scalar_t>(device_sqrt(A_j * x01_square));
    const scalar_t parent_K0_k =
        modified_bessel_k0_forward<scalar_t>(device_sqrt(A_k * x01_square));

    const scalar_t tripole_j = square(z1) * coefficient_j / g.x2_j_square *
        square(tripole_K0_j);
    const scalar_t tripole_k = square(z0) * coefficient_k / g.x2_k_square *
        square(tripole_K0_k);
    const scalar_t tripole_jk = -scalar_t(2) * z0 * z1 *
        (z0 * (scalar_t(1) - z0) + z1 * (scalar_t(1) - z1)) * g.x20x21 /
        (g.x2_j_square * g.x2_k_square) * tripole_K0_j * tripole_K0_k;

    const scalar_t dipole_j = -square(z1) * coefficient_j / g.x2_j_square *
        device_exp(-g.x2_j_square / (x01_square * scalar_t(exp_euler))) *
        square(parent_K0_j);
    const scalar_t dipole_k = -square(z0) * coefficient_k / g.x2_k_square *
        device_exp(-g.x2_k_square / (x01_square * scalar_t(exp_euler))) *
        square(parent_K0_k);
    return {
        scalar_t(4) * Q2 * (dipole_j + dipole_k),
        scalar_t(4) * Q2 * (tripole_j + tripole_k + tripole_jk),
    };
}

template <typename scalar_t>
__device__ __forceinline__ Impacts<scalar_t> transverse_impacts(
    scalar_t mf_square,
    scalar_t z1,
    scalar_t z2,
    scalar_t x01_square,
    const Geometry<scalar_t>& g
) {
    constexpr double exp_euler = 1.7810724179901979852;
    const scalar_t z0 = g.z0;
    const scalar_t z0_z2 = z0 + z2;
    const scalar_t z1_z2 = z1 + z2;
    const scalar_t coefficient_j =
        (scalar_t(2) * z0 * z0_z2 + square(z2)) / square(z0_z2);
    const scalar_t coefficient_k =
        (scalar_t(2) * z1 * z1_z2 + square(z2)) / square(z1_z2);
    const scalar_t A_j = g.Qbar_j_square + mf_square;
    const scalar_t A_k = g.Qbar_k_square + mf_square;
    const scalar_t B_j = g.x3_j_square + g.omega_j * g.x2_j_square;
    const scalar_t B_k = g.x3_k_square + g.omega_k * g.x2_k_square;
    const OuterBesselValues<scalar_t> j = outer_bessel_values(
        g.Qbar_j_square,
        mf_square,
        g.x2_j_square,
        g.x3_j_square,
        g.omega_j,
        g.lambda_j
    );
    const OuterBesselValues<scalar_t> k = outer_bessel_values(
        g.Qbar_k_square,
        mf_square,
        g.x2_k_square,
        g.x3_k_square,
        g.omega_k,
        g.lambda_k
    );
    const scalar_t parent_argument_j = device_sqrt(A_j * x01_square);
    const scalar_t parent_argument_k = device_sqrt(A_k * x01_square);
    const scalar_t parent_K0_j = modified_bessel_k0_forward<scalar_t>(parent_argument_j);
    const scalar_t parent_K1_j = modified_bessel_k1_forward<scalar_t>(parent_argument_j);
    const scalar_t parent_K0_k = modified_bessel_k0_forward<scalar_t>(parent_argument_k);
    const scalar_t parent_K1_k = modified_bessel_k1_forward<scalar_t>(parent_argument_k);
    const scalar_t subtraction_j = -device_exp(
        -g.x2_j_square / (x01_square * scalar_t(exp_euler))
    );
    const scalar_t subtraction_k = -device_exp(
        -g.x2_k_square / (x01_square * scalar_t(exp_euler))
    );

    const scalar_t dipole_jk =
        coefficient_j * (scalar_t(1) - scalar_t(2) * z1 * (scalar_t(1) - z1)) *
            A_j / g.x2_j_square * subtraction_j * square(parent_K1_j) +
        coefficient_k * (scalar_t(1) - scalar_t(2) * z0 * (scalar_t(1) - z0)) *
            A_k / g.x2_k_square * subtraction_k * square(parent_K1_k);
    const scalar_t dipole_jkm = mf_square * (
        coefficient_j / g.x2_j_square * subtraction_j * square(parent_K0_j) +
        coefficient_k / g.x2_k_square * subtraction_k * square(parent_K0_k)
    );

    const scalar_t tripole_jk =
        coefficient_j * (scalar_t(1) - scalar_t(2) * z1 * (scalar_t(1) - z1)) *
            A_j / g.x2_j_square * g.x3_j_square / B_j * square(j.K1) +
        coefficient_k * (scalar_t(1) - scalar_t(2) * z0 * (scalar_t(1) - z0)) *
            A_k / g.x2_k_square * g.x3_k_square / B_k * square(k.K1);
    const scalar_t tripole_jkm = mf_square * (
        coefficient_j / g.x2_j_square * square(j.K0) +
        coefficient_k / g.x2_k_square * square(k.K0)
    );

    const scalar_t common_F =
        z2 * square(z0 - z1) *
            (g.x2j_x3j * g.x2k_x3k - g.x2k_x3j * g.x2j_x3k) -
        (z1 * z0_z2 + z0 * z1_z2) *
            (z0 * z0_z2 + z1 * z1_z2) * g.x20x21 * g.x3j_x3k;
    const scalar_t F_1 = scalar_t(4) / (z0_z2 * z1_z2) *
        common_F * j.G22_sing * k.G22_sing;
    const scalar_t F_2j = -z0_z2 * z1 * z2 / square(z1_z2) *
        g.x2j_x3j * k.H * j.G22_sing;
    const scalar_t F_2k = z1_z2 * z0 * z2 / square(z0_z2) *
        g.x2k_x3k * j.H * k.G22_sing;
    const scalar_t F_3j = -square(z0) * z1 * z2 /
        (z0_z2 * square(z0_z2)) * g.x2j_x3j * j.H * j.G22_sing;
    const scalar_t F_3k = square(z1) * z0 * z2 /
        (z1_z2 * square(z1_z2)) * g.x2k_x3k * k.H * k.G22_sing;
    const scalar_t F_4j = square(z0 * z2) /
        (scalar_t(8) * square(square(z0_z2))) * square(j.H);
    const scalar_t F_4k = square(z1 * z2) /
        (scalar_t(8) * square(square(z1_z2))) * square(k.H);
    const scalar_t tripole_F = scalar_t(0.5) *
        (F_1 + F_2j + F_2k + F_3j + F_3k + F_4j + F_4k);

    const scalar_t tripole_Fm = scalar_t(0.5) * mf_square * (
        -((scalar_t(2) * z0 + z2) * (scalar_t(2) * z1 + z2) + square(z2)) *
        g.x20x21 * scalar_t(8) * j.G12_sing * scalar_t(8) * k.G12_sing /
        (scalar_t(32) * z0_z2 * z1_z2)
    );
    return {
        dipole_jk + dipole_jkm,
        tripole_jk + tripole_jkm + tripole_F + tripole_Fm,
    };
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t gbw_amplitude(
    scalar_t r_square,
    scalar_t rapidity,
    scalar_t Qs0_square,
    scalar_t lambda,
    scalar_t gamma,
    scalar_t initial_rapidity
) {
    const scalar_t effective_rapidity = device_max(rapidity, initial_rapidity);
    const scalar_t Qs_square = Qs0_square * device_exp(lambda * effective_rapidity);
    const scalar_t argument = scalar_t(0.25) *
        device_pow(r_square * Qs_square, gamma);
    return device_abs(argument) < scalar_t(1.0e-7)
        ? argument
        : scalar_t(1) - device_exp(-argument);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t running_coupling(
    scalar_t r_square,
    scalar_t C2,
    int active_flavors,
    scalar_t maximum_alpha,
    bool smooth
) {
    const scalar_t b0 = (scalar_t(33) - scalar_t(2 * active_flavors)) /
        (scalar_t(12) * scalar_t(pi));
    const scalar_t scale = scalar_t(4) * C2 /
        (r_square * square(scalar_t(lambda_qcd)));
    if (!smooth) {
        const scalar_t alpha = scalar_t(1) / (b0 * device_log(scale));
        return (alpha > maximum_alpha || scale < scalar_t(1))
            ? maximum_alpha
            : alpha;
    }
    const scalar_t freeze_c = scalar_t(0.2);
    const scalar_t first = scalar_t(2) / freeze_c * device_log(scalar_t(2.5));
    const scalar_t second = device_log(scale) / freeze_c;
    const scalar_t larger = device_max(first, second);
    const scalar_t smaller = device_min(first, second);
    const scalar_t log_sum = larger + device_log1p(device_exp(smaller - larger));
    return scalar_t(1) / (b0 * freeze_c * log_sum);
}

// The classifier deliberately uses float32 because it partitions the phase space
// seen by the regular float32 calculation. Both kernels call this same function.
__device__ __forceinline__ bool is_sensitive_sample(
    const float* unit_sample,
    int64_t dimension_stride,
    float maxr,
    float ratio
) {
    const float x01 = maxr * unit_sample[2 * dimension_stride];
    const float x02 = maxr * unit_sample[3 * dimension_stride];
    const float angle = static_cast<float>(2.0 * pi) * unit_sample[4 * dimension_stride];
    const float x01_square = square(x01);
    const float x02_square = square(x02);
    const float cross = __fmul_rn(
        __fmul_rn(__fmul_rn(2.0F, x01), x02),
        cosf(angle)
    );
    const float x21_square = fmaxf(
        __fsub_rn(__fadd_rn(x01_square, x02_square), cross),
        0.0F
    );
    return fminf(x02_square, x21_square) < ratio * x01_square;
}

template <bool transverse, typename scalar_t>
__device__ __forceinline__ scalar_t evaluate_sample(
    const float* Q2_input,
    const float* xbj_input,
    scalar_t mf,
    const float* unit_sample,
    int64_t dimension_stride,
    scalar_t maxr,
    scalar_t Qs0_square,
    scalar_t lambda,
    scalar_t gamma,
    scalar_t x0,
    bool finite_Nc,
    bool parent_coupling,
    bool smooth_coupling,
    scalar_t coupling_C2,
    int active_flavors,
    scalar_t maximum_alpha
) {
    const scalar_t Q2 = scalar_t(Q2_input[0]);
    const scalar_t xbj = scalar_t(xbj_input[0]);
    const scalar_t z2_min = xbj / Q2;
    if (z2_min > scalar_t(1)) {
        return 0;
    }

    const scalar_t z1 = (scalar_t(1) - z2_min) * scalar_t(unit_sample[0]);
    const scalar_t z2 = (scalar_t(1) - z1 - z2_min) *
        scalar_t(unit_sample[dimension_stride]) + z2_min;
    const scalar_t x01 = maxr * scalar_t(unit_sample[2 * dimension_stride]);
    const scalar_t x02 = maxr * scalar_t(unit_sample[3 * dimension_stride]);
    const scalar_t angle = scalar_t(2.0 * pi) *
        scalar_t(unit_sample[4 * dimension_stride]);
    const scalar_t x01_square = square(x01);
    const scalar_t x02_square = square(x02);
    const scalar_t x21_square = device_max(
        x01_square + x02_square - scalar_t(2) * x01 * x02 * device_cos(angle),
        scalar_t(0)
    );
    const Geometry<scalar_t> geometry = tripole_geometry<transverse>(
        Q2,
        z1,
        z2,
        x01_square,
        x02_square,
        x21_square
    );
    const scalar_t mf_square = square(mf);
    Impacts<scalar_t> impact;
    if constexpr (transverse) {
        impact = transverse_impacts(mf_square, z1, z2, x01_square, geometry);
    } else {
        impact = longitudinal_impacts(Q2, mf_square, z1, z2, x01_square, geometry);
    }

    const scalar_t rapidity = device_log(Q2 / xbj * z2);
    const scalar_t initial_rapidity = device_log(scalar_t(1) / x0);
    const scalar_t N01 = gbw_amplitude(
        x01_square, rapidity, Qs0_square, lambda, gamma, initial_rapidity
    );
    const scalar_t S01 = scalar_t(1) - N01;
    const scalar_t S02 = scalar_t(1) - gbw_amplitude(
        x02_square, rapidity, Qs0_square, lambda, gamma, initial_rapidity
    );
    const scalar_t S12 = scalar_t(1) - gbw_amplitude(
        x21_square, rapidity, Qs0_square, lambda, gamma, initial_rapidity
    );
    const scalar_t tripole = finite_Nc
        ? scalar_t(1) - scalar_t(1.125) * (S02 * S12 - S01 / scalar_t(9))
        : scalar_t(1) - S02 * S12;
    const scalar_t coupling_r_square = parent_coupling
        ? x01_square
        : device_min(x01_square, device_min(x02_square, x21_square));
    const scalar_t alpha = running_coupling(
        coupling_r_square,
        coupling_C2,
        active_flavors,
        maximum_alpha,
        smooth_coupling
    );
    const scalar_t jacobian = (scalar_t(1) - z2_min) *
        (scalar_t(1) - z1 - z2_min) * x01 * x02 * square(maxr) * scalar_t(2.0 * pi);
    const scalar_t result = (N01 * impact.dipole + tripole * impact.tripole) *
        jacobian * (alpha * scalar_t(color_factor) / scalar_t(pi)) / z2;
    return rapidity >= scalar_t(0) && isfinite(result) ? result : scalar_t(0);
}

template <bool transverse>
__global__ void regular_kernel(
    const float* __restrict__ Q2,
    const float* __restrict__ xbj,
    float mf,
    const float* __restrict__ unit_samples,
    int64_t sample_stride,
    int64_t dimension_stride,
    float* __restrict__ output,
    float sensitive_ratio,
    float maxr,
    float Qs0_square,
    float lambda,
    float gamma,
    float x0,
    bool finite_Nc,
    bool parent_coupling,
    bool smooth_coupling,
    float coupling_C2,
    int active_flavors,
    float maximum_alpha,
    int64_t samples
) {
    const int64_t sample = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sample >= samples) {
        return;
    }
    const float* unit_sample = unit_samples + sample * sample_stride;
    output[sample] = is_sensitive_sample(
        unit_sample, dimension_stride, maxr, sensitive_ratio
    ) ? 0.0F : evaluate_sample<transverse, float>(
        Q2,
        xbj,
        mf,
        unit_sample,
        dimension_stride,
        maxr,
        Qs0_square,
        lambda,
        gamma,
        x0,
        finite_Nc,
        parent_coupling,
        smooth_coupling,
        coupling_C2,
        active_flavors,
        maximum_alpha
    );
}

template <bool transverse>
__global__ void sensitive_kernel(
    const float* __restrict__ Q2,
    const float* __restrict__ xbj,
    double mf,
    const float* __restrict__ unit_samples,
    int64_t sample_stride,
    int64_t dimension_stride,
    double* __restrict__ output,
    float sensitive_ratio,
    double maxr,
    double Qs0_square,
    double lambda,
    double gamma,
    double x0,
    bool finite_Nc,
    bool parent_coupling,
    bool smooth_coupling,
    double coupling_C2,
    int active_flavors,
    double maximum_alpha,
    int64_t samples
) {
    const int64_t sample = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sample >= samples) {
        return;
    }
    const float* unit_sample = unit_samples + sample * sample_stride;
    output[sample] = is_sensitive_sample(
        unit_sample,
        dimension_stride,
        static_cast<float>(maxr),
        sensitive_ratio
    ) ? evaluate_sample<transverse, double>(
        Q2,
        xbj,
        mf,
        unit_sample,
        dimension_stride,
        maxr,
        Qs0_square,
        lambda,
        gamma,
        x0,
        finite_Nc,
        parent_coupling,
        smooth_coupling,
        coupling_C2,
        active_flavors,
        maximum_alpha
    ) : 0.0;
}

template <bool transverse, bool sensitive_region, typename scalar_t>
__global__ void region_sum_kernel(
    const float* __restrict__ Q2,
    const float* __restrict__ xbj,
    scalar_t mf,
    const float* __restrict__ unit_samples,
    int64_t sample_stride,
    int64_t dimension_stride,
    const float* __restrict__ sample_weights,
    int64_t weight_stride,
    scalar_t* __restrict__ block_sums,
    float sensitive_ratio,
    scalar_t maxr,
    scalar_t Qs0_square,
    scalar_t lambda,
    scalar_t gamma,
    scalar_t x0,
    bool finite_Nc,
    bool parent_coupling,
    bool smooth_coupling,
    scalar_t coupling_C2,
    int active_flavors,
    scalar_t maximum_alpha,
    int64_t samples
) {
    __shared__ scalar_t partial[threads];
    const int64_t sample = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    scalar_t value = 0;
    if (sample < samples) {
        const float* unit_sample = unit_samples + sample * sample_stride;
        const bool sensitive = is_sensitive_sample(
            unit_sample,
            dimension_stride,
            static_cast<float>(maxr),
            sensitive_ratio
        );
        if (sensitive == sensitive_region) {
            value = evaluate_sample<transverse, scalar_t>(
                Q2,
                xbj,
                mf,
                unit_sample,
                dimension_stride,
                maxr,
                Qs0_square,
                lambda,
                gamma,
                x0,
                finite_Nc,
                parent_coupling,
                smooth_coupling,
                coupling_C2,
                active_flavors,
                maximum_alpha
            ) * scalar_t(sample_weights[sample * weight_stride]);
        }
    }
    partial[threadIdx.x] = value;
    __syncthreads();

#pragma unroll
    for (int offset = threads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            partial[threadIdx.x] += partial[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = partial[0];
    }
}

std::vector<torch::Tensor> dis_gbw_i1_regions(
    torch::Tensor Q2,
    torch::Tensor xbj,
    double mf,
    torch::Tensor unit_samples,
    bool transverse,
    double sensitive_ratio,
    double maxr,
    double Qs0_square,
    double lambda,
    double gamma,
    double x0,
    bool finite_Nc,
    bool parent_coupling,
    bool smooth_coupling,
    double coupling_C2,
    int active_flavors,
    double maximum_alpha
) {
    TORCH_CHECK(
        Q2.is_cuda() && Q2.scalar_type() == torch::kFloat32 && Q2.numel() == 1 &&
            xbj.is_cuda() && xbj.scalar_type() == torch::kFloat32 && xbj.numel() == 1 &&
            Q2.get_device() == xbj.get_device(),
        "DIS I1 requires matching CUDA float32 scalar Q2 and xbj"
    );
    TORCH_CHECK(
        unit_samples.is_cuda() && unit_samples.scalar_type() == torch::kFloat32 &&
            unit_samples.dim() == 2 && unit_samples.size(1) == 5 &&
            unit_samples.get_device() == Q2.get_device() &&
            unit_samples.stride(0) > 0 && unit_samples.stride(1) > 0,
        "DIS I1 samples must be CUDA float32 [samples, 5] with positive strides"
    );
    TORCH_CHECK(
        sensitive_ratio > 0.0 && sensitive_ratio < 1.0 && maxr > 0.0 &&
            Qs0_square > 0.0 && gamma > 0.0 && x0 > 0.0 && coupling_C2 > 0.0 &&
            active_flavors >= 0 && maximum_alpha > 0.0,
        "DIS I1 scalar parameters must be physical"
    );

    const int64_t samples = unit_samples.size(0);
    auto regular = torch::empty({samples}, unit_samples.options());
    auto sensitive = torch::empty(
        {samples},
        unit_samples.options().dtype(torch::kFloat64)
    );
    if (samples == 0) {
        return {regular, sensitive};
    }

    const int blocks = (samples + threads - 1) / threads;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_I1_REGIONS(TRANSVERSE) \
    regular_kernel<TRANSVERSE><<<blocks, threads, 0, stream>>>( \
        Q2.data_ptr<float>(), \
        xbj.data_ptr<float>(), \
        static_cast<float>(mf), \
        unit_samples.data_ptr<float>(), \
        unit_samples.stride(0), \
        unit_samples.stride(1), \
        regular.data_ptr<float>(), \
        static_cast<float>(sensitive_ratio), \
        static_cast<float>(maxr), \
        static_cast<float>(Qs0_square), \
        static_cast<float>(lambda), \
        static_cast<float>(gamma), \
        static_cast<float>(x0), \
        finite_Nc, \
        parent_coupling, \
        smooth_coupling, \
        static_cast<float>(coupling_C2), \
        active_flavors, \
        static_cast<float>(maximum_alpha), \
        samples \
    ); \
    sensitive_kernel<TRANSVERSE><<<blocks, threads, 0, stream>>>( \
        Q2.data_ptr<float>(), \
        xbj.data_ptr<float>(), \
        mf, \
        unit_samples.data_ptr<float>(), \
        unit_samples.stride(0), \
        unit_samples.stride(1), \
        sensitive.data_ptr<double>(), \
        static_cast<float>(sensitive_ratio), \
        maxr, \
        Qs0_square, \
        lambda, \
        gamma, \
        x0, \
        finite_Nc, \
        parent_coupling, \
        smooth_coupling, \
        coupling_C2, \
        active_flavors, \
        maximum_alpha, \
        samples \
    )
    if (transverse) {
        LAUNCH_I1_REGIONS(true);
    } else {
        LAUNCH_I1_REGIONS(false);
    }
#undef LAUNCH_I1_REGIONS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {regular, sensitive};
}

std::vector<torch::Tensor> dis_gbw_i1_region_sums(
    torch::Tensor Q2,
    torch::Tensor xbj,
    double mf,
    torch::Tensor unit_samples,
    torch::Tensor sample_weights,
    bool transverse,
    double sensitive_ratio,
    double maxr,
    double Qs0_square,
    double lambda,
    double gamma,
    double x0,
    bool finite_Nc,
    bool parent_coupling,
    bool smooth_coupling,
    double coupling_C2,
    int active_flavors,
    double maximum_alpha
) {
    TORCH_CHECK(
        Q2.is_cuda() && Q2.scalar_type() == torch::kFloat32 && Q2.numel() == 1 &&
            xbj.is_cuda() && xbj.scalar_type() == torch::kFloat32 && xbj.numel() == 1 &&
            Q2.get_device() == xbj.get_device(),
        "DIS I1 requires matching CUDA float32 scalar Q2 and xbj"
    );
    TORCH_CHECK(
        unit_samples.is_cuda() && unit_samples.scalar_type() == torch::kFloat32 &&
            unit_samples.dim() == 2 && unit_samples.size(1) == 5 &&
            unit_samples.get_device() == Q2.get_device() &&
            unit_samples.stride(0) > 0 && unit_samples.stride(1) > 0,
        "DIS I1 samples must be CUDA float32 [samples, 5] with positive strides"
    );
    TORCH_CHECK(
        sample_weights.is_cuda() && sample_weights.scalar_type() == torch::kFloat32 &&
            sample_weights.dim() == 1 && sample_weights.numel() == unit_samples.size(0) &&
            sample_weights.get_device() == Q2.get_device() && sample_weights.stride(0) > 0,
        "DIS I1 weights must be a matching CUDA float32 vector"
    );
    TORCH_CHECK(
        sensitive_ratio > 0.0 && sensitive_ratio < 1.0 && maxr > 0.0 &&
            Qs0_square > 0.0 && gamma > 0.0 && x0 > 0.0 && coupling_C2 > 0.0 &&
            active_flavors >= 0 && maximum_alpha > 0.0,
        "DIS I1 scalar parameters must be physical"
    );

    const int64_t samples = unit_samples.size(0);
    const int blocks = (samples + threads - 1) / threads;
    auto regular_sums = torch::empty({blocks}, unit_samples.options());
    auto sensitive_sums = torch::empty(
        {blocks},
        unit_samples.options().dtype(torch::kFloat64)
    );
    if (samples == 0) {
        return {regular_sums, sensitive_sums};
    }

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_I1_REGION_SUMS(TRANSVERSE) \
    region_sum_kernel<TRANSVERSE, false, float><<<blocks, threads, 0, stream>>>( \
        Q2.data_ptr<float>(), \
        xbj.data_ptr<float>(), \
        static_cast<float>(mf), \
        unit_samples.data_ptr<float>(), \
        unit_samples.stride(0), \
        unit_samples.stride(1), \
        sample_weights.data_ptr<float>(), \
        sample_weights.stride(0), \
        regular_sums.data_ptr<float>(), \
        static_cast<float>(sensitive_ratio), \
        static_cast<float>(maxr), \
        static_cast<float>(Qs0_square), \
        static_cast<float>(lambda), \
        static_cast<float>(gamma), \
        static_cast<float>(x0), \
        finite_Nc, \
        parent_coupling, \
        smooth_coupling, \
        static_cast<float>(coupling_C2), \
        active_flavors, \
        static_cast<float>(maximum_alpha), \
        samples \
    ); \
    region_sum_kernel<TRANSVERSE, true, double><<<blocks, threads, 0, stream>>>( \
        Q2.data_ptr<float>(), \
        xbj.data_ptr<float>(), \
        mf, \
        unit_samples.data_ptr<float>(), \
        unit_samples.stride(0), \
        unit_samples.stride(1), \
        sample_weights.data_ptr<float>(), \
        sample_weights.stride(0), \
        sensitive_sums.data_ptr<double>(), \
        static_cast<float>(sensitive_ratio), \
        maxr, \
        Qs0_square, \
        lambda, \
        gamma, \
        x0, \
        finite_Nc, \
        parent_coupling, \
        smooth_coupling, \
        coupling_C2, \
        active_flavors, \
        maximum_alpha, \
        samples \
    )
    if (transverse) {
        LAUNCH_I1_REGION_SUMS(true);
    } else {
        LAUNCH_I1_REGION_SUMS(false);
    }
#undef LAUNCH_I1_REGION_SUMS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {regular_sums, sensitive_sums};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "dis_gbw_i1_regions",
        &dis_gbw_i1_regions,
        "GBW DIS I1 split into regular float32 and singular float64 regions"
    );
    module.def(
        "dis_gbw_i1_region_sums",
        &dis_gbw_i1_region_sums,
        "Weighted GBW DIS I1 block sums for regular and singular regions"
    );
}
