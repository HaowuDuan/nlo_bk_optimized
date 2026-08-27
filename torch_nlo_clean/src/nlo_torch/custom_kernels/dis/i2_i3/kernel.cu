#include <cmath>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Math.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int threads = 128;
constexpr int transverse_channels = 8;
constexpr int longitudinal_channels = 4;
constexpr float pi = 3.14159265358979323846F;
constexpr float lambda_qcd = 0.241F;
constexpr float color_factor = 4.0F / 3.0F;

__device__ __forceinline__ float square(float value) {
    return value * value;
}

// ATen's float K0 implementation promotes log(0.5 * x) to float64 for x <= 2
// because 0.5 is an untyped literal. Keep the same polynomial coefficients and
// recurrence while evaluating that expression natively in float32.
__device__ __forceinline__ float bessel_k0(float x) {
    if (x > 2.0F || !(x > 0.0F)) {
        return modified_bessel_k0_forward<float>(x);
    }
    constexpr float coefficients[] = {
        +1.37446543561352307156e-16F,
        +4.25981614279661018399e-14F,
        +1.03496952576338420167e-11F,
        +1.90451637722020886025e-09F,
        +2.53479107902614945675e-07F,
        +2.28621210311945178607e-05F,
        +1.26461541144692592338e-03F,
        +3.59799365153615016266e-02F,
        +3.44289899924628486886e-01F,
        -5.35327393233902768720e-01F,
    };
    float previous = 0.0F;
    float current = 0.0F;
    float next = coefficients[0];
#pragma unroll
    for (int index = 1; index < 10; ++index) {
        previous = current;
        current = next;
        next = (x * x - 2.0F) * current - previous + coefficients[index];
    }
    return 0.5F * (next - previous) -
        logf(0.5F * x) * modified_bessel_i0_forward<float>(x);
}

template <int count>
__device__ __forceinline__ float bessel_polynomial(
    float argument,
    const float (&coefficients)[count]
) {
    float previous = 0.0F;
    float current = 0.0F;
    float next = coefficients[0];
#pragma unroll
    for (int index = 1; index < count; ++index) {
        previous = current;
        current = next;
        next = argument * current - previous + coefficients[index];
    }
    return 0.5F * (next - previous);
}

struct BesselK01 {
    float K0;
    float K1;
};

// K0 and K1 use the same asymptotic transformation for x > 2. Evaluate their
// recurrences independently, but share the exponential and square root.
__device__ __forceinline__ BesselK01 bessel_k01(float x) {
    if (!(x > 0.0F)) {
        return {
            bessel_k0(x),
            modified_bessel_k1_forward<float>(x),
        };
    }
    if (x <= 2.0F) {
        constexpr float K0_coefficients[] = {
            +1.37446543561352307156e-16F,
            +4.25981614279661018399e-14F,
            +1.03496952576338420167e-11F,
            +1.90451637722020886025e-09F,
            +2.53479107902614945675e-07F,
            +2.28621210311945178607e-05F,
            +1.26461541144692592338e-03F,
            +3.59799365153615016266e-02F,
            +3.44289899924628486886e-01F,
            -5.35327393233902768720e-01F,
        };
        constexpr float K1_coefficients[] = {
            -7.02386347938628759343e-18F,
            -2.42744985051936593393e-15F,
            -6.66690169419932900609e-13F,
            -1.41148839263352776110e-10F,
            -2.21338763073472585583e-08F,
            -2.43340614156596823496e-06F,
            -1.73028895751305206302e-04F,
            -6.97572385963986435018e-03F,
            -1.22611180822657148235e-01F,
            -3.53155960776544875667e-01F,
            +1.52530022733894777053e+00F,
        };
        constexpr float I0_coefficients[] = {
            -4.41534164647933937950e-18F,
            +3.33079451882223809783e-17F,
            -2.43127984654795469359e-16F,
            +1.71539128555513303061e-15F,
            -1.16853328779934516808e-14F,
            +7.67618549860493561688e-14F,
            -4.85644678311192946090e-13F,
            +2.95505266312963983461e-12F,
            -1.72682629144155570723e-11F,
            +9.67580903537323691224e-11F,
            -5.18979560163526290666e-10F,
            +2.65982372468238665035e-09F,
            -1.30002500998624804212e-08F,
            +6.04699502254191894932e-08F,
            -2.67079385394061173391e-07F,
            +1.11738753912010371815e-06F,
            -4.41673835845875056359e-06F,
            +1.64484480707288970893e-05F,
            -5.75419501008210370398e-05F,
            +1.88502885095841655729e-04F,
            -5.76375574538582365885e-04F,
            +1.63947561694133579842e-03F,
            -4.32430999505057594430e-03F,
            +1.05464603945949983183e-02F,
            -2.37374148058994688156e-02F,
            +4.93052842396707084878e-02F,
            -9.49010970480476444210e-02F,
            +1.71620901522208775349e-01F,
            -3.04682672343198398683e-01F,
            +6.76795274409476084995e-01F,
        };
        constexpr float I1_coefficients[] = {
            +2.77791411276104639959e-18F,
            -2.11142121435816608115e-17F,
            +1.55363195773620046921e-16F,
            -1.10559694773538630805e-15F,
            +7.60068429473540693410e-15F,
            -5.04218550472791168711e-14F,
            +3.22379336594557470981e-13F,
            -1.98397439776494371520e-12F,
            +1.17361862988909016308e-11F,
            -6.66348972350202774223e-11F,
            +3.62559028155211703701e-10F,
            -1.88724975172282928790e-09F,
            +9.38153738649577178388e-09F,
            -4.44505912879632808065e-08F,
            +2.00329475355213526229e-07F,
            -8.56872026469545474066e-07F,
            +3.47025130813767847674e-06F,
            -1.32731636560394358279e-05F,
            +4.78156510755005422638e-05F,
            -1.61760815825896745588e-04F,
            +5.12285956168575772895e-04F,
            -1.51357245063125314899e-03F,
            +4.15642294431288815669e-03F,
            -1.05640848946261981558e-02F,
            +2.47264490306265168283e-02F,
            -5.29459812080949914269e-02F,
            +1.02643658689847095384e-01F,
            -1.76416518357834055153e-01F,
            +2.52587186443633654823e-01F,
        };
        const float K_argument = x * x - 2.0F;
        const float I_argument = x / 2.0F - 2.0F;
        const float exponential = expf(x);
        const float logarithm = logf(0.5F * x);
        const float I0 = exponential * bessel_polynomial(I_argument, I0_coefficients);
        const float I1 = bessel_polynomial(I_argument, I1_coefficients) * x * exponential;
        return {
            bessel_polynomial(K_argument, K0_coefficients) - logarithm * I0,
            logarithm * I1 + bessel_polynomial(K_argument, K1_coefficients) / x,
        };
    }
    constexpr float K0_coefficients[] = {
        +5.30043377268626276149e-18F,
        -1.64758043015242134646e-17F,
        +5.21039150503902756861e-17F,
        -1.67823109680541210385e-16F,
        +5.51205597852431940784e-16F,
        -1.84859337734377901440e-15F,
        +6.34007647740507060557e-15F,
        -2.22751332699166985548e-14F,
        +8.03289077536357521100e-14F,
        -2.98009692317273043925e-13F,
        +1.14034058820847496303e-12F,
        -4.51459788337394416547e-12F,
        +1.85594911495471785253e-11F,
        -7.95748924447710747776e-11F,
        +3.57739728140030116597e-10F,
        -1.69753450938905987466e-09F,
        +8.57403401741422608519e-09F,
        -4.66048989768794782956e-08F,
        +2.76681363944501510342e-07F,
        -1.83175552271911948767e-06F,
        +1.39498137188764993662e-05F,
        -1.28495495816278026384e-04F,
        +1.56988388573005337491e-03F,
        -3.14481013119645005427e-02F,
        +2.44030308206595545468e+00F,
    };
    constexpr float K1_coefficients[] = {
        -5.75674448366501715755e-18F,
        +1.79405087314755922667e-17F,
        -5.68946255844285935196e-17F,
        +1.83809354436663880070e-16F,
        -6.05704724837331885336e-16F,
        +2.03870316562433424052e-15F,
        -7.01983709041831346144e-15F,
        +2.47715442448130437068e-14F,
        -8.97670518232499435011e-14F,
        +3.34841966607842919884e-13F,
        -1.28917396095102890680e-12F,
        +5.13963967348173025100e-12F,
        -2.12996783842756842877e-11F,
        +9.21831518760500529508e-11F,
        -4.19035475934189648750e-10F,
        +2.01504975519703286596e-09F,
        -1.03457624656780970260e-08F,
        +5.74108412545004946722e-08F,
        -3.50196060308781257119e-07F,
        +2.40648494783721712015e-06F,
        -1.93619797416608296024e-05F,
        +1.95215518471351631108e-04F,
        -2.85781685962277938680e-03F,
        +1.03923736576817238437e-01F,
        +2.72062619048444266945e+00F,
    };
    const float polynomial_argument = 8.0F / x - 2.0F;
    const float K0_polynomial = bessel_polynomial(polynomial_argument, K0_coefficients);
    const float K1_polynomial = bessel_polynomial(polynomial_argument, K1_coefficients);
    const float exponential = expf(-x);
    const float root = sqrtf(x);
    return {
        exponential * K0_polynomial / root,
        exponential * K1_polynomial / root,
    };
}

__device__ __forceinline__ float gbw_S_matrix(
    float r_square,
    float rapidity,
    float Qs0_square,
    float lambda,
    float gamma,
    float initial_rapidity
) {
    const float effective_rapidity = fmaxf(rapidity, initial_rapidity);
    const float Qs_square = Qs0_square * expf(lambda * effective_rapidity);
    const float argument = 0.25F * powf(r_square * Qs_square, gamma);
    const float amplitude = fabsf(argument) < 1.0e-7F
        ? argument
        : 1.0F - expf(-argument);
    return 1.0F - amplitude;
}

__device__ __forceinline__ float dis_running_coupling(
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
    const float log_sum = larger + log1pf(expf(smaller - larger));
    return 1.0F / (b0 * freeze_c * log_sum);
}

struct Geometry {
    float z0;
    float x20x21;
    float Qbar_j_square;
    float Qbar_k_square;
    float omega_j;
    float omega_k;
    float lambda_j;
    float lambda_k;
    float x2_j_square;
    float x2_k_square;
    float x3_j_square;
    float x3_k_square;
    float x2j_x3j;
    float x2k_x3k;
    float x2j_x3k;
    float x2k_x3j;
    float x3j_x3k;
};

struct GBWSampleCoordinates {
    float z1;
    float z2;
    float x01;
    float x02;
    float x01_square;
    float x02_square;
    float x21_square;
};

template <bool transverse>
__device__ __forceinline__ Geometry tripole_geometry(
    float Q2,
    float z1,
    float z2,
    float x01_square,
    float x02_square,
    float x21_square
) {
    const float z0 = 1.0F - z1 - z2;
    const float x20x21 = -0.5F * (x01_square - x21_square - x02_square);
    const float z0_z2 = z0 + z2;
    const float z1_z2 = z1 + z2;
    const float Qbar_j_square = Q2 * z1 * (1.0F - z1);
    const float Qbar_k_square = Q2 * z0 * (1.0F - z0);
    const float omega_j = z0 * z2 / (z1 * square(z0_z2));
    const float omega_k = z1 * z2 / (z0 * square(z1_z2));
    const float lambda_j = z1 * z2 / z0;
    const float lambda_k = z0 * z2 / z1;
    const float x3_j_square = square(z0 / z0_z2) * x02_square + x21_square -
        2.0F * z0 / z0_z2 * x20x21;
    const float x3_k_square = square(z1 / z1_z2) * x21_square + x02_square -
        2.0F * z1 / z1_z2 * x20x21;
    float x2j_x3j = 0.0F;
    float x2k_x3k = 0.0F;
    float x2j_x3k = 0.0F;
    float x2k_x3j = 0.0F;
    float x3j_x3k = 0.0F;
    if constexpr (transverse) {
        x2j_x3j = x20x21 - z0 / z0_z2 * x02_square;
        x2k_x3k = -x20x21 + z1 / z1_z2 * x21_square;
        x2j_x3k = -x02_square + z1 / z1_z2 * x20x21;
        x2k_x3j = x21_square - z0 / z0_z2 * x20x21;
        x3j_x3k = z0 / z0_z2 * x02_square + z1 / z1_z2 * x21_square -
            (1.0F + z0 * z1 / (z0_z2 * z1_z2)) * x20x21;
    }
    return {
        z0,
        x20x21,
        Qbar_j_square,
        Qbar_k_square,
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

struct GMomentsAtNode {
    float G22_bar;
    float G12_bar;
    float G21;
    float G11;
};

struct LongitudinalMomentsAtNode {
    float G12_bar;
    float G11;
};

__device__ __forceinline__ float longitudinal_g12_bar_value(
    float Qbar_square,
    float mf_square,
    float x2_square,
    float x3_square,
    float omega,
    float lambda,
    float y
);

__device__ __forceinline__ LongitudinalMomentsAtNode longitudinal_g_values_paired(
    float Qbar_square,
    float mf_square,
    float x2_square,
    float x3_square,
    float omega,
    float lambda,
    float y
) {
    const float A0 = Qbar_square + mf_square;
    const float A = A0 + y * lambda * mf_square;
    const float B = y * x3_square + omega * x2_square;
    const float inverse_y = 1.0F / y;
    const float inverse_sqrt_y = rsqrtf(y);
    const float argument = sqrtf(A * B * inverse_y);
    const float argument_zero = sqrtf(A0 * B * inverse_y);
    const BesselK01 K = bessel_k01(argument);
    const float G12 = 4.0F * omega * inverse_y * inverse_sqrt_y *
        sqrtf(A / B) * K.K1;
    const float G12_zero = 4.0F * omega * inverse_y * inverse_sqrt_y *
        sqrtf(A0 / B) * modified_bessel_k1_forward<float>(argument_zero);
    return {
        G12 - G12_zero,
        2.0F * inverse_y * K.K0,
    };
}

__device__ __forceinline__ LongitudinalMomentsAtNode longitudinal_g_values(
    float Qbar_square,
    float mf_square,
    float x2_square,
    float x3_square,
    float omega,
    float lambda,
    float y
) {
    const float A0 = Qbar_square + mf_square;
    const float A = A0 + y * lambda * mf_square;
    const float B = y * x3_square + omega * x2_square;
    const float inverse_y = 1.0F / y;
    const float argument = sqrtf(A * B * inverse_y);
    return {
        longitudinal_g12_bar_value(
            Qbar_square, mf_square, x2_square, x3_square, omega, lambda, y
        ),
        2.0F * inverse_y * bessel_k0(argument),
    };
}

__device__ __forceinline__ float longitudinal_g12_bar_value(
    float Qbar_square,
    float mf_square,
    float x2_square,
    float x3_square,
    float omega,
    float lambda,
    float y
) {
    const float A0 = Qbar_square + mf_square;
    const float A = A0 + y * lambda * mf_square;
    const float B = y * x3_square + omega * x2_square;
    const float inverse_y = 1.0F / y;
    const float inverse_sqrt_y = rsqrtf(y);
    const float argument = sqrtf(A * B * inverse_y);
    const float argument_zero = sqrtf(A0 * B * inverse_y);
    const float G12 = 4.0F * omega * inverse_y * inverse_sqrt_y *
        sqrtf(A / B) * modified_bessel_k1_forward<float>(argument);
    const float G12_zero = 4.0F * omega * inverse_y * inverse_sqrt_y *
        sqrtf(A0 / B) * modified_bessel_k1_forward<float>(argument_zero);
    return G12 - G12_zero;
}

__device__ __forceinline__ GMomentsAtNode g_values(
    float Qbar_square,
    float mf_square,
    float x2_square,
    float x3_square,
    float omega,
    float lambda,
    float y
) {
    const float A0 = Qbar_square + mf_square;
    const float A = A0 + y * lambda * mf_square;
    const float B = y * x3_square + omega * x2_square;
    const float inverse_y = 1.0F / y;
    const float inverse_sqrt_y = rsqrtf(y);
    const float argument = sqrtf(A * B * inverse_y);
    const float argument_zero = sqrtf(A0 * B * inverse_y);
    const BesselK01 K = bessel_k01(argument);
    const BesselK01 K_zero = bessel_k01(argument_zero);
    const float ratio = A / B;
    const float ratio_zero = A0 / B;
    const float root_ratio = sqrtf(ratio);
    const float root_ratio_zero = sqrtf(ratio_zero);
    const float G12 = 4.0F * omega * inverse_y * inverse_sqrt_y * root_ratio * K.K1;
    const float G12_zero =
        4.0F * omega * inverse_y * inverse_sqrt_y * root_ratio_zero * K_zero.K1;
    const float K2 = K.K0 + 2.0F * K.K1 / argument;
    const float K2_zero = K_zero.K0 + 2.0F * K_zero.K1 / argument_zero;
    const float G22 = 8.0F * omega * inverse_y * ratio * K2;
    const float G22_zero = 8.0F * omega * inverse_y * ratio_zero * K2_zero;
    return {
        G22 - G22_zero,
        G12 - G12_zero,
        4.0F * inverse_sqrt_y * root_ratio * K.K1,
        2.0F * inverse_y * K.K0,
    };
}

struct OuterBesselValues {
    float K0;
    float root_ratio_K1;
    float G12_sing;
    float G22_sing;
    float H;
};

__device__ __forceinline__ OuterBesselValues outer_bessel_values(
    float Qbar_square,
    float mf_square,
    float x2_square,
    float x3_square,
    float omega,
    float lambda
) {
    const float A0 = Qbar_square + mf_square;
    const float B0 = x3_square + omega * x2_square;
    const float argument = sqrtf(A0 * B0);
    const BesselK01 K = bessel_k01(argument);
    const float root_ratio_K1 = sqrtf(A0 / B0) * K.K1;
    const float AH = Qbar_square + mf_square * (1.0F + lambda);
    const float H = 4.0F * sqrtf(AH / B0) *
        modified_bessel_k1_forward<float>(sqrtf(AH * B0));
    return {
        K.K0,
        root_ratio_K1,
        K.K0 / x2_square,
        root_ratio_K1 / x2_square,
        H,
    };
}

__device__ __forceinline__ float longitudinal_I2(
    float Q2,
    float mf_square,
    float z1,
    float z2,
    const Geometry& geometry,
    float G12_bar_j,
    float G12_bar_k
) {
    const float z0 = geometry.z0;
    const float B_j = bessel_k0(sqrtf(
        (geometry.Qbar_j_square + mf_square) *
        (geometry.x3_j_square + geometry.omega_j * geometry.x2_j_square)
    ));
    const float B_k = bessel_k0(sqrtf(
        (geometry.Qbar_k_square + mf_square) *
        (geometry.x3_k_square + geometry.omega_k * geometry.x2_k_square)
    ));
    const float term_j = square(z1) * (2.0F * z0 * (z0 + z2) + square(z2)) /
        4.0F * G12_bar_j * B_j;
    const float term_k = square(z0) * (2.0F * z1 * (z1 + z2) + square(z2)) /
        4.0F * G12_bar_k * B_k;
    const float term_jk = -z0 * z1 *
        (z0 * (1.0F - z0) + z1 * (1.0F - z1)) * geometry.x20x21 / 4.0F *
        (G12_bar_j * B_k / geometry.x2_k_square +
         G12_bar_k * B_j / geometry.x2_j_square);
    return 4.0F * Q2 * (term_j + term_k + term_jk);
}

__device__ __forceinline__ float longitudinal_I3(
    float Q2,
    float mf_square,
    float z1,
    float z2,
    const Geometry& geometry,
    float G12_bar_j,
    float G12_bar_k,
    float G11_j,
    float G11_k
) {
    const float z0 = geometry.z0;
    const float term_j = square(z1) * (2.0F * z0 * (z0 + z2) + square(z2)) *
        geometry.x2_j_square / 64.0F * square(G12_bar_j);
    const float term_k = square(z0) * (2.0F * z1 * (z1 + z2) + square(z2)) *
        geometry.x2_k_square / 64.0F * square(G12_bar_k);
    const float term_jk = -z1 * z0 *
        (z1 * (1.0F - z1) + z0 * (1.0F - z0)) * geometry.x20x21 /
        32.0F * G12_bar_j * G12_bar_k;
    const float ratio_j = z1 / (z0 + z2);
    const float ratio_k = z0 / (z1 + z2);
    const float term_mf = mf_square / 16.0F * square(square(z2)) *
        (square(ratio_j * G11_j) + square(ratio_k * G11_k) -
         2.0F * ratio_j * ratio_k * G11_j * G11_k);
    return 4.0F * Q2 * (term_j + term_k + term_jk + term_mf);
}

__device__ __forceinline__ float transverse_I2(
    float mf_square,
    float z1,
    float z2,
    const Geometry& g,
    const float* moments
) {
    const float G22_bar_j = moments[0];
    const float G22_bar_k = moments[1];
    const float G12_bar_j = moments[2];
    const float G12_bar_k = moments[3];
    const float G21_j = moments[4];
    const float G21_k = moments[5];
    const float G11_j = moments[6];
    const float G11_k = moments[7];
    const float z0 = g.z0;
    const float z0_z2 = z0 + z2;
    const float z1_z2 = z1 + z2;
    const OuterBesselValues j = outer_bessel_values(
        g.Qbar_j_square,
        mf_square,
        g.x2_j_square,
        g.x3_j_square,
        g.omega_j,
        g.lambda_j
    );
    const OuterBesselValues k = outer_bessel_values(
        g.Qbar_k_square,
        mf_square,
        g.x2_k_square,
        g.x3_k_square,
        g.omega_k,
        g.lambda_k
    );

    const float jk_j = (2.0F * z0 * z0_z2 + square(z2)) / square(z0_z2) *
        (1.0F - 2.0F * z1 * (1.0F - z1)) * G22_bar_j *
        g.x3_j_square / 8.0F * j.root_ratio_K1;
    const float jk_k = (2.0F * z1 * z1_z2 + square(z2)) / square(z1_z2) *
        (1.0F - 2.0F * z0 * (1.0F - z0)) * G22_bar_k *
        g.x3_k_square / 8.0F * k.root_ratio_K1;
    const float term_jkm = mf_square * (
        (2.0F * z0 * z0_z2 + square(z2)) / square(z0_z2) * G12_bar_j /
            4.0F * j.K0 +
        (2.0F * z1 * z1_z2 + square(z2)) / square(z1_z2) * G12_bar_k /
            4.0F * k.K0
    );

    const float common_F =
        (z2 * square(z0 - z1) *
            (g.x2j_x3j * g.x2k_x3k - g.x2k_x3j * g.x2j_x3k) -
         (z1 * z0_z2 + z0 * z1_z2) *
            (z0 * z0_z2 + z1 * z1_z2) * g.x20x21 * g.x3j_x3k) /
        (4.0F * z0_z2 * z1_z2);
    const float F_1 = common_F *
        (G22_bar_k * j.G22_sing + G22_bar_j * k.G22_sing);
    const float F_2j = -z0_z2 * z1 * z2 / (16.0F * square(z1_z2)) *
        g.x2j_x3j * k.H * G22_bar_j;
    const float F_2k = z1_z2 * z0 * z2 / (16.0F * square(z0_z2)) *
        g.x2k_x3k * j.H * G22_bar_k;
    const float F_3j = -square(z0) * z1 * z2 / (16.0F * z0_z2 * square(z0_z2)) *
        g.x2j_x3j * j.H * G22_bar_j;
    const float F_3k = square(z1) * z0 * z2 / (16.0F * z1_z2 * square(z1_z2)) *
        g.x2k_x3k * k.H * G22_bar_k;
    const float term_F = 0.5F * (F_1 + F_2j + F_2k + F_3j + F_3k);

    const float Fm_1j = -z0 * z1 * square(z2) /
        (16.0F * z0_z2 * square(z0_z2)) * g.x2j_x3j * G21_j * 8.0F * j.G12_sing;
    const float Fm_1k = z0 * z1 * square(z2) /
        (16.0F * z1_z2 * square(z1_z2)) * g.x2k_x3k * G21_k * 8.0F * k.G12_sing;
    const float Fm_2 = -((2.0F * z0 + z2) * (2.0F * z1 + z2) + square(z2)) *
        g.x20x21 / (32.0F * z0_z2 * z1_z2) *
        (G12_bar_k * 8.0F * j.G12_sing + G12_bar_j * 8.0F * k.G12_sing);
    const float Fm_3j = -square(z0 * z2) /
        (16.0F * z0_z2 * square(z1_z2)) *
        g.x2j_x3k * G21_k * 8.0F * j.G12_sing;
    const float Fm_3k = square(z1 * z2) /
        (16.0F * square(z0_z2) * z1_z2) *
        g.x2k_x3j * G21_j * 8.0F * k.G12_sing;
    const float Fm_4j = -z0 * z1 * square(z2) /
        (16.0F * z0_z2 * square(z0_z2)) *
        g.x2j_x3j * G11_j * 16.0F * j.G22_sing;
    const float Fm_4k = z0 * z1 * square(z2) /
        (16.0F * z1_z2 * square(z1_z2)) *
        g.x2k_x3k * G11_k * 16.0F * k.G22_sing;
    const float Fm_5j = -z0_z2 * square(z2) / (16.0F * square(z1_z2)) *
        g.x2j_x3j * G11_k * 16.0F * j.G22_sing;
    const float Fm_5k = z1_z2 * square(z2) / (16.0F * square(z0_z2)) *
        g.x2k_x3k * G11_j * 16.0F * k.G22_sing;
    const float Fm_6j = z0 * z2 * square(z2) / (4.0F * square(square(z0_z2))) *
        j.H * G11_j;
    const float Fm_6k = z1 * z2 * square(z2) / (4.0F * square(square(z1_z2))) *
        k.H * G11_k;
    const float term_Fm = 0.5F * mf_square *
        (Fm_1j + Fm_1k + Fm_2 + Fm_3j + Fm_3k + Fm_4j + Fm_4k +
         Fm_5j + Fm_5k + Fm_6j + Fm_6k);
    return jk_j + jk_k + term_jkm + term_F + term_Fm;
}

__device__ __forceinline__ float transverse_I3(
    float mf_square,
    float z1,
    float z2,
    const Geometry& g,
    const float* m
) {
    const float G22_bar_j = m[0];
    const float G22_bar_k = m[1];
    const float G12_bar_j = m[2];
    const float G12_bar_k = m[3];
    const float G21_j = m[4];
    const float G21_k = m[5];
    const float G11_j = m[6];
    const float G11_k = m[7];
    const float z0 = g.z0;
    const float z0_z2 = z0 + z2;
    const float z1_z2 = z1 + z2;
    const float term_jkm = mf_square * (
        (2.0F * z0 * z0_z2 + square(z2)) / square(z0_z2) *
            g.x2_j_square / 64.0F * square(G12_bar_j) +
        (2.0F * z1 * z1_z2 + square(z2)) / square(z1_z2) *
            g.x2_k_square / 64.0F * square(G12_bar_k)
    );
    const float term_jk =
        (2.0F * z0 * z0_z2 + square(z2)) / square(z0_z2) *
            (1.0F - 2.0F * z1 * (1.0F - z1)) *
            g.x3_j_square * g.x2_j_square / 256.0F * square(G22_bar_j) +
        (2.0F * z1 * z1_z2 + square(z2)) / square(z1_z2) *
            (1.0F - 2.0F * z0 * (1.0F - z0)) *
            g.x3_k_square * g.x2_k_square / 256.0F * square(G22_bar_k);
    const float term_F = 0.5F * G22_bar_j * G22_bar_k /
        (64.0F * z0_z2 * z1_z2) *
        (z2 * square(z0 - z1) *
            (g.x2j_x3j * g.x2k_x3k - g.x2k_x3j * g.x2j_x3k) -
         (z1 * z0_z2 + z0 * z1_z2) *
            (z0 * z0_z2 + z1 * z1_z2) * g.x20x21 * g.x3j_x3k);

    const float Fm_1j = square(square(z2)) / (64.0F * square(square(z0_z2))) *
        (4.0F * z1 * (z1 - 1.0F) + 2.0F) * g.x3_j_square * square(G21_j);
    const float Fm_1k = square(square(z2)) / (64.0F * square(square(z1_z2))) *
        (4.0F * z0 * (z0 - 1.0F) + 2.0F) * g.x3_k_square * square(G21_k);
    const float Fm_2j = -z0 * z1 * square(z2) /
        (16.0F * z0_z2 * square(z0_z2)) * g.x2j_x3j * G12_bar_j * G21_j;
    const float Fm_2k = z0 * z1 * square(z2) /
        (16.0F * z1_z2 * square(z1_z2)) * g.x2k_x3k * G12_bar_k * G21_k;
    const float Fm_3a = -((2.0F * z0 + z2) * (2.0F * z1 + z2) + square(z2)) *
        g.x20x21 / (32.0F * z0_z2 * z1_z2) * G12_bar_j * G12_bar_k;
    const float Fm_3b = square(square(z2)) /
        (32.0F * square(z0_z2) * square(z1_z2)) *
        ((2.0F * z0 + z2) * (2.0F * z1 + z2) + square(z2)) *
        g.x3j_x3k * G21_j * G21_k;
    const float Fm_4j = mf_square / 8.0F * square(square(z2 / z0_z2)) * square(G11_j);
    const float Fm_4k = mf_square / 8.0F * square(square(z2 / z1_z2)) * square(G11_k);
    const float Fm_5j = -square(z0 * z2) / (16.0F * z0_z2 * square(z1_z2)) *
        g.x2j_x3k * G12_bar_j * G21_k;
    const float Fm_5k = square(z1 * z2) / (16.0F * z1_z2 * square(z0_z2)) *
        g.x2k_x3j * G12_bar_k * G21_j;
    const float Fm_6j = -z0 * z1 * square(z2) /
        (16.0F * z0_z2 * square(z0_z2)) * g.x2j_x3j * G11_j * G22_bar_j;
    const float Fm_6k = z0 * z1 * square(z2) /
        (16.0F * z1_z2 * square(z1_z2)) * g.x2k_x3k * G11_k * G22_bar_k;
    const float Fm_7j = -z0_z2 * square(z2) / (16.0F * square(z1_z2)) *
        g.x2j_x3j * G11_k * G22_bar_j;
    const float Fm_7k = z1_z2 * square(z2) / (16.0F * square(z0_z2)) *
        g.x2k_x3k * G11_j * G22_bar_k;
    const float term_Fm = 0.5F * mf_square *
        (Fm_1j + Fm_2j + Fm_1k + Fm_2k + Fm_3a + Fm_3b + Fm_4j + Fm_4k +
         Fm_5j + Fm_5k + Fm_6j + Fm_6k + Fm_7j + Fm_7k);
    return term_jk + term_jkm + term_F + term_Fm;
}

template <bool transverse>
__global__ void dis_nested_kernel(
    const float* __restrict__ Q2_input,
    float mf,
    const float* __restrict__ z1,
    const float* __restrict__ z2,
    const float* __restrict__ x01_square,
    const float* __restrict__ x02_square,
    const float* __restrict__ x21_square,
    const float* __restrict__ nodes,
    const float* __restrict__ weights,
    float* __restrict__ output_I2,
    float* __restrict__ output_I3,
    int points
) {
    const int sample = blockIdx.x;
    const int node = threadIdx.x;
    constexpr int channels = transverse ? transverse_channels : longitudinal_channels;
    __shared__ float partial[transverse_channels][threads];
    const float Q2 = Q2_input[0];
    const float sample_z1 = z1[sample];
    const float sample_z2 = z2[sample];
    const Geometry geometry = tripole_geometry<transverse>(
        Q2,
        sample_z1,
        sample_z2,
        x01_square[sample],
        x02_square[sample],
        x21_square[sample]
    );
    float values[channels] = {};
    if (node < points) {
        const float y = nodes[node];
        const float weight = weights[node];
        const float mf_square = square(mf);
        if constexpr (transverse) {
            const GMomentsAtNode j = g_values(
                geometry.Qbar_j_square,
                mf_square,
                geometry.x2_j_square,
                geometry.x3_j_square,
                geometry.omega_j,
                geometry.lambda_j,
                y
            );
            const GMomentsAtNode k = g_values(
                geometry.Qbar_k_square,
                mf_square,
                geometry.x2_k_square,
                geometry.x3_k_square,
                geometry.omega_k,
                geometry.lambda_k,
                y
            );
            values[0] = weight * j.G22_bar;
            values[1] = weight * k.G22_bar;
            values[2] = weight * j.G12_bar;
            values[3] = weight * k.G12_bar;
            values[4] = weight * j.G21;
            values[5] = weight * k.G21;
            values[6] = weight * j.G11;
            values[7] = weight * k.G11;
        } else {
            const LongitudinalMomentsAtNode j = longitudinal_g_values(
                geometry.Qbar_j_square,
                mf_square,
                geometry.x2_j_square,
                geometry.x3_j_square,
                geometry.omega_j,
                geometry.lambda_j,
                y
            );
            const LongitudinalMomentsAtNode k = longitudinal_g_values(
                geometry.Qbar_k_square,
                mf_square,
                geometry.x2_k_square,
                geometry.x3_k_square,
                geometry.omega_k,
                geometry.lambda_k,
                y
            );
            values[0] = weight * j.G12_bar;
            values[1] = weight * k.G12_bar;
            values[2] = weight * j.G11;
            values[3] = weight * k.G11;
        }
    }
#pragma unroll
    for (int channel = 0; channel < channels; ++channel) {
        partial[channel][node] = values[channel];
    }
    __syncthreads();

    if (node < channels) {
        float sum = 0.0F;
        for (int index = 0; index < points; ++index) {
            sum += partial[node][index];
        }
        partial[node][0] = sum;
    }
    __syncthreads();

    if (node == 0) {
        const float mf_square = square(mf);
        if constexpr (transverse) {
            float moments[transverse_channels];
#pragma unroll
            for (int channel = 0; channel < transverse_channels; ++channel) {
                moments[channel] = partial[channel][0];
            }
            output_I2[sample] = transverse_I2(
                mf_square, sample_z1, sample_z2, geometry, moments
            );
            output_I3[sample] = transverse_I3(
                mf_square, sample_z1, sample_z2, geometry, moments
            );
        } else {
            output_I2[sample] = longitudinal_I2(
                Q2,
                mf_square,
                sample_z1,
                sample_z2,
                geometry,
                partial[0][0],
                partial[1][0]
            );
            output_I3[sample] = longitudinal_I3(
                Q2,
                mf_square,
                sample_z1,
                sample_z2,
                geometry,
                partial[0][0],
                partial[1][0],
                partial[2][0],
                partial[3][0]
            );
        }
    }
}

template <
    bool transverse,
    bool contribution_I3,
    bool both_contributions = false,
    bool split_transverse_moments = false
>
__global__ void dis_nested_gbw_integrand_kernel(
    const float* __restrict__ Q2_input,
    const float* __restrict__ xbj_input,
    float mf,
    const float* __restrict__ unit_samples,
    int64_t sample_stride,
    int64_t dimension_stride,
    const float* __restrict__ nodes,
    const float* __restrict__ weights,
    float* __restrict__ output,
    float* __restrict__ output_I3,
    int points,
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
    float maximum_alpha
) {
    const int sample = blockIdx.x;
    const int node = threadIdx.x;
    constexpr int channels = transverse
        ? transverse_channels
        : ((contribution_I3 || both_contributions) ? longitudinal_channels : 2);
    __shared__ float partial[transverse_channels][threads];
    __shared__ float outer_factor;
    __shared__ GBWSampleCoordinates sample_coordinates;

    const float Q2 = Q2_input[0];
    const float xbj = xbj_input[0];
    const float z2_min = xbj / Q2;
    if (z2_min > 1.0F) {
        if (node == 0) {
            output[sample] = 0.0F;
            if constexpr (both_contributions) {
                output_I3[sample] = 0.0F;
            }
        }
        return;
    }

    if (node == 0) {
        const int64_t sample_offset = sample * sample_stride;
        const float z1 = (1.0F - z2_min) * unit_samples[sample_offset];
        const float z2 = (1.0F - z1 - z2_min) *
            unit_samples[sample_offset + dimension_stride] + z2_min;
        const float x01 = maxr * unit_samples[sample_offset + 2 * dimension_stride];
        const float x02 = maxr * unit_samples[sample_offset + 3 * dimension_stride];
        const float angle = 2.0F * pi *
            unit_samples[sample_offset + 4 * dimension_stride];
        const float x01_square = square(x01);
        const float x02_square = square(x02);
        // Preserve the eager float32 operation order near x21=0, where the I2 subtraction is
        // sensitive to a one-ulp change in the squared separation.
        const float cross = __fmul_rn(
            __fmul_rn(__fmul_rn(2.0F, x01), x02),
            cosf(angle)
        );
        const float x21_square = fmaxf(
            __fsub_rn(__fadd_rn(x01_square, x02_square), cross),
            0.0F
        );
        sample_coordinates = {
            z1,
            z2,
            x01,
            x02,
            x01_square,
            x02_square,
            x21_square,
        };
    }
    __syncthreads();
    const float z1 = sample_coordinates.z1;
    const float z2 = sample_coordinates.z2;
    const float x01 = sample_coordinates.x01;
    const float x02 = sample_coordinates.x02;
    const float x01_square = sample_coordinates.x01_square;
    const float x02_square = sample_coordinates.x02_square;
    const float x21_square = sample_coordinates.x21_square;
    const Geometry geometry = tripole_geometry<transverse>(
        Q2,
        z1,
        z2,
        x01_square,
        x02_square,
        x21_square
    );

    if constexpr (transverse && both_contributions && split_transverse_moments) {
        if (node < 2 * points) {
            const int quadrature_node = node < points ? node : node - points;
            const float y = nodes[quadrature_node];
            const float weight = weights[quadrature_node];
            const float mf_square = square(mf);
            if (node < points) {
                const GMomentsAtNode j = g_values(
                    geometry.Qbar_j_square,
                    mf_square,
                    geometry.x2_j_square,
                    geometry.x3_j_square,
                    geometry.omega_j,
                    geometry.lambda_j,
                    y
                );
                partial[0][quadrature_node] = weight * j.G22_bar;
                partial[2][quadrature_node] = weight * j.G12_bar;
                partial[4][quadrature_node] = weight * j.G21;
                partial[6][quadrature_node] = weight * j.G11;
            } else {
                const GMomentsAtNode k = g_values(
                    geometry.Qbar_k_square,
                    mf_square,
                    geometry.x2_k_square,
                    geometry.x3_k_square,
                    geometry.omega_k,
                    geometry.lambda_k,
                    y
                );
                partial[1][quadrature_node] = weight * k.G22_bar;
                partial[3][quadrature_node] = weight * k.G12_bar;
                partial[5][quadrature_node] = weight * k.G21;
                partial[7][quadrature_node] = weight * k.G11;
            }
        }
    } else {
        float values[channels] = {};
        if (node < points) {
            const float y = nodes[node];
            const float weight = weights[node];
            const float mf_square = square(mf);
            if constexpr (transverse) {
                const GMomentsAtNode j = g_values(
                    geometry.Qbar_j_square,
                    mf_square,
                    geometry.x2_j_square,
                    geometry.x3_j_square,
                    geometry.omega_j,
                    geometry.lambda_j,
                    y
                );
                const GMomentsAtNode k = g_values(
                    geometry.Qbar_k_square,
                    mf_square,
                    geometry.x2_k_square,
                    geometry.x3_k_square,
                    geometry.omega_k,
                    geometry.lambda_k,
                    y
                );
                values[0] = weight * j.G22_bar;
                values[1] = weight * k.G22_bar;
                values[2] = weight * j.G12_bar;
                values[3] = weight * k.G12_bar;
                values[4] = weight * j.G21;
                values[5] = weight * k.G21;
                values[6] = weight * j.G11;
                values[7] = weight * k.G11;
            } else if constexpr (contribution_I3 || both_contributions) {
                const LongitudinalMomentsAtNode j = longitudinal_g_values_paired(
                    geometry.Qbar_j_square,
                    mf_square,
                    geometry.x2_j_square,
                    geometry.x3_j_square,
                    geometry.omega_j,
                    geometry.lambda_j,
                    y
                );
                const LongitudinalMomentsAtNode k = longitudinal_g_values_paired(
                    geometry.Qbar_k_square,
                    mf_square,
                    geometry.x2_k_square,
                    geometry.x3_k_square,
                    geometry.omega_k,
                    geometry.lambda_k,
                    y
                );
                values[0] = weight * j.G12_bar;
                values[1] = weight * k.G12_bar;
                values[2] = weight * j.G11;
                values[3] = weight * k.G11;
            } else {
                values[0] = weight * longitudinal_g12_bar_value(
                    geometry.Qbar_j_square,
                    mf_square,
                    geometry.x2_j_square,
                    geometry.x3_j_square,
                    geometry.omega_j,
                    geometry.lambda_j,
                    y
                );
                values[1] = weight * longitudinal_g12_bar_value(
                    geometry.Qbar_k_square,
                    mf_square,
                    geometry.x2_k_square,
                    geometry.x3_k_square,
                    geometry.omega_k,
                    geometry.lambda_k,
                    y
                );
            }
        }
#pragma unroll
        for (int channel = 0; channel < channels; ++channel) {
            partial[channel][node] = values[channel];
        }
    }
    // Evaluate the outer factor in an otherwise idle thread while quadrature threads evaluate
    // Bessel functions. The two calculations meet only at the final multiplication.
    if (node == blockDim.x - 1) {
        const float rapidity = logf(Q2 / xbj * z2);
        const float initial_rapidity = logf(1.0F / x0);
        const float S01 = gbw_S_matrix(
            x01_square, rapidity, Qs0_square, lambda, gamma, initial_rapidity
        );
        const float S02 = gbw_S_matrix(
            x02_square, rapidity, Qs0_square, lambda, gamma, initial_rapidity
        );
        const float S12 = gbw_S_matrix(
            x21_square, rapidity, Qs0_square, lambda, gamma, initial_rapidity
        );
        const float tripole = finite_Nc
            ? 1.0F - 1.125F * (S02 * S12 - S01 / 9.0F)
            : 1.0F - S02 * S12;
        const float coupling_r_square = parent_coupling
            ? x01_square
            : fminf(x01_square, fminf(x02_square, x21_square));
        const float alpha = dis_running_coupling(
            coupling_r_square,
            coupling_C2,
            active_flavors,
            maximum_alpha,
            smooth_coupling
        );
        const float jacobian = (1.0F - z2_min) * (1.0F - z1 - z2_min) *
            x01 * x02 * square(maxr) * 2.0F * pi;
        const float factor = tripole * jacobian * (alpha * color_factor / pi) / z2;
        outer_factor = rapidity >= 0.0F && isfinite(factor) ? factor : 0.0F;
    }
    __syncthreads();

    if (node < channels) {
        float sum = 0.0F;
        for (int index = 0; index < points; ++index) {
            sum += partial[node][index];
        }
        partial[node][0] = sum;
    }
    __syncthreads();

    if constexpr (both_contributions) {
        const float mf_square = square(mf);
        if (node == 0) {
            float impact_I2;
            if constexpr (transverse) {
                float moments[transverse_channels];
#pragma unroll
                for (int channel = 0; channel < transverse_channels; ++channel) {
                    moments[channel] = partial[channel][0];
                }
                impact_I2 = transverse_I2(mf_square, z1, z2, geometry, moments);
            } else {
                impact_I2 = longitudinal_I2(
                    Q2,
                    mf_square,
                    z1,
                    z2,
                    geometry,
                    partial[0][0],
                    partial[1][0]
                );
            }
            const float result = outer_factor * impact_I2;
            output[sample] = isfinite(result) ? result : 0.0F;
        }
        if (node == 1) {
            float impact_I3;
            if constexpr (transverse) {
                float moments[transverse_channels];
#pragma unroll
                for (int channel = 0; channel < transverse_channels; ++channel) {
                    moments[channel] = partial[channel][0];
                }
                impact_I3 = transverse_I3(mf_square, z1, z2, geometry, moments);
            } else {
                impact_I3 = longitudinal_I3(
                    Q2,
                    mf_square,
                    z1,
                    z2,
                    geometry,
                    partial[0][0],
                    partial[1][0],
                    partial[2][0],
                    partial[3][0]
                );
            }
            const float result = outer_factor * impact_I3;
            output_I3[sample] = isfinite(result) ? result : 0.0F;
        }
    } else if (node == 0) {
        const float mf_square = square(mf);
        if constexpr (transverse) {
            float moments[transverse_channels];
#pragma unroll
            for (int channel = 0; channel < transverse_channels; ++channel) {
                moments[channel] = partial[channel][0];
            }
            float impact_factor;
            if constexpr (contribution_I3) {
                impact_factor = transverse_I3(mf_square, z1, z2, geometry, moments);
            } else {
                impact_factor = transverse_I2(mf_square, z1, z2, geometry, moments);
            }
            const float result = outer_factor * impact_factor;
            output[sample] = isfinite(result) ? result : 0.0F;
        } else if constexpr (contribution_I3) {
            const float impact_factor = longitudinal_I3(
                Q2,
                mf_square,
                z1,
                z2,
                geometry,
                partial[0][0],
                partial[1][0],
                partial[2][0],
                partial[3][0]
            );
            const float result = outer_factor * impact_factor;
            output[sample] = isfinite(result) ? result : 0.0F;
        } else {
            const float impact_factor = longitudinal_I2(
                Q2,
                mf_square,
                z1,
                z2,
                geometry,
                partial[0][0],
                partial[1][0]
            );
            const float result = outer_factor * impact_factor;
            output[sample] = isfinite(result) ? result : 0.0F;
        }
    }
}

void validate_nested_gbw_inputs(
    const torch::Tensor& Q2,
    const torch::Tensor& xbj,
    const torch::Tensor& unit_samples,
    const torch::Tensor& nodes,
    const torch::Tensor& weights,
    double maxr,
    double Qs0_square,
    double gamma,
    double x0,
    double coupling_C2,
    int active_flavors,
    double maximum_alpha
) {
    TORCH_CHECK(
        Q2.is_cuda() && Q2.scalar_type() == torch::kFloat32 && Q2.numel() == 1 &&
            xbj.is_cuda() && xbj.scalar_type() == torch::kFloat32 && xbj.numel() == 1 &&
            Q2.get_device() == xbj.get_device(),
        "nested DIS requires matching CUDA float32 scalar Q2 and xbj"
    );
    TORCH_CHECK(
        unit_samples.is_cuda() && unit_samples.scalar_type() == torch::kFloat32 &&
            unit_samples.dim() == 2 && unit_samples.size(1) == 5 &&
            unit_samples.get_device() == Q2.get_device() &&
            unit_samples.stride(0) > 0 && unit_samples.stride(1) > 0,
        "nested DIS unit samples must be CUDA float32 [samples, 5] with positive strides"
    );
    TORCH_CHECK(
        nodes.is_cuda() && weights.is_cuda() &&
            nodes.scalar_type() == torch::kFloat32 && weights.scalar_type() == torch::kFloat32 &&
            nodes.dim() == 1 && weights.dim() == 1 && nodes.numel() == weights.numel() &&
            nodes.is_contiguous() && weights.is_contiguous() &&
            nodes.get_device() == Q2.get_device() && weights.get_device() == Q2.get_device(),
        "nested DIS nodes and weights must be matching contiguous CUDA float32 vectors"
    );
    TORCH_CHECK(
        nodes.numel() >= 8 && nodes.numel() <= threads,
        "nested DIS supports between 8 and 128 inner points"
    );
    TORCH_CHECK(
        maxr > 0.0 && Qs0_square > 0.0 && gamma > 0.0 && x0 > 0.0 &&
            coupling_C2 > 0.0 && active_flavors >= 0 && maximum_alpha > 0.0,
        "nested DIS scalar parameters must be physical"
    );
}

torch::Tensor dis_nested_gbw_integrand(
    torch::Tensor Q2,
    torch::Tensor xbj,
    double mf,
    torch::Tensor unit_samples,
    torch::Tensor nodes,
    torch::Tensor weights,
    bool transverse,
    bool contribution_I3,
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
    validate_nested_gbw_inputs(
        Q2,
        xbj,
        unit_samples,
        nodes,
        weights,
        maxr,
        Qs0_square,
        gamma,
        x0,
        coupling_C2,
        active_flavors,
        maximum_alpha
    );

    const int64_t samples = unit_samples.size(0);
    auto output = torch::empty({samples}, unit_samples.options());
    if (samples == 0) {
        return output;
    }
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int block_threads = transverse && nodes.numel() <= 64 ? 64 : threads;
#define LAUNCH_NESTED_GBW(TRANSVERSE, I3) \
    dis_nested_gbw_integrand_kernel<TRANSVERSE, I3><<<samples, block_threads, 0, stream>>>( \
        Q2.data_ptr<float>(), \
        xbj.data_ptr<float>(), \
        static_cast<float>(mf), \
        unit_samples.data_ptr<float>(), \
        unit_samples.stride(0), \
        unit_samples.stride(1), \
        nodes.data_ptr<float>(), \
        weights.data_ptr<float>(), \
        output.data_ptr<float>(), \
        nullptr, \
        nodes.numel(), \
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
        static_cast<float>(maximum_alpha) \
    )
    if (transverse && contribution_I3) {
        LAUNCH_NESTED_GBW(true, true);
    } else if (transverse) {
        LAUNCH_NESTED_GBW(true, false);
    } else if (contribution_I3) {
        LAUNCH_NESTED_GBW(false, true);
    } else {
        LAUNCH_NESTED_GBW(false, false);
    }
#undef LAUNCH_NESTED_GBW
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<torch::Tensor> dis_nested_gbw_i2_i3_integrand(
    torch::Tensor Q2,
    torch::Tensor xbj,
    double mf,
    torch::Tensor unit_samples,
    torch::Tensor nodes,
    torch::Tensor weights,
    bool transverse,
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
    validate_nested_gbw_inputs(
        Q2,
        xbj,
        unit_samples,
        nodes,
        weights,
        maxr,
        Qs0_square,
        gamma,
        x0,
        coupling_C2,
        active_flavors,
        maximum_alpha
    );

    const int64_t samples = unit_samples.size(0);
    auto output_I2 = torch::empty({samples}, unit_samples.options());
    auto output_I3 = torch::empty({samples}, unit_samples.options());
    if (samples == 0) {
        return {output_I2, output_I3};
    }
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const bool split_moments = transverse && nodes.numel() <= 64;
    const int block_threads = split_moments ? 2 * nodes.numel() : threads;
#define LAUNCH_NESTED_GBW_PAIR(TRANSVERSE) \
    dis_nested_gbw_integrand_kernel<TRANSVERSE, false, true> \
        <<<samples, block_threads, 0, stream>>>( \
            Q2.data_ptr<float>(), \
            xbj.data_ptr<float>(), \
            static_cast<float>(mf), \
            unit_samples.data_ptr<float>(), \
            unit_samples.stride(0), \
            unit_samples.stride(1), \
            nodes.data_ptr<float>(), \
            weights.data_ptr<float>(), \
            output_I2.data_ptr<float>(), \
            output_I3.data_ptr<float>(), \
            nodes.numel(), \
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
            static_cast<float>(maximum_alpha) \
        )
    if (split_moments) {
        dis_nested_gbw_integrand_kernel<true, false, true, true> \
            <<<samples, block_threads, 0, stream>>>( \
                Q2.data_ptr<float>(), \
                xbj.data_ptr<float>(), \
                static_cast<float>(mf), \
                unit_samples.data_ptr<float>(), \
                unit_samples.stride(0), \
                unit_samples.stride(1), \
                nodes.data_ptr<float>(), \
                weights.data_ptr<float>(), \
                output_I2.data_ptr<float>(), \
                output_I3.data_ptr<float>(), \
                nodes.numel(), \
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
                static_cast<float>(maximum_alpha) \
            );
    } else if (transverse) {
        LAUNCH_NESTED_GBW_PAIR(true);
    } else {
        LAUNCH_NESTED_GBW_PAIR(false);
    }
#undef LAUNCH_NESTED_GBW_PAIR
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output_I2, output_I3};
}

std::vector<torch::Tensor> dis_nested_i2_i3(
    torch::Tensor Q2,
    double mf,
    torch::Tensor z1,
    torch::Tensor z2,
    torch::Tensor x01_square,
    torch::Tensor x02_square,
    torch::Tensor x21_square,
    torch::Tensor nodes,
    torch::Tensor weights,
    bool transverse
) {
    TORCH_CHECK(
        Q2.is_cuda() && Q2.scalar_type() == torch::kFloat32 && Q2.numel() == 1,
        "nested DIS requires a CUDA float32 scalar Q2"
    );
    const int64_t samples = z1.numel();
    for (const auto& tensor : {z1, z2, x01_square, x02_square, x21_square}) {
        TORCH_CHECK(
            tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32 &&
                tensor.dim() == 1 && tensor.numel() == samples && tensor.is_contiguous(),
            "nested DIS geometry inputs must be contiguous CUDA float32 vectors of equal length"
        );
    }
    TORCH_CHECK(
        nodes.is_cuda() && weights.is_cuda() &&
            nodes.scalar_type() == torch::kFloat32 && weights.scalar_type() == torch::kFloat32 &&
            nodes.dim() == 1 && weights.dim() == 1 && nodes.numel() == weights.numel() &&
            nodes.is_contiguous() && weights.is_contiguous(),
        "nested DIS nodes and weights must be contiguous CUDA float32 vectors"
    );
    TORCH_CHECK(
        nodes.numel() >= 8 && nodes.numel() <= threads,
        "nested DIS supports between 8 and 128 inner points"
    );
    auto output_I2 = torch::empty_like(z1);
    auto output_I3 = torch::empty_like(z1);
    if (samples == 0) {
        return {output_I2, output_I3};
    }
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int block_threads = transverse && nodes.numel() <= 64 ? 64 : threads;
    if (transverse) {
        dis_nested_kernel<true><<<samples, block_threads, 0, stream>>>(
            Q2.data_ptr<float>(),
            static_cast<float>(mf),
            z1.data_ptr<float>(),
            z2.data_ptr<float>(),
            x01_square.data_ptr<float>(),
            x02_square.data_ptr<float>(),
            x21_square.data_ptr<float>(),
            nodes.data_ptr<float>(),
            weights.data_ptr<float>(),
            output_I2.data_ptr<float>(),
            output_I3.data_ptr<float>(),
            nodes.numel()
        );
    } else {
        dis_nested_kernel<false><<<samples, block_threads, 0, stream>>>(
            Q2.data_ptr<float>(),
            static_cast<float>(mf),
            z1.data_ptr<float>(),
            z2.data_ptr<float>(),
            x01_square.data_ptr<float>(),
            x02_square.data_ptr<float>(),
            x21_square.data_ptr<float>(),
            nodes.data_ptr<float>(),
            weights.data_ptr<float>(),
            output_I2.data_ptr<float>(),
            output_I3.data_ptr<float>(),
            nodes.numel()
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output_I2, output_I3};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "dis_nested_i2_i3",
        &dis_nested_i2_i3,
        "Fixed-inner-quadrature factorized DIS I2 and I3"
    );
    module.def(
        "dis_nested_gbw_integrand",
        &dis_nested_gbw_integrand,
        "Fused GBW DIS outer integrand and fixed inner quadrature"
    );
    module.def(
        "dis_nested_gbw_i2_i3_integrand",
        &dis_nested_gbw_i2_i3_integrand,
        "Fused GBW DIS outer integrand returning I2 and I3"
    );
}
