#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <vector>

namespace {

__device__ float chebyshev(float x, const float* coefficients, int length) {
    float b0 = coefficients[0];
    float b1 = 0.0f;
    float b2 = 0.0f;
    for (int index = 1; index < length; ++index) {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + coefficients[index];
    }
    return 0.5f * (b0 - b2);
}

__device__ float pytorch_bessel_i1(float input) {
    constexpr float A[] = {
        9.38153738649577178388E-9f,
        -4.44505912879632808065E-8f,
        2.00329475355213526229E-7f,
        -8.56872026469545474066E-7f,
        3.47025130813767847674E-6f,
        -1.32731636560394358279E-5f,
        4.78156510755005422638E-5f,
        -1.61760815825896745588E-4f,
        5.12285956168575772895E-4f,
        -1.51357245063125314899E-3f,
        4.15642294431288815669E-3f,
        -1.05640848946261981558E-2f,
        2.47264490306265168283E-2f,
        -5.29459812080949914269E-2f,
        1.02643658689847095384E-1f,
        -1.76416518357834055153E-1f,
        2.52587186443633654823E-1f,
    };
    constexpr float B[] = {
        -3.83538038596423702205E-9f,
        -2.63146884688951950684E-8f,
        -2.51223623787020892529E-7f,
        -3.88256480887769039346E-6f,
        -1.10588938762623716291E-4f,
        -9.76109749136146840777E-3f,
        7.78576235018280120474E-1f,
    };
    const float x = fabsf(input);
    float output;
    if (x <= 8.0f) {
        output = expf(x) * x * chebyshev(x / 2.0f - 2.0f, A, 17);
    } else {
        output = expf(x) * chebyshev(32.0f / x - 2.0f, B, 7) / sqrtf(x);
    }
    return input < 0.0f ? -output : output;
}

__device__ float pytorch_bessel_j1(float x) {
    // This is the Cephes approximation used by PyTorch's bessel_j1 CUDA kernel.
    constexpr float PP[] = {
        7.62125616208173112003e-04f,
        7.31397056940917570436e-02f,
        1.12719608129684925192e+00f,
        5.11207951146807644818e+00f,
        8.42404590141772420927e+00f,
        5.21451598682361504063e+00f,
        1.00000000000000000254e+00f,
    };
    constexpr float PQ[] = {
        5.71323128072548699714e-04f,
        6.88455908754495404082e-02f,
        1.10514232634061696926e+00f,
        5.07386386128601488557e+00f,
        8.39985554327604159757e+00f,
        5.20982848682361821619e+00f,
        9.99999999999999997461e-01f,
    };
    constexpr float QP[] = {
        5.10862594750176621635e-02f,
        4.98213872951233449420e+00f,
        7.58238284132545283818e+01f,
        3.66779609360150777800e+02f,
        7.10856304998926107277e+02f,
        5.97489612400613639965e+02f,
        2.11688757100572135698e+02f,
        2.52070205858023719784e+01f,
    };
    constexpr float QQ[] = {
        7.42373277035675149943e+01f,
        1.05644886038262816351e+03f,
        4.98641058337653607651e+03f,
        9.56231892404756170795e+03f,
        7.99704160447350683650e+03f,
        2.82619278517639096600e+03f,
        3.36093607810698293419e+02f,
    };
    constexpr float RP[] = {
        -8.99971225705559398224e+08f,
        4.52228297998194034323e+11f,
        -7.27494245221818276015e+13f,
        3.68295732863852883286e+15f,
    };
    constexpr float RQ[] = {
        6.20836478118054335476e+02f,
        2.56987256757748830383e+05f,
        8.35146791431949253037e+07f,
        2.21511595479792499675e+10f,
        4.74914122079991414898e+12f,
        7.84369607876235854894e+14f,
        8.95222336184627338078e+16f,
        5.32278620332680085395e+18f,
    };

    if (x < 0.0f) {
        return -pytorch_bessel_j1(-x);
    }
    if (x <= 5.0f) {
        float rp = 0.0f;
        for (const float coefficient : RP) {
            rp = rp * (x * x) + coefficient;
        }
        float rq = 0.0f;
        for (const float coefficient : RQ) {
            rq = rq * (x * x) + coefficient;
        }
        return rp / rq * x * (x * x - 1.46819706421238932572e+01f) *
            (x * x - 4.92184563216946036703e+01f);
    }

    const float scaled = 5.0f / x * (5.0f / x);
    float pp = 0.0f;
    for (const float coefficient : PP) {
        pp = pp * scaled + coefficient;
    }
    float pq = 0.0f;
    for (const float coefficient : PQ) {
        pq = pq * scaled + coefficient;
    }
    float qp = 0.0f;
    for (const float coefficient : QP) {
        qp = qp * scaled + coefficient;
    }
    float qq = 0.0f;
    for (const float coefficient : QQ) {
        qq = qq * scaled + coefficient;
    }
    constexpr float phase = 2.356194490192344928846982537459627163f;
    constexpr float normalization = 0.797884560802865355879892119868763737f;
    return (
               pp / pq * cosf(x - phase) -
               5.0f / x * (qp / qq) * sinf(x - phase)
           ) *
        normalization / sqrtf(x);
}

__device__ double chebyshev64(double x, const double* coefficients, int length) {
    double b0 = coefficients[0];
    double b1 = 0.0;
    double b2 = 0.0;
    for (int index = 1; index < length; ++index) {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + coefficients[index];
    }
    return 0.5 * (b0 - b2);
}

__device__ double pytorch_bessel_i1(double input) {
    constexpr double A[] = {
        9.38153738649577178388E-9,
        -4.44505912879632808065E-8,
        2.00329475355213526229E-7,
        -8.56872026469545474066E-7,
        3.47025130813767847674E-6,
        -1.32731636560394358279E-5,
        4.78156510755005422638E-5,
        -1.61760815825896745588E-4,
        5.12285956168575772895E-4,
        -1.51357245063125314899E-3,
        4.15642294431288815669E-3,
        -1.05640848946261981558E-2,
        2.47264490306265168283E-2,
        -5.29459812080949914269E-2,
        1.02643658689847095384E-1,
        -1.76416518357834055153E-1,
        2.52587186443633654823E-1,
    };
    constexpr double B[] = {
        -3.83538038596423702205E-9,
        -2.63146884688951950684E-8,
        -2.51223623787020892529E-7,
        -3.88256480887769039346E-6,
        -1.10588938762623716291E-4,
        -9.76109749136146840777E-3,
        7.78576235018280120474E-1,
    };
    const double x = fabs(input);
    const double output = x <= 8.0
        ? exp(x) * x * chebyshev64(x / 2.0 - 2.0, A, 17)
        : exp(x) * chebyshev64(32.0 / x - 2.0, B, 7) / sqrt(x);
    return input < 0.0 ? -output : output;
}

__device__ double pytorch_bessel_j1(double x) {
    constexpr double PP[] = {
        7.62125616208173112003e-04,
        7.31397056940917570436e-02,
        1.12719608129684925192e+00,
        5.11207951146807644818e+00,
        8.42404590141772420927e+00,
        5.21451598682361504063e+00,
        1.00000000000000000254e+00,
    };
    constexpr double PQ[] = {
        5.71323128072548699714e-04,
        6.88455908754495404082e-02,
        1.10514232634061696926e+00,
        5.07386386128601458580e+00,
        8.39985554327604159757e+00,
        5.20982848682361821619e+00,
        9.99999999999999997461e-01,
    };
    constexpr double QP[] = {
        5.10862594750176621635e-02,
        4.98213872951233449420e+00,
        7.58238284132545283818e+01,
        3.66779609360150777800e+02,
        7.10856304998926107277e+02,
        5.97489612400613639965e+02,
        2.11688757183606607801e+02,
        2.52070205858023719784e+01,
    };
    constexpr double QQ[] = {
        7.42373277035675149943e+01,
        1.05644886038262816351e+03,
        4.98641058337653607651e+03,
        9.56231892404756170795e+03,
        7.99704160447350683650e+03,
        2.82619278517639096600e+03,
        3.36093607810698293419e+02,
    };
    constexpr double RP[] = {
        -8.99971225705559398224e+08,
        4.52228297998194034323e+11,
        -7.27494245221818276015e+13,
        3.68295732863852883286e+15,
    };
    constexpr double RQ[] = {
        6.20836478118054335476e+02,
        2.56987256757748830383e+05,
        8.35146791431949253037e+07,
        2.21511595479792499675e+10,
        4.74914122079991414898e+12,
        7.84369607876235854894e+14,
        8.95222336184627338078e+16,
        5.32278620332680085395e+18,
    };

    if (x < 0.0) {
        return -pytorch_bessel_j1(-x);
    }
    if (x <= 5.0) {
        double rp = 0.0;
        for (const double coefficient : RP) {
            rp = rp * (x * x) + coefficient;
        }
        double rq = 0.0;
        for (const double coefficient : RQ) {
            rq = rq * (x * x) + coefficient;
        }
        return rp / rq * x * (x * x - 1.46819706421238932572e+01) *
            (x * x - 4.92184563216946036703e+01);
    }

    const double scaled = 25.0 / (x * x);
    double pp = 0.0;
    for (const double coefficient : PP) {
        pp = pp * scaled + coefficient;
    }
    double pq = 0.0;
    for (const double coefficient : PQ) {
        pq = pq * scaled + coefficient;
    }
    double qp = 0.0;
    for (const double coefficient : QP) {
        qp = qp * scaled + coefficient;
    }
    double qq = 0.0;
    for (const double coefficient : QQ) {
        qq = qq * scaled + coefficient;
    }
    constexpr double phase = 2.356194490192344928846982537459627163;
    constexpr double normalization = 0.797884560802865355879892119868763737;
    return (pp / pq * cos(x - phase) - 5.0 / x * (qp / qq) * sin(x - phase)) *
        normalization / sqrt(x);
}

__device__ __forceinline__ float add_rn(float left, float right) {
    return __fadd_rn(left, right);
}

__device__ __forceinline__ float subtract_rn(float left, float right) {
    return __fsub_rn(left, right);
}

__device__ __forceinline__ float multiply_rn(float left, float right) {
    return __fmul_rn(left, right);
}

__device__ __forceinline__ float divide_rn(float left, float right) {
    return __fdiv_rn(left, right);
}

__device__ float logaddexp(float left, float right) {
    return add_rn(fmaxf(left, right), log1pf(expf(-fabsf(subtract_rn(left, right)))));
}

__device__ float smooth_alpha_s(
    float distance,
    float b0,
    float lambda_squared,
    float scale_numerator,
    float log_mu0_term
) {
    const float distance_squared = multiply_rn(distance, distance);
    const float denominator = multiply_rn(distance_squared, lambda_squared);
    const float scale = multiply_rn(__frcp_rn(denominator), scale_numerator);
    const float log_scale_term = multiply_rn(logf(scale), 5.0f);
    const float log_argument = multiply_rn(logaddexp(log_mu0_term, log_scale_term), 0.2f);
    return __frcp_rn(multiply_rn(b0, log_argument));
}

__device__ float n_spline(
    float value,
    const float* r_grid,
    const float* log_grid,
    const float* a,
    const float* b,
    const float* c,
    const float* d,
    int grid_points,
    int search_steps
) {
    const float minimum_r = r_grid[0];
    const float maximum_r = r_grid[grid_points - 1];
    const float evaluation_r = fminf(fmaxf(value, minimum_r), maximum_r);
    const float log_r = logf(evaluation_r);
    const int last_interval = grid_points - 2;
    const float grid_position = divide_rn(
        multiply_rn(
            subtract_rn(log_r, log_grid[0]),
            static_cast<float>(grid_points - 1)
        ),
        subtract_rn(log_grid[grid_points - 1], log_grid[0])
    );
    int interval = min(max(__float2int_rd(grid_position), 0), last_interval);
    const bool estimate_missed = log_r < log_grid[interval]
        || (interval < last_interval && log_r >= log_grid[interval + 1]);
    if (estimate_missed) {
        int lower = 0;
        int upper = grid_points;
        for (int step = 0; step < search_steps; ++step) {
            const int middle = (lower + upper) / 2;
            if (log_r >= log_grid[middle]) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        interval = min(max(lower - 1, 0), last_interval);
    }
    const float dx = subtract_rn(log_r, log_grid[interval]);
    const float polynomial = add_rn(
        a[interval],
        multiply_rn(
            dx,
            add_rn(
                b[interval],
                multiply_rn(dx, add_rn(c[interval], multiply_rn(dx, d[interval])))
            )
        )
    );
    float result = expf(polynomial);
    result = isfinite(result) ? result : 0.0f;
    result = value < minimum_r ? 0.0f : result;
    return value > maximum_r ? 1.0f : result;
}

struct K1Parameters {
    const float* r_grid;
    const float* log_grid;
    const float* a;
    const float* b;
    const float* c;
    const float* d;
    int grid_points;
    int search_steps;
    float nc;
    float nf;
    float minimum_r;
    float ksub;
    float coupling_b0;
    float coupling_lambda_squared;
    float coupling_scale_numerator;
    float coupling_log_mu0_term;
};

struct K1Parameters64 {
    const double* r_grid;
    const double* log_grid;
    const double* a;
    const double* b;
    const double* c;
    const double* d;
    int grid_points;
    double nc;
    double nf;
    double minimum_r;
    double ksub;
    double coupling_b0;
    double coupling_lambda_squared;
    double coupling_scale_numerator;
    double coupling_log_mu0_term;
};

__device__ double n_spline64(double value, const K1Parameters64& parameters) {
    const double minimum_r = parameters.r_grid[0];
    const double maximum_r = parameters.r_grid[parameters.grid_points - 1];
    const double evaluation_r = fmin(fmax(value, minimum_r), maximum_r);
    const double log_r = log(evaluation_r);
    int lower = 0;
    int upper = parameters.grid_points;
    while (lower < upper) {
        const int middle = (lower + upper) / 2;
        if (log_r >= parameters.log_grid[middle]) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    const int interval = max(0, min(parameters.grid_points - 2, lower - 1));
    const double dx = log_r - parameters.log_grid[interval];
    const double polynomial = parameters.a[interval] +
        dx * (parameters.b[interval] +
              dx * (parameters.c[interval] + dx * parameters.d[interval]));
    double result = exp(polynomial);
    result = isfinite(result) ? result : 0.0;
    result = value < minimum_r ? 0.0 : result;
    return value > maximum_r ? 1.0 : result;
}

__device__ double smooth_alpha_s64(double distance, const K1Parameters64& parameters) {
    const double scale = parameters.coupling_scale_numerator /
        (distance * distance * parameters.coupling_lambda_squared);
    const double left = parameters.coupling_log_mu0_term;
    const double right = 5.0 * log(scale);
    const double maximum = fmax(left, right);
    const double log_argument = 0.2 * (maximum + log1p(exp(-fabs(left - right))));
    return 1.0 / (parameters.coupling_b0 * log_argument);
}

__device__ double k1_integrand64(
    double r,
    double z,
    double theta,
    const K1Parameters64& parameters
) {
    const double sin_half = sin(theta / 2.0);
    const double x_squared =
        (r - z) * (r - z) + 4.0 * r * z * sin_half * sin_half;
    const bool invalid = x_squared < 1e-40 || z < 1e-20 || r < 1e-20;
    const double x_distance = sqrt(fmax(x_squared, 0.0));
    const double safe_x = invalid ? 1.0 : x_distance;
    const double safe_y = invalid ? 1.0 : z;
    const double n_x = fmin(n_spline64(safe_x, parameters), 1.0);
    const double n_y = fmin(n_spline64(safe_y, parameters), 1.0);
    const double n_r = fmin(n_spline64(r, parameters), 1.0);
    const double dipole = n_x + n_y - n_r - n_x * n_y;

    const double alpha_r = smooth_alpha_s64(r, parameters);
    const double alpha_x = smooth_alpha_s64(safe_x, parameters);
    const double alpha_y = smooth_alpha_s64(safe_y, parameters);
    const double alpha_xy_smallest = safe_x <= safe_y ? alpha_x : alpha_y;
    const double alpha_smallest = fmin(safe_x, safe_y) <= r
        ? alpha_xy_smallest
        : alpha_r;
    const double first = r * r / (safe_x * safe_x * safe_y * safe_y);
    const double second = (alpha_y / alpha_x - 1.0) / (safe_y * safe_y);
    const double third = (alpha_x / alpha_y - 1.0) / (safe_x * safe_x);
    double lo_kernel = parameters.nc / (2.0 * M_PI * M_PI) * alpha_r *
        (first + second + third);
    lo_kernel = isfinite(lo_kernel) && !invalid ? lo_kernel : 0.0;

    const double double_log_argument =
        4.0 * log(safe_x / r) * log(safe_y / r);
    const double as_x = sqrt(
        alpha_smallest * parameters.nc / M_PI * fabs(double_log_argument)
    );
    const double bessel = double_log_argument >= 0.0
        ? pytorch_bessel_j1(2.0 * as_x)
        : pytorch_bessel_i1(2.0 * as_x);
    double doublelog = as_x == 0.0 ? 1.0 : bessel / as_x;
    doublelog = r > 1.01 * parameters.minimum_r ? doublelog : 1.0;
    const bool valid_resummation = isfinite(doublelog);

    const double min_xy = fmin(safe_x, safe_y);
    const double alphabar = alpha_smallest * parameters.nc / M_PI;
    const double ratio = r / min_xy;
    const double singlelog_exponent =
        -alphabar * (11.0 / 12.0) * fabs(log(parameters.ksub * ratio * ratio));
    const double singlelog = exp(singlelog_exponent);
    const double expansion_log = log(sqrt(parameters.ksub) * r / min_xy);
    const double singlelog_expansion =
        -alphabar * (11.0 / 12.0) * fabs(2.0 * expansion_log);
    const double ratio_xy = r / (safe_x * safe_y);
    const double lo_kernel_single_as =
        alpha_r * parameters.nc / (2.0 * M_PI * M_PI) * ratio_xy * ratio_xy;
    const double subtract = lo_kernel_single_as * singlelog_expansion;
    const double finite_constant =
        67.0 / 9.0 - M_PI * M_PI / 3.0 - 10.0 / 9.0 * parameters.nf / parameters.nc;
    const double k1fin =
        lo_kernel * alpha_r * parameters.nc / (4.0 * M_PI) * finite_constant;
    const double result = (doublelog * singlelog * lo_kernel - subtract + k1fin) * dipole;
    return invalid || !valid_resummation ? 0.0 : result;
}

struct K1SliceConstants {
    float r;
    float z;
    float r_squared;
    float z_squared;
    float n_r;
    float n_z;
    float alpha_r;
    float alpha_z;
    float log_z_over_r;
};

__device__ K1SliceConstants make_slice_constants(
    float r,
    float z,
    const K1Parameters& parameters
) {
    return {
        r,
        z,
        multiply_rn(r, r),
        multiply_rn(z, z),
        fminf(
            n_spline(
                r,
                parameters.r_grid,
                parameters.log_grid,
                parameters.a,
                parameters.b,
                parameters.c,
                parameters.d,
                parameters.grid_points,
                parameters.search_steps
            ),
            1.0f
        ),
        fminf(
            n_spline(
                z,
                parameters.r_grid,
                parameters.log_grid,
                parameters.a,
                parameters.b,
                parameters.c,
                parameters.d,
                parameters.grid_points,
                parameters.search_steps
            ),
            1.0f
        ),
        smooth_alpha_s(
            r,
            parameters.coupling_b0,
            parameters.coupling_lambda_squared,
            parameters.coupling_scale_numerator,
            parameters.coupling_log_mu0_term
        ),
        smooth_alpha_s(
            z,
            parameters.coupling_b0,
            parameters.coupling_lambda_squared,
            parameters.coupling_scale_numerator,
            parameters.coupling_log_mu0_term
        ),
        logf(divide_rn(z, r)),
    };
}

__device__ K1SliceConstants make_fixed_slice_constants(
    float r,
    float z,
    float r_squared,
    float n_r,
    float alpha_r,
    const K1Parameters& parameters
) {
    return {
        r,
        z,
        r_squared,
        multiply_rn(z, z),
        n_r,
        fminf(
            n_spline(
                z,
                parameters.r_grid,
                parameters.log_grid,
                parameters.a,
                parameters.b,
                parameters.c,
                parameters.d,
                parameters.grid_points,
                parameters.search_steps
            ),
            1.0f
        ),
        alpha_r,
        smooth_alpha_s(
            z,
            parameters.coupling_b0,
            parameters.coupling_lambda_squared,
            parameters.coupling_scale_numerator,
            parameters.coupling_log_mu0_term
        ),
        logf(divide_rn(z, r)),
    };
}

template <bool StableGeometry, bool ReuseSingleLogExpansion = false>
__device__ float k1_integrand(
    const K1SliceConstants& slice,
    float cos_theta,
    float sin_half_squared,
    const K1Parameters& parameters
) {
    const float r = slice.r;
    const float z = slice.z;
    const float x_squared = StableGeometry
        ? add_rn(
              multiply_rn(subtract_rn(r, z), subtract_rn(r, z)),
              multiply_rn(multiply_rn(multiply_rn(4.0f, r), z), sin_half_squared)
          )
        : subtract_rn(
              add_rn(slice.r_squared, slice.z_squared),
              multiply_rn(multiply_rn(multiply_rn(2.0f, r), z), cos_theta)
          );
    const bool invalid_outer = x_squared < 1e-40f || z < 1e-20f || r < 1e-20f;
    const float x_distance = sqrtf(fmaxf(x_squared, 0.0f));
    const bool invalid_kernel = x_squared <= 0.0f || x_distance < 1e-20f || z < 1e-20f;
    const bool invalid = invalid_outer || invalid_kernel;
    const float safe_x = invalid ? 1.0f : x_distance;
    const float safe_y = invalid ? 1.0f : z;
    const float n_x = fminf(
        n_spline(
            safe_x,
            parameters.r_grid,
            parameters.log_grid,
            parameters.a,
            parameters.b,
            parameters.c,
            parameters.d,
            parameters.grid_points,
            parameters.search_steps
        ),
        1.0f
    );
    const float n_y = invalid
        ? fminf(
              n_spline(
                  safe_y,
                  parameters.r_grid,
                  parameters.log_grid,
                  parameters.a,
                  parameters.b,
                  parameters.c,
                  parameters.d,
                  parameters.grid_points,
                  parameters.search_steps
              ),
              1.0f
          )
        : slice.n_z;
    const float dipole = subtract_rn(
        subtract_rn(add_rn(n_x, n_y), slice.n_r),
        multiply_rn(n_x, n_y)
    );

    const float alpha_x = smooth_alpha_s(
        safe_x,
        parameters.coupling_b0,
        parameters.coupling_lambda_squared,
        parameters.coupling_scale_numerator,
        parameters.coupling_log_mu0_term
    );
    const float alpha_y = invalid
        ? smooth_alpha_s(
              safe_y,
              parameters.coupling_b0,
              parameters.coupling_lambda_squared,
              parameters.coupling_scale_numerator,
              parameters.coupling_log_mu0_term
          )
        : slice.alpha_z;
    const float alpha_xy_smallest = safe_x <= safe_y ? alpha_x : alpha_y;
    const float alpha_smallest = fminf(safe_x, safe_y) <= r
        ? alpha_xy_smallest
        : slice.alpha_r;

    const float safe_x_squared = multiply_rn(safe_x, safe_x);
    const float safe_y_squared = multiply_rn(safe_y, safe_y);
    const float first = divide_rn(
        slice.r_squared,
        multiply_rn(safe_x_squared, safe_y_squared)
    );
    const float second = divide_rn(
        subtract_rn(divide_rn(alpha_y, alpha_x), 1.0f),
        safe_y_squared
    );
    const float third = divide_rn(
        subtract_rn(divide_rn(alpha_x, alpha_y), 1.0f),
        safe_x_squared
    );
    float lo_kernel = multiply_rn(
        multiply_rn(
            parameters.nc / (2.0f * static_cast<float>(M_PI) * static_cast<float>(M_PI)),
            slice.alpha_r
        ),
        add_rn(add_rn(first, second), third)
    );
    lo_kernel = isfinite(lo_kernel) && !invalid ? lo_kernel : 0.0f;

    const float log_x = logf(divide_rn(safe_x, r));
    const float log_y = invalid
        ? logf(divide_rn(safe_y, r))
        : slice.log_z_over_r;
    const float double_log_argument = multiply_rn(
        multiply_rn(multiply_rn(4.0f, log_x), log_y),
        1.0f
    );
    const float as_x = sqrtf(
        multiply_rn(
            multiply_rn(alpha_smallest, parameters.nc / static_cast<float>(M_PI)),
            fabsf(double_log_argument)
        )
    );
    const float bessel_argument = multiply_rn(2.0f, as_x);
    const float safe_as_x = as_x == 0.0f ? 1.0f : as_x;
    const float bessel = double_log_argument >= 0.0f
        ? pytorch_bessel_j1(bessel_argument)
        : pytorch_bessel_i1(bessel_argument);
    float doublelog = divide_rn(bessel, safe_as_x);
    doublelog = as_x == 0.0f ? 1.0f : doublelog;
    doublelog = r > 1.01f * parameters.minimum_r ? doublelog : 1.0f;
    const bool valid_resummation = isfinite(doublelog);

    const float min_xy = fminf(safe_x, safe_y);
    const float alphabar = multiply_rn(alpha_smallest, parameters.nc / static_cast<float>(M_PI));
    const float ratio = divide_rn(r, min_xy);
    const float singlelog_exponent = -multiply_rn(
        multiply_rn(alphabar, 11.0f / 12.0f),
        fabsf(logf(multiply_rn(parameters.ksub, multiply_rn(ratio, ratio))))
    );
    const float singlelog = expf(singlelog_exponent);
    float singlelog_expansion;
    if constexpr (ReuseSingleLogExpansion) {
        singlelog_expansion = singlelog_exponent;
    } else {
        const float expansion_log = logf(
            divide_rn(multiply_rn(sqrtf(parameters.ksub), r), min_xy)
        );
        singlelog_expansion = -multiply_rn(
            multiply_rn(alphabar, 11.0f / 12.0f),
            fabsf(multiply_rn(2.0f, expansion_log))
        );
    }
    const float ratio_xy = divide_rn(r, multiply_rn(safe_x, safe_y));
    const float lo_kernel_single_as = multiply_rn(
        divide_rn(
            multiply_rn(slice.alpha_r, parameters.nc),
            2.0f * static_cast<float>(M_PI) * static_cast<float>(M_PI)
        ),
        multiply_rn(ratio_xy, ratio_xy)
    );
    const float subtract = multiply_rn(lo_kernel_single_as, singlelog_expansion);
    const float finite_constant =
        67.0f / 9.0f - static_cast<float>(M_PI) * static_cast<float>(M_PI) / 3.0f -
        10.0f / 9.0f * parameters.nf / parameters.nc;
    const float k1fin = multiply_rn(
        divide_rn(
            multiply_rn(multiply_rn(lo_kernel, slice.alpha_r), parameters.nc),
            4.0f * static_cast<float>(M_PI)
        ),
        finite_constant
    );

    float result = multiply_rn(doublelog, singlelog);
    result = multiply_rn(result, lo_kernel);
    result = subtract_rn(result, subtract);
    result = add_rn(result, k1fin);
    result = multiply_rn(result, dipole);
    return invalid || !valid_resummation ? 0.0f : result;
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = add_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

struct IntervalResult {
    float estimate;
    float error;
};

__device__ IntervalResult quadrature_from_nodes(
    float lower,
    float upper,
    float* node_values
) {
    constexpr float wgk[] = {
        0.011694638867371874f,
        0.03255816230796473f,
        0.054755896574351996f,
        0.07503967481091995f,
        0.0931254545836976f,
        0.10938715880229764f,
        0.12349197626206585f,
        0.13470921731147333f,
        0.14277593857706008f,
        0.14773910490133849f,
    };
    constexpr float gauss_weight[] = {
        0.06667134430868814f,
        0.1494513491505806f,
        0.21908636251598204f,
        0.26926671930999635f,
        0.29552422471475287f,
    };
    constexpr float center_weight = 0.1494455540029169f;
    const int lane = threadIdx.x % 32;
    const float half_width = divide_rn(subtract_rn(upper, lower), 2.0f);

    float pair = 0.0f;
    float weighted_pair = 0.0f;
    if (lane < 10) {
        pair = add_rn(node_values[lane], node_values[lane + 11]);
        weighted_pair = multiply_rn(pair, wgk[lane]);
    }
    float kronrod_unscaled = warp_sum(weighted_pair);
    if (lane == 0) {
        kronrod_unscaled = add_rn(
            kronrod_unscaled,
            multiply_rn(center_weight, node_values[10])
        );
    }
    kronrod_unscaled = __shfl_sync(0xffffffffu, kronrod_unscaled, 0);

    float weighted_gauss = 0.0f;
    if (lane < 5) {
        const int pair_index = 2 * lane + 1;
        weighted_gauss = multiply_rn(
            add_rn(node_values[pair_index], node_values[pair_index + 11]),
            gauss_weight[lane]
        );
    }
    const float gauss_unscaled = warp_sum(weighted_gauss);
    const float mean = divide_rn(kronrod_unscaled, 2.0f);

    float absolute_weight = 0.0f;
    float ascending_weight = 0.0f;
    if (lane < 10) {
        absolute_weight = multiply_rn(
            add_rn(fabsf(node_values[lane]), fabsf(node_values[lane + 11])),
            wgk[lane]
        );
        ascending_weight = multiply_rn(
            add_rn(
                fabsf(subtract_rn(node_values[lane], mean)),
                fabsf(subtract_rn(node_values[lane + 11], mean))
            ),
            wgk[lane]
        );
    }
    float resabs_sum = warp_sum(absolute_weight);
    float resasc_sum = warp_sum(ascending_weight);
    IntervalResult output{};
    if (lane == 0) {
        output.estimate = multiply_rn(half_width, kronrod_unscaled);
        output.error = multiply_rn(
            half_width,
            fabsf(subtract_rn(kronrod_unscaled, gauss_unscaled))
        );
        const float resabs = multiply_rn(
            half_width,
            add_rn(resabs_sum, multiply_rn(center_weight, fabsf(node_values[10])))
        );
        const float resasc = multiply_rn(
            half_width,
            add_rn(
                resasc_sum,
                multiply_rn(center_weight, fabsf(subtract_rn(node_values[10], mean)))
            )
        );
        const float scale_argument = divide_rn(
            multiply_rn(200.0f, output.error),
            resasc > 0.0f ? resasc : 1.0f
        );
        const float scale = powf(scale_argument, 1.5f);
        if (resasc > 0.0f && output.error > 0.0f) {
            output.error = multiply_rn(resasc, fminf(scale, 1.0f));
        }
        output.error = fmaxf(
            output.error,
            multiply_rn(multiply_rn(50.0f, FLT_EPSILON), resabs)
        );
    }
    return output;
}

__device__ IntervalResult evaluate_interval(
    const K1SliceConstants& slice,
    float lower,
    float upper,
    const K1Parameters& parameters,
    float* node_values
) {
    constexpr float xgk[] = {
        0.9956571630258081f,
        0.9739065285171717f,
        0.9301574913557082f,
        0.8650633666889845f,
        0.7808177265864169f,
        0.6794095682990244f,
        0.5627571346686047f,
        0.4333953941292472f,
        0.2943928627014602f,
        0.1488743389816312f,
    };
    const int lane = threadIdx.x % 32;
    const float center = divide_rn(add_rn(lower, upper), 2.0f);
    const float half_width = divide_rn(subtract_rn(upper, lower), 2.0f);
    if (lane < 21) {
        float theta;
        if (lane < 10) {
            theta = subtract_rn(center, multiply_rn(half_width, xgk[lane]));
        } else if (lane == 10) {
            theta = center;
        } else {
            theta = add_rn(center, multiply_rn(half_width, xgk[lane - 11]));
        }
        node_values[lane] = k1_integrand<false>(slice, cosf(theta), 0.0f, parameters);
    }
    __syncwarp();
    return quadrature_from_nodes(lower, upper, node_values);
}

constexpr int warps_per_block = 4;
constexpr int maximum_intervals = 85;

__device__ float adaptive_theta_integral(
    float r,
    float z,
    int interval_limit,
    float epsrel,
    const K1Parameters& parameters,
    K1SliceConstants* slice,
    float* interval_lower,
    float* interval_upper,
    float* interval_estimate,
    float* interval_error,
    float* node_values
) {
    const int lane = threadIdx.x % 32;
    int interval_count = 1;
    if (lane == 0) {
        *slice = make_slice_constants(r, z, parameters);
        interval_lower[0] = 0.0f;
        interval_upper[0] = static_cast<float>(M_PI);
    }
    __syncwarp();
    const IntervalResult initial = evaluate_interval(
        *slice,
        0.0f,
        static_cast<float>(M_PI),
        parameters,
        node_values
    );
    if (lane == 0) {
        interval_estimate[0] = initial.estimate;
        interval_error[0] = initial.error;
    }
    __syncwarp();

    while (true) {
        float estimate_part = 0.0f;
        float error_part = 0.0f;
        for (int index = lane; index < interval_count; index += 32) {
            estimate_part = add_rn(estimate_part, interval_estimate[index]);
            error_part = add_rn(error_part, interval_error[index]);
        }
        float total_estimate = warp_sum(estimate_part);
        float total_error = warp_sum(error_part);
        total_estimate = __shfl_sync(0xffffffffu, total_estimate, 0);
        total_error = __shfl_sync(0xffffffffu, total_error, 0);
        const bool finished =
            interval_count >= interval_limit || total_error <= epsrel * fabsf(total_estimate);
        if (finished) {
            return total_estimate;
        }

        int split = 0;
        float selected_lower = 0.0f;
        float selected_upper = 0.0f;
        if (lane == 0) {
            for (int index = 1; index < interval_count; ++index) {
                if (interval_error[index] > interval_error[split]) {
                    split = index;
                }
            }
            selected_lower = interval_lower[split];
            selected_upper = interval_upper[split];
        }
        split = __shfl_sync(0xffffffffu, split, 0);
        selected_lower = __shfl_sync(0xffffffffu, selected_lower, 0);
        selected_upper = __shfl_sync(0xffffffffu, selected_upper, 0);
        const float midpoint = divide_rn(add_rn(selected_lower, selected_upper), 2.0f);
        const IntervalResult left = evaluate_interval(
            *slice,
            selected_lower,
            midpoint,
            parameters,
            node_values
        );
        const IntervalResult right = evaluate_interval(
            *slice,
            midpoint,
            selected_upper,
            parameters,
            node_values
        );
        if (lane == 0) {
            for (int index = interval_count; index > split + 1; --index) {
                interval_lower[index] = interval_lower[index - 1];
                interval_upper[index] = interval_upper[index - 1];
                interval_estimate[index] = interval_estimate[index - 1];
                interval_error[index] = interval_error[index - 1];
            }
            interval_lower[split] = selected_lower;
            interval_upper[split] = midpoint;
            interval_estimate[split] = left.estimate;
            interval_error[split] = left.error;
            interval_lower[split + 1] = midpoint;
            interval_upper[split + 1] = selected_upper;
            interval_estimate[split + 1] = right.estimate;
            interval_error[split + 1] = right.error;
        }
        ++interval_count;
        __syncwarp();
    }
}

__global__ void theta_integrals_kernel(
    const float* r_values,
    const float* z_values,
    float* output,
    int rows,
    int r_stride,
    int interval_limit,
    float epsrel,
    K1Parameters parameters
) {
    __shared__ float interval_lower[warps_per_block][maximum_intervals];
    __shared__ float interval_upper[warps_per_block][maximum_intervals];
    __shared__ float interval_estimate[warps_per_block][maximum_intervals];
    __shared__ float interval_error[warps_per_block][maximum_intervals];
    __shared__ float node_values[warps_per_block][32];
    __shared__ K1SliceConstants slice_constants[warps_per_block];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp;
    if (row >= rows) {
        return;
    }
    const float r = r_values[row * r_stride];
    const float z = z_values[row];
    const float result = adaptive_theta_integral(
        r,
        z,
        interval_limit,
        epsrel,
        parameters,
        &slice_constants[warp],
        interval_lower[warp],
        interval_upper[warp],
        interval_estimate[warp],
        interval_error[warp],
        node_values[warp]
    );
    if (lane == 0) {
        output[row] = result;
    }
}

constexpr int radial_tasks_per_parent = 42;
constexpr int task_warps_per_block = 2;

__global__ void radial_tasks_kernel(
    const float* r_values,
    const float* radial_interval_lower,
    const float* radial_interval_upper,
    const int* radial_split,
    const bool* parent_finished,
    float* task_output,
    int* task_counter,
    int work_items,
    int rows,
    int iteration,
    int theta_interval_limit,
    float epsrel,
    K1Parameters parameters
) {
    __shared__ float theta_lower[task_warps_per_block][maximum_intervals];
    __shared__ float theta_upper[task_warps_per_block][maximum_intervals];
    __shared__ float theta_estimate[task_warps_per_block][maximum_intervals];
    __shared__ float theta_error[task_warps_per_block][maximum_intervals];
    __shared__ float theta_nodes[task_warps_per_block][32];
    __shared__ K1SliceConstants slice_constants[task_warps_per_block];
    constexpr float xgk[] = {
        0.9956571630258081f,
        0.9739065285171717f,
        0.9301574913557082f,
        0.8650633666889845f,
        0.7808177265864169f,
        0.6794095682990244f,
        0.5627571346686047f,
        0.4333953941292472f,
        0.2943928627014602f,
        0.1488743389816312f,
    };

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    while (true) {
        int work = lane == 0 ? atomicAdd(task_counter, 1) : 0;
        work = __shfl_sync(0xffffffffu, work, 0);
        if (work >= work_items) {
            return;
        }
        const int parent = iteration == 0
            ? work / 21
            : work / radial_tasks_per_parent;
        const int task_node = iteration == 0
            ? work % 21
            : work % radial_tasks_per_parent;
        const int task = parent * radial_tasks_per_parent + task_node;
        if (parent_finished[parent]) {
            continue;
        }

        float lower;
        float upper;
        int node;
        if (iteration == 0) {
            lower = logf(parameters.r_grid[0]);
            upper = logf(parameters.r_grid[parameters.grid_points - 1]);
            node = task_node;
        } else {
            const int interval_offset = parent * maximum_intervals;
            const int selected = radial_split[parent];
            const float selected_lower = radial_interval_lower[interval_offset + selected];
            const float selected_upper = radial_interval_upper[interval_offset + selected];
            const float midpoint = divide_rn(add_rn(selected_lower, selected_upper), 2.0f);
            const bool right = task_node >= 21;
            lower = right ? midpoint : selected_lower;
            upper = right ? selected_upper : midpoint;
            node = right ? task_node - 21 : task_node;
        }

        const float center = divide_rn(add_rn(lower, upper), 2.0f);
        const float half_width = divide_rn(subtract_rn(upper, lower), 2.0f);
        float log_z;
        if (node < 10) {
            log_z = subtract_rn(center, multiply_rn(half_width, xgk[node]));
        } else if (node == 10) {
            log_z = center;
        } else {
            log_z = add_rn(center, multiply_rn(half_width, xgk[node - 11]));
        }
        const float theta_integral = adaptive_theta_integral(
            r_values[parent],
            expf(log_z),
            theta_interval_limit,
            epsrel,
            parameters,
            &slice_constants[warp],
            theta_lower[warp],
            theta_upper[warp],
            theta_estimate[warp],
            theta_error[warp],
            theta_nodes[warp]
        );
        if (lane == 0) {
            task_output[task] = multiply_rn(
                multiply_rn(theta_integral, expf(multiply_rn(2.0f, log_z))),
                2.0f
            );
        }
        __syncwarp();
    }
}

__global__ void radial_controller_kernel(
    const float* task_output,
    float* radial_interval_lower,
    float* radial_interval_upper,
    float* radial_interval_estimate,
    float* radial_interval_error,
    int* radial_interval_count,
    int* radial_split,
    bool* parent_finished,
    float* output,
    float* output_error,
    int64_t* output_evaluations,
    bool* output_converged,
    int rows,
    int iteration,
    int radial_interval_limit,
    float epsrel,
    K1Parameters parameters
) {
    __shared__ float node_values[warps_per_block][32];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int parent = blockIdx.x * warps_per_block + warp;
    if (parent >= rows || parent_finished[parent]) {
        return;
    }

    const int task_offset = parent * radial_tasks_per_parent;
    const int interval_offset = parent * maximum_intervals;
    IntervalResult left{};
    IntervalResult right{};
    if (lane < 21) {
        node_values[warp][lane] = task_output[task_offset + lane];
    }
    __syncwarp();

    if (iteration == 0) {
        const float lower = logf(parameters.r_grid[0]);
        const float upper = logf(parameters.r_grid[parameters.grid_points - 1]);
        left = quadrature_from_nodes(lower, upper, node_values[warp]);
    } else {
        const int selected = radial_split[parent];
        const float selected_lower = radial_interval_lower[interval_offset + selected];
        const float selected_upper = radial_interval_upper[interval_offset + selected];
        const float midpoint = divide_rn(add_rn(selected_lower, selected_upper), 2.0f);
        left = quadrature_from_nodes(selected_lower, midpoint, node_values[warp]);
        if (lane < 21) {
            node_values[warp][lane] = task_output[task_offset + 21 + lane];
        }
        __syncwarp();
        right = quadrature_from_nodes(midpoint, selected_upper, node_values[warp]);
    }

    if (lane == 0) {
        if (iteration == 0) {
            radial_interval_lower[interval_offset] = logf(parameters.r_grid[0]);
            radial_interval_upper[interval_offset] =
                logf(parameters.r_grid[parameters.grid_points - 1]);
            radial_interval_estimate[interval_offset] = left.estimate;
            radial_interval_error[interval_offset] = left.error;
            radial_interval_count[parent] = 1;
        } else {
            const int interval_count = radial_interval_count[parent];
            const int selected = radial_split[parent];
            const float selected_lower = radial_interval_lower[interval_offset + selected];
            const float selected_upper = radial_interval_upper[interval_offset + selected];
            const float midpoint = divide_rn(add_rn(selected_lower, selected_upper), 2.0f);
            for (int index = interval_count; index > selected + 1; --index) {
                radial_interval_lower[interval_offset + index] =
                    radial_interval_lower[interval_offset + index - 1];
                radial_interval_upper[interval_offset + index] =
                    radial_interval_upper[interval_offset + index - 1];
                radial_interval_estimate[interval_offset + index] =
                    radial_interval_estimate[interval_offset + index - 1];
                radial_interval_error[interval_offset + index] =
                    radial_interval_error[interval_offset + index - 1];
            }
            radial_interval_lower[interval_offset + selected] = selected_lower;
            radial_interval_upper[interval_offset + selected] = midpoint;
            radial_interval_estimate[interval_offset + selected] = left.estimate;
            radial_interval_error[interval_offset + selected] = left.error;
            radial_interval_lower[interval_offset + selected + 1] = midpoint;
            radial_interval_upper[interval_offset + selected + 1] = selected_upper;
            radial_interval_estimate[interval_offset + selected + 1] = right.estimate;
            radial_interval_error[interval_offset + selected + 1] = right.error;
            radial_interval_count[parent] = interval_count + 1;
        }
    }
    __syncwarp();

    const int interval_count = radial_interval_count[parent];
    float estimate_part = 0.0f;
    float error_part = 0.0f;
    for (int index = lane; index < interval_count; index += 32) {
        estimate_part = add_rn(
            estimate_part,
            radial_interval_estimate[interval_offset + index]
        );
        error_part = add_rn(error_part, radial_interval_error[interval_offset + index]);
    }
    const float total_estimate = warp_sum(estimate_part);
    const float total_error = warp_sum(error_part);
    if (lane == 0) {
        const bool converged = total_error <= epsrel * fabsf(total_estimate);
        output[parent] = total_estimate;
        output_error[parent] = total_error;
        output_evaluations[parent] = 21 + 42 * (interval_count - 1);
        output_converged[parent] = converged;
        parent_finished[parent] = converged || interval_count >= radial_interval_limit;
        if (!parent_finished[parent]) {
            int selected = 0;
            for (int index = 1; index < interval_count; ++index) {
                if (
                    radial_interval_error[interval_offset + index] >
                    radial_interval_error[interval_offset + selected]
                ) {
                    selected = index;
                }
            }
            radial_split[parent] = selected;
        }
    }
}

template <int Order>
struct GaussLegendreRule;

template <>
struct GaussLegendreRule<6> {
    __device__ static float node(int index) {
        constexpr float values[] = {
            -0.93246951420315194f,
            -0.66120938646626448f,
            -0.23861918608319690f,
            0.23861918608319690f,
            0.66120938646626448f,
            0.93246951420315194f,
        };
        return values[index];
    }

    __device__ static double weight(int index) {
        constexpr double values[] = {
            0.17132449237917027,
            0.36076157304813872,
            0.46791393457269104,
            0.46791393457269104,
            0.36076157304813872,
            0.17132449237917027,
        };
        return values[index];
    }
};

template <>
struct GaussLegendreRule<8> {
    __device__ static float node(int index) {
        constexpr float values[] = {
            -0.96028985649753618f,
            -0.79666647741362673f,
            -0.52553240991632899f,
            -0.18343464249564978f,
            0.18343464249564978f,
            0.52553240991632899f,
            0.79666647741362673f,
            0.96028985649753618f,
        };
        return values[index];
    }

    __device__ static double weight(int index) {
        constexpr double values[] = {
            0.10122853629037706,
            0.22238103445337443,
            0.31370664587788688,
            0.36268378337836166,
            0.36268378337836166,
            0.31370664587788688,
            0.22238103445337443,
            0.10122853629037706,
        };
        return values[index];
    }
};

template <>
struct GaussLegendreRule<20> {
    __device__ static float node(int index) {
        constexpr float values[] = {
            -0.99312859918509500f,
            -0.96397192727791381f,
            -0.91223442825132595f,
            -0.83911697182221878f,
            -0.74633190646015080f,
            -0.63605368072651502f,
            -0.51086700195082713f,
            -0.37370608871541955f,
            -0.22778585114164507f,
            -0.07652652113349734f,
            0.07652652113349734f,
            0.22778585114164507f,
            0.37370608871541955f,
            0.51086700195082713f,
            0.63605368072651502f,
            0.74633190646015080f,
            0.83911697182221878f,
            0.91223442825132595f,
            0.96397192727791381f,
            0.99312859918509500f,
        };
        return values[index];
    }

    __device__ static double weight(int index) {
        constexpr double values[] = {
            0.017614007139150893,
            0.040601429800386446,
            0.062672048334108790,
            0.083276741576704713,
            0.101930119817240710,
            0.118194531961518610,
            0.131688638449176890,
            0.142096109318382400,
            0.149172986472604240,
            0.152753387130726280,
            0.152753387130726280,
            0.149172986472604240,
            0.142096109318382400,
            0.131688638449176890,
            0.118194531961518610,
            0.101930119817240710,
            0.083276741576704713,
            0.062672048334108790,
            0.040601429800386446,
            0.017614007139150893,
        };
        return values[index];
    }
};

template <>
struct GaussLegendreRule<24> {
    __device__ static float node(int index) {
        constexpr float values[] = {
            -0.99518721999702131f,
            -0.97472855597130947f,
            -0.93827455200273280f,
            -0.88641552700440107f,
            -0.82000198597390295f,
            -0.74012419157855436f,
            -0.64809365193697555f,
            -0.54542147138883956f,
            -0.43379350762604513f,
            -0.31504267969616340f,
            -0.19111886747361631f,
            -0.06405689286260563f,
            0.06405689286260563f,
            0.19111886747361631f,
            0.31504267969616340f,
            0.43379350762604513f,
            0.54542147138883956f,
            0.64809365193697555f,
            0.74012419157855436f,
            0.82000198597390295f,
            0.88641552700440107f,
            0.93827455200273280f,
            0.97472855597130947f,
            0.99518721999702131f,
        };
        return values[index];
    }

    __device__ static double weight(int index) {
        constexpr double values[] = {
            0.012341229799988690,
            0.028531388628933559,
            0.044277438817419412,
            0.059298584915436360,
            0.073346481411080161,
            0.086190161531953205,
            0.097618652104113926,
            0.107444270115965560,
            0.115505668053725520,
            0.121670472927803290,
            0.125837456346828250,
            0.127938195346752020,
            0.127938195346752020,
            0.125837456346828250,
            0.121670472927803290,
            0.115505668053725520,
            0.107444270115965560,
            0.097618652104113926,
            0.086190161531953205,
            0.073346481411080161,
            0.059298584915436360,
            0.044277438817419412,
            0.028531388628933559,
            0.012341229799988690,
        };
        return values[index];
    }
};

constexpr int fixed_grid_warps_per_block = 8;
constexpr int original_fixed_grid_radial_splits = 3;
constexpr int singular_radial_parts = 4;
constexpr int singular_angular_parts = 32;
constexpr int singular_angular_order = 24;

template <bool RefineSingularPanels>
struct SingularSharedStorage;

template <>
struct SingularSharedStorage<false> {
    float sin_half_squared[1];
    double angular_weights[1];
};

template <>
struct SingularSharedStorage<true> {
    float sin_half_squared[singular_angular_parts * singular_angular_order];
    double angular_weights[singular_angular_parts * singular_angular_order];
};

template <
    int RadialOrder,
    int AngularOrder,
    int RadialSplits,
    bool RefineSingularPanels,
    bool StableGeometry,
    bool OriginalFixedPath
>
__device__ double fixed_grid_rule(
    float r,
    float parent_squared,
    float parent_n,
    float parent_alpha,
    int parent_knot,
    int radial_split,
    int warp,
    K1SliceConstants* slice_constants,
    float* radial_measures,
    double* radial_weights,
    double* angular_values,
    const float* singular_sin_half_squared,
    const double* singular_angular_weights,
    const K1Parameters& parameters,
    bool exclude_singular_panels
) {
    const int lane = threadIdx.x % 32;
    const int packed_index = threadIdx.x;
    const bool evaluates_angle = packed_index < fixed_grid_warps_per_block * AngularOrder;
    const int radial_slot = evaluates_angle ? packed_index / AngularOrder : 0;
    const int angular_node = evaluates_angle ? packed_index % AngularOrder : 0;
    double warp_total = 0.0;
    float base_v = 0.0f;
    float base_cos_theta = 0.0f;
    float base_sin_half_squared = 0.0f;
    double base_angular_weight = 0.0;
    if (evaluates_angle) {
        base_v = divide_rn(
            add_rn(GaussLegendreRule<AngularOrder>::node(angular_node), 1.0f),
            2.0f
        );
        const float theta = multiply_rn(
            static_cast<float>(M_PI),
            multiply_rn(base_v, base_v)
        );
        if constexpr (StableGeometry) {
            const float sin_half = sinf(divide_rn(theta, 2.0f));
            base_sin_half_squared = multiply_rn(sin_half, sin_half);
        } else {
            base_cos_theta = cosf(theta);
        }
        base_angular_weight = GaussLegendreRule<AngularOrder>::weight(angular_node)
            * M_PI * static_cast<double>(base_v);
    }
    const int radial_rows = (parameters.grid_points - 1) * RadialOrder;
    const int radial_batches =
        (radial_rows + fixed_grid_warps_per_block - 1) / fixed_grid_warps_per_block;
    const int batches_per_split = (radial_batches + RadialSplits - 1) / RadialSplits;
    const int radial_start = radial_split * batches_per_split * fixed_grid_warps_per_block;
    const int radial_end = min(
        radial_rows,
        radial_start + batches_per_split * fixed_grid_warps_per_block
    );
    int setup_iteration = 0;
    for (
        int radial_base = radial_start;
        radial_base < radial_end;
        radial_base += fixed_grid_warps_per_block
    ) {
        const int batch_rows = min(fixed_grid_warps_per_block, radial_rows - radial_base);
        const int batch_interval = radial_base / RadialOrder;
        if (
            exclude_singular_panels &&
            (batch_interval == parent_knot - 1 || batch_interval == parent_knot)
        ) {
            continue;
        }
        const bool refine = RefineSingularPanels
            && (batch_interval == parent_knot - 1 || batch_interval == parent_knot);
        const int radial_parts = refine ? singular_radial_parts : 1;
        const int angular_parts = refine ? singular_angular_parts : 1;

        for (int radial_part = 0; radial_part < radial_parts; ++radial_part) {
            // The original rule alternates its small setup buffers. A warp can begin
            // preparing the next batch without overwriting weights still in use by
            // another warp, so the end-of-batch block barrier is unnecessary.
            const int radial_buffer = OriginalFixedPath ? (setup_iteration & 1) : 0;
            K1SliceConstants* batch_slice_constants =
                slice_constants + radial_buffer * fixed_grid_warps_per_block;
            float* batch_radial_measures =
                radial_measures + radial_buffer * fixed_grid_warps_per_block;
            double* batch_radial_weights =
                radial_weights + radial_buffer * fixed_grid_warps_per_block;
            // Pack the eight per-row setup calculations into one warp instead of using
            // one active lane in each of eight warps.
            if (threadIdx.x < batch_rows) {
                const int setup_slot = threadIdx.x;
                const int radial_row = radial_base + setup_slot;
                const int interval = radial_row / RadialOrder;
                const int radial_node = radial_row % RadialOrder;
                const float panel_lower = parameters.log_grid[interval];
                const float panel_upper = parameters.log_grid[interval + 1];
                const float panel_width = subtract_rn(panel_upper, panel_lower);
                const float lower = refine
                    ? add_rn(
                          panel_lower,
                          multiply_rn(
                              divide_rn(panel_width, static_cast<float>(radial_parts)),
                              static_cast<float>(radial_part)
                          )
                      )
                    : panel_lower;
                const float upper = refine
                    ? add_rn(
                          panel_lower,
                          multiply_rn(
                              divide_rn(panel_width, static_cast<float>(radial_parts)),
                              static_cast<float>(radial_part + 1)
                          )
                      )
                    : panel_upper;
                const float center = divide_rn(add_rn(lower, upper), 2.0f);
                const float half_width = divide_rn(subtract_rn(upper, lower), 2.0f);
                const float log_z = add_rn(
                    center,
                    multiply_rn(
                        half_width,
                        GaussLegendreRule<RadialOrder>::node(radial_node)
                    )
                );
                const float z = expf(log_z);
                batch_slice_constants[setup_slot] = make_fixed_slice_constants(
                    r,
                    z,
                    parent_squared,
                    parent_n,
                    parent_alpha,
                    parameters
                );
                batch_radial_measures[setup_slot] = multiply_rn(
                    2.0f,
                    expf(multiply_rn(2.0f, log_z))
                );
                batch_radial_weights[setup_slot] = static_cast<double>(half_width)
                    * GaussLegendreRule<RadialOrder>::weight(radial_node);
            }
            __syncthreads();

            double angular_node_total = 0.0;
            if (evaluates_angle && radial_slot < batch_rows) {
                for (int angular_part_index = 0; angular_part_index < angular_parts;
                     ++angular_part_index) {
                    const int singular_angular_index =
                        angular_part_index * AngularOrder + angular_node;
                    const float sin_half_squared = refine
                        ? singular_sin_half_squared[singular_angular_index]
                        : base_sin_half_squared;
                    const double angular_weight = refine
                        ? singular_angular_weights[singular_angular_index]
                        : base_angular_weight;
                    const float integrand = multiply_rn(
                        k1_integrand<StableGeometry, OriginalFixedPath>(
                            batch_slice_constants[radial_slot],
                            base_cos_theta,
                            sin_half_squared,
                            parameters
                        ),
                        batch_radial_measures[radial_slot]
                    );
                    angular_node_total += static_cast<double>(integrand) * angular_weight;
                }
                angular_values[radial_slot * 32 + angular_node] = angular_node_total;
            }
            __syncthreads();

            double angular_value = lane < AngularOrder && warp < batch_rows
                ? angular_values[warp * 32 + lane]
                : 0.0;
            for (int offset = 16; offset > 0; offset /= 2) {
                angular_value += __shfl_down_sync(0xffffffffu, angular_value, offset);
            }
            if (lane == 0 && warp < batch_rows) {
                warp_total += angular_value * batch_radial_weights[warp];
            }
            if constexpr (!OriginalFixedPath) {
                __syncthreads();
            }
            ++setup_iteration;
        }
    }
    return warp_total;
}

template <bool RefineSingularPanels, bool FineOnly = false>
__global__ void fixed_grid_integrals_kernel(
    const float* r_values,
    double* coarse_output,
    double* fine_output,
    int rows,
    K1Parameters parameters,
    bool exclude_singular_panels
) {
    // Three independent radial ranges give the original rule enough blocks to
    // occupy the GPU. The refined rule keeps one block because its two denser
    // singular-neighbor panels have a different work distribution.
    constexpr int radial_splits = RefineSingularPanels
        ? 1
        : original_fixed_grid_radial_splits;
    constexpr int radial_buffers = RefineSingularPanels ? 1 : 2;
    const int radial_split = blockIdx.x % radial_splits;
    const int work_index = blockIdx.x / radial_splits;
    const bool fine_rule = FineOnly || work_index >= rows;
    const int parent = FineOnly ? work_index : (fine_rule ? work_index - rows : work_index);
    if (parent >= rows) {
        return;
    }

    __shared__ float parent_squared;
    __shared__ float parent_n;
    __shared__ float parent_alpha;
    __shared__ int parent_knot;
    __shared__ K1SliceConstants
        slice_constants[radial_buffers * fixed_grid_warps_per_block];
    __shared__ float radial_measures[radial_buffers * fixed_grid_warps_per_block];
    __shared__ double radial_weights[radial_buffers * fixed_grid_warps_per_block];
    __shared__ double angular_values[fixed_grid_warps_per_block * 32];
    __shared__ SingularSharedStorage<RefineSingularPanels> singular;
    __shared__ double warp_totals[fixed_grid_warps_per_block];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const float r = r_values[parent];
    if (threadIdx.x == 0) {
        parent_squared = multiply_rn(r, r);
        parent_n = fminf(
            n_spline(
                r,
                parameters.r_grid,
                parameters.log_grid,
                parameters.a,
                parameters.b,
                parameters.c,
                parameters.d,
                parameters.grid_points,
                parameters.search_steps
            ),
            1.0f
        );
        parent_alpha = smooth_alpha_s(
            r,
            parameters.coupling_b0,
            parameters.coupling_lambda_squared,
            parameters.coupling_scale_numerator,
            parameters.coupling_log_mu0_term
        );
        if (RefineSingularPanels || exclude_singular_panels) {
            const float log_r = logf(r);
            const float grid_position = divide_rn(
                multiply_rn(
                    subtract_rn(log_r, parameters.log_grid[0]),
                    static_cast<float>(parameters.grid_points - 1)
                ),
                subtract_rn(
                    parameters.log_grid[parameters.grid_points - 1],
                    parameters.log_grid[0]
                )
            );
            parent_knot = min(
                max(__float2int_rn(grid_position), 1),
                parameters.grid_points - 2
            );
        } else {
            parent_knot = 0;
        }
    }
    if constexpr (RefineSingularPanels) {
        if (fine_rule) {
            for (
                int index = threadIdx.x;
                index < singular_angular_parts * singular_angular_order;
                index += blockDim.x
            ) {
                const int angular_part = index / singular_angular_order;
                const int angular_node = index % singular_angular_order;
                const float base_v = divide_rn(
                    add_rn(GaussLegendreRule<singular_angular_order>::node(angular_node), 1.0f),
                    2.0f
                );
                const float v = divide_rn(
                    add_rn(base_v, static_cast<float>(angular_part)),
                    static_cast<float>(singular_angular_parts)
                );
                const float theta = multiply_rn(
                    static_cast<float>(M_PI),
                    multiply_rn(v, v)
                );
                const float sin_half = sinf(divide_rn(theta, 2.0f));
                singular.sin_half_squared[index] = multiply_rn(sin_half, sin_half);
                singular.angular_weights[index] =
                    GaussLegendreRule<singular_angular_order>::weight(angular_node)
                    * M_PI * static_cast<double>(v)
                    / static_cast<double>(singular_angular_parts);
            }
        }
    }
    __syncthreads();

    double warp_total;
    if (fine_rule) {
        warp_total = fixed_grid_rule<
            8,
            24,
            radial_splits,
            RefineSingularPanels,
            RefineSingularPanels,
            !RefineSingularPanels
        >(
              r,
              parent_squared,
              parent_n,
              parent_alpha,
              parent_knot,
              radial_split,
              warp,
              slice_constants,
              radial_measures,
              radial_weights,
              angular_values,
              singular.sin_half_squared,
              singular.angular_weights,
              parameters,
              exclude_singular_panels
          );
    } else {
        warp_total = fixed_grid_rule<
            6,
            20,
            radial_splits,
            false,
            RefineSingularPanels,
            !RefineSingularPanels
        >(
              r,
              parent_squared,
              parent_n,
              parent_alpha,
              parent_knot,
              radial_split,
              warp,
              slice_constants,
              radial_measures,
              radial_weights,
              angular_values,
              singular.sin_half_squared,
              singular.angular_weights,
              parameters,
              false
          );
    }

    if (lane == 0) {
        warp_totals[warp] = warp_total;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double total = 0.0;
        for (int index = 0; index < fixed_grid_warps_per_block; ++index) {
            total += warp_totals[index];
        }
        if (fine_rule) {
            if constexpr (RefineSingularPanels) {
                fine_output[parent] = total;
            } else {
                atomicAdd(&fine_output[parent], total);
            }
        } else {
            if constexpr (RefineSingularPanels) {
                coarse_output[parent] = total;
            } else {
                atomicAdd(&coarse_output[parent], total);
            }
        }
    }
}

__device__ double singular_radial_node64(int index) {
    constexpr double values[] = {
        -0.96028985649753618,
        -0.79666647741362673,
        -0.52553240991632899,
        -0.18343464249564978,
        0.18343464249564978,
        0.52553240991632899,
        0.79666647741362673,
        0.96028985649753618,
    };
    return values[index];
}

__device__ double singular_angular_node64(int index) {
    constexpr double values[] = {
        -0.99518721999702131,
        -0.97472855597130947,
        -0.93827455200273280,
        -0.88641552700440107,
        -0.82000198597390295,
        -0.74012419157855436,
        -0.64809365193697555,
        -0.54542147138883956,
        -0.43379350762604513,
        -0.31504267969616340,
        -0.19111886747361631,
        -0.06405689286260563,
        0.06405689286260563,
        0.19111886747361631,
        0.31504267969616340,
        0.43379350762604513,
        0.54542147138883956,
        0.64809365193697555,
        0.74012419157855436,
        0.82000198597390295,
        0.88641552700440107,
        0.93827455200273280,
        0.97472855597130947,
        0.99518721999702131,
    };
    return values[index];
}

__device__ __forceinline__ double warp_sum64(double value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void k1_sensitive_panels64_kernel(
    const float* r_values,
    const int64_t* parent_index,
    double* mixed_output,
    int rows,
    K1Parameters64 parameters
) {
    constexpr int radial_order = 8;
    constexpr int angular_order = 24;
    constexpr int samples_per_parent = 2 * radial_order * angular_order;
    __shared__ double warp_totals[8];
    const int parent = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (parent >= rows) {
        return;
    }

    const int requested_knot = static_cast<int>(parent_index[parent]);
    const int knot = min(max(requested_knot, 1), parameters.grid_points - 2);
    const double r = static_cast<double>(r_values[parent]);
    double total = 0.0;
    for (int sample = threadIdx.x; sample < samples_per_parent; sample += blockDim.x) {
        const int angular_node = sample % angular_order;
        const int radial_sample = sample / angular_order;
        const int radial_node = radial_sample % radial_order;
        const int panel = radial_sample / radial_order;
        const int interval = knot - 1 + panel;
        const double lower = parameters.log_grid[interval];
        const double upper = parameters.log_grid[interval + 1];
        const double center = (lower + upper) / 2.0;
        const double half_width = (upper - lower) / 2.0;
        const double log_z = center + half_width * singular_radial_node64(radial_node);
        const double z = exp(log_z);
        const double v = (singular_angular_node64(angular_node) + 1.0) / 2.0;
        const double theta = M_PI * v * v;
        const double radial_weight =
            half_width * GaussLegendreRule<radial_order>::weight(radial_node);
        const double angular_weight =
            GaussLegendreRule<angular_order>::weight(angular_node) * M_PI * v;
        total += k1_integrand64(r, z, theta, parameters) *
            (2.0 * z * z) * radial_weight * angular_weight;
    }
    total = warp_sum64(total);
    if (lane == 0) {
        warp_totals[warp] = total;
    }
    __syncthreads();
    if (warp == 0) {
        const double warp_value = lane < 8 ? warp_totals[lane] : 0.0;
        const double block_total = warp_sum64(warp_value);
        if (lane == 0) {
            mixed_output[parent] += block_total;
        }
    }
}

__global__ void k1_full_grid64_kernel(
    const float* r_values,
    double* output,
    int rows,
    K1Parameters64 parameters
) {
    constexpr int radial_order = 8;
    constexpr int angular_order = 24;
    constexpr int samples_per_interval = radial_order * angular_order;
    __shared__ double warp_totals[8];
    const int parent = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (parent >= rows) {
        return;
    }

    const int samples = (parameters.grid_points - 1) * samples_per_interval;
    const double r = static_cast<double>(r_values[parent]);
    double total = 0.0;
    for (int sample = threadIdx.x; sample < samples; sample += blockDim.x) {
        const int interval = sample / samples_per_interval;
        const int local_sample = sample % samples_per_interval;
        const int radial_node = local_sample / angular_order;
        const int angular_node = local_sample % angular_order;
        const double lower = parameters.log_grid[interval];
        const double upper = parameters.log_grid[interval + 1];
        const double center = (lower + upper) / 2.0;
        const double half_width = (upper - lower) / 2.0;
        const double log_z = center + half_width * singular_radial_node64(radial_node);
        const double z = exp(log_z);
        const double v = (singular_angular_node64(angular_node) + 1.0) / 2.0;
        const double theta = M_PI * v * v;
        const double radial_weight =
            half_width * GaussLegendreRule<radial_order>::weight(radial_node);
        const double angular_weight =
            GaussLegendreRule<angular_order>::weight(angular_node) * M_PI * v;
        total += k1_integrand64(r, z, theta, parameters) *
            (2.0 * z * z) * radial_weight * angular_weight;
    }
    total = warp_sum64(total);
    if (lane == 0) {
        warp_totals[warp] = total;
    }
    __syncthreads();
    if (warp == 0) {
        const double warp_value = lane < 8 ? warp_totals[lane] : 0.0;
        const double block_total = warp_sum64(warp_value);
        if (lane == 0) {
            output[parent] = block_total;
        }
    }
}

__global__ void k1_mixed_finalize_kernel(
    const double* mixed_output,
    float* output,
    int rows
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows) {
        output[index] = static_cast<float>(mixed_output[index]);
    }
}

__global__ void bessel_values_kernel(
    const float* input,
    float* j1_output,
    float* i1_output,
    int64_t elements
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const float value = input[index];
    j1_output[index] = pytorch_bessel_j1(value);
    i1_output[index] = pytorch_bessel_i1(value);
}

std::vector<torch::Tensor> bessel_values(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    auto j1 = torch::empty_like(input);
    auto i1 = torch::empty_like(input);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((input.numel() + threads - 1) / threads);
    bessel_values_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        j1.data_ptr<float>(),
        i1.data_ptr<float>(),
        input.numel()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {j1, i1};
}

torch::Tensor theta_integrals(
    torch::Tensor r,
    torch::Tensor z,
    torch::Tensor r_grid,
    torch::Tensor log_grid,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d,
    int64_t interval_limit,
    double epsrel,
    double nc,
    int64_t nf,
    double minimum_r,
    double ksub,
    double coupling_b0,
    double coupling_lambda_squared,
    double coupling_scale_numerator,
    double coupling_log_mu0_term
) {
    TORCH_CHECK(r.is_cuda() && z.is_cuda(), "r and z must be on CUDA");
    TORCH_CHECK(r.scalar_type() == torch::kFloat32, "r must be float32");
    TORCH_CHECK(z.scalar_type() == torch::kFloat32, "z must be float32");
    TORCH_CHECK(r.is_contiguous(), "r must be contiguous");
    TORCH_CHECK(r.numel() == 1 || r.numel() == z.numel(), "r must be scalar or match z");
    TORCH_CHECK(z.dim() == 1 && z.is_contiguous(), "z must be contiguous and one-dimensional");
    TORCH_CHECK(interval_limit > 0 && interval_limit <= maximum_intervals, "invalid interval limit");
    const std::vector<torch::Tensor> spline_tensors = {r_grid, log_grid, a, b, c, d};
    for (const auto& tensor : spline_tensors) {
        TORCH_CHECK(tensor.is_cuda(), "spline tensors must be on CUDA");
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "spline tensors must be float32");
        TORCH_CHECK(tensor.dim() == 1 && tensor.is_contiguous(), "spline tensors must be contiguous");
        TORCH_CHECK(tensor.device() == z.device(), "all inputs must use one CUDA device");
    }
    TORCH_CHECK(r_grid.numel() == log_grid.numel(), "r and log grids must have equal lengths");
    TORCH_CHECK(a.numel() + 1 == r_grid.numel(), "invalid spline coefficient length");
    TORCH_CHECK(
        b.numel() == a.numel() && c.numel() == a.numel() && d.numel() == a.numel(),
        "spline coefficients must have equal lengths"
    );

    auto output = torch::empty_like(z);
    const int grid_points = static_cast<int>(r_grid.numel());
    int search_steps = 0;
    for (int values = grid_points - 1; values > 0; values >>= 1) {
        ++search_steps;
    }
    K1Parameters parameters{
        r_grid.data_ptr<float>(),
        log_grid.data_ptr<float>(),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        d.data_ptr<float>(),
        grid_points,
        search_steps,
        static_cast<float>(nc),
        static_cast<float>(nf),
        static_cast<float>(minimum_r),
        static_cast<float>(ksub),
        static_cast<float>(coupling_b0),
        static_cast<float>(coupling_lambda_squared),
        static_cast<float>(coupling_scale_numerator),
        static_cast<float>(coupling_log_mu0_term),
    };
    constexpr int threads = warps_per_block * 32;
    const int blocks = static_cast<int>((z.numel() + warps_per_block - 1) / warps_per_block);
    theta_integrals_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        r.data_ptr<float>(),
        z.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(z.numel()),
        r.numel() == 1 ? 0 : 1,
        static_cast<int>(interval_limit),
        static_cast<float>(epsrel),
        parameters
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<torch::Tensor> fixed_grid_integrals(
    torch::Tensor r,
    torch::Tensor r_grid,
    torch::Tensor log_grid,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d,
    double nc,
    int64_t nf,
    double minimum_r,
    double ksub,
    double coupling_b0,
    double coupling_lambda_squared,
    double coupling_scale_numerator,
    double coupling_log_mu0_term,
    bool refine,
    bool exclude_singular_panels
) {
    TORCH_CHECK(r.is_cuda(), "r must be on CUDA");
    TORCH_CHECK(r.scalar_type() == torch::kFloat32, "r must be float32");
    TORCH_CHECK(r.dim() == 1 && r.is_contiguous(), "r must be contiguous and one-dimensional");
    TORCH_CHECK(r.numel() > 0, "r must contain at least one parent");
    const std::vector<torch::Tensor> spline_tensors = {r_grid, log_grid, a, b, c, d};
    for (const auto& tensor : spline_tensors) {
        TORCH_CHECK(tensor.is_cuda(), "spline tensors must be on CUDA");
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "spline tensors must be float32");
        TORCH_CHECK(tensor.dim() == 1 && tensor.is_contiguous(), "spline tensors must be contiguous");
        TORCH_CHECK(tensor.device() == r.device(), "all inputs must use one CUDA device");
    }
    TORCH_CHECK(r_grid.numel() == log_grid.numel(), "r and log grids must have equal lengths");
    TORCH_CHECK(a.numel() + 1 == r_grid.numel(), "invalid spline coefficient length");
    TORCH_CHECK(
        b.numel() == a.numel() && c.numel() == a.numel() && d.numel() == a.numel(),
        "spline coefficients must have equal lengths"
    );

    auto output_options = r.options().dtype(torch::kFloat64);
    auto coarse_output = torch::empty(r.sizes(), output_options);
    auto fine_output = torch::empty(r.sizes(), output_options);
    const int grid_points = static_cast<int>(r_grid.numel());
    int search_steps = 0;
    for (int values = grid_points - 1; values > 0; values >>= 1) {
        ++search_steps;
    }
    K1Parameters parameters{
        r_grid.data_ptr<float>(),
        log_grid.data_ptr<float>(),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        d.data_ptr<float>(),
        grid_points,
        search_steps,
        static_cast<float>(nc),
        static_cast<float>(nf),
        static_cast<float>(minimum_r),
        static_cast<float>(ksub),
        static_cast<float>(coupling_b0),
        static_cast<float>(coupling_lambda_squared),
        static_cast<float>(coupling_scale_numerator),
        static_cast<float>(coupling_log_mu0_term),
    };
    const int rows = static_cast<int>(r.numel());
    constexpr int threads = fixed_grid_warps_per_block * 32;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (refine) {
        fixed_grid_integrals_kernel<true><<<2 * rows, threads, 0, stream>>>(
            r.data_ptr<float>(),
            coarse_output.data_ptr<double>(),
            fine_output.data_ptr<double>(),
            rows,
            parameters,
            exclude_singular_panels
        );
    } else {
        C10_CUDA_CHECK(cudaMemsetAsync(
            coarse_output.data_ptr<double>(),
            0,
            coarse_output.nbytes(),
            stream
        ));
        C10_CUDA_CHECK(cudaMemsetAsync(
            fine_output.data_ptr<double>(),
            0,
            fine_output.nbytes(),
            stream
        ));
        const int blocks = 2 * rows * original_fixed_grid_radial_splits;
        fixed_grid_integrals_kernel<false><<<blocks, threads, 0, stream>>>(
            r.data_ptr<float>(),
            coarse_output.data_ptr<double>(),
            fine_output.data_ptr<double>(),
            rows,
            parameters,
            exclude_singular_panels
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {coarse_output, fine_output};
}

torch::Tensor mixed_fixed_grid_integrals(
    torch::Tensor r,
    torch::Tensor parent_index,
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
    double nc,
    int64_t nf,
    double minimum_r,
    double ksub,
    double coupling_b0,
    double coupling_lambda_squared,
    double coupling_scale_numerator,
    double coupling_log_mu0_term
) {
    TORCH_CHECK(
        r.is_cuda() && r.scalar_type() == torch::kFloat32 && r.dim() == 1 &&
            r.is_contiguous() && r.numel() > 0,
        "r must be a non-empty contiguous CUDA float32 vector"
    );
    TORCH_CHECK(
        parent_index.is_cuda() && parent_index.scalar_type() == torch::kInt64 &&
            parent_index.dim() == 1 && parent_index.is_contiguous() &&
            parent_index.sizes() == r.sizes(),
        "parent_index must be a matching contiguous CUDA int64 vector"
    );
    for (const auto& tensor : {r_grid, log_grid, a, b, c, d}) {
        TORCH_CHECK(
            tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32 && tensor.dim() == 1 &&
                tensor.is_contiguous(),
            "regular spline tensors must be contiguous CUDA float32 vectors"
        );
        TORCH_CHECK(tensor.device() == r.device(), "all inputs must use one CUDA device");
    }
    for (const auto& tensor : {r_grid64, log_grid64, a64, b64, c64, d64}) {
        TORCH_CHECK(
            tensor.is_cuda() && tensor.scalar_type() == torch::kFloat64 && tensor.dim() == 1 &&
                tensor.is_contiguous(),
            "sensitive spline tensors must be contiguous CUDA float64 vectors"
        );
        TORCH_CHECK(tensor.device() == r.device(), "all inputs must use one CUDA device");
    }
    TORCH_CHECK(parent_index.device() == r.device(), "all inputs must use one CUDA device");
    TORCH_CHECK(
        r_grid.numel() == log_grid.numel() && r_grid64.numel() == r_grid.numel() &&
            log_grid64.numel() == r_grid.numel(),
        "regular and sensitive grids must have equal lengths"
    );
    TORCH_CHECK(
        a.numel() + 1 == r_grid.numel() && b.numel() == a.numel() &&
            c.numel() == a.numel() && d.numel() == a.numel() &&
            a64.numel() == a.numel() && b64.numel() == a.numel() &&
            c64.numel() == a.numel() && d64.numel() == a.numel(),
        "spline coefficients must contain one value per interval"
    );
    K1Parameters64 parameters64{
        r_grid64.data_ptr<double>(),
        log_grid64.data_ptr<double>(),
        a64.data_ptr<double>(),
        b64.data_ptr<double>(),
        c64.data_ptr<double>(),
        d64.data_ptr<double>(),
        static_cast<int>(r_grid64.numel()),
        nc,
        static_cast<double>(nf),
        minimum_r,
        ksub,
        coupling_b0,
        coupling_lambda_squared,
        coupling_scale_numerator,
        coupling_log_mu0_term,
    };
    const int rows = static_cast<int>(r.numel());
    auto mixed_output = torch::zeros(r.sizes(), r.options().dtype(torch::kFloat64));
    auto output = torch::empty_like(r);
    constexpr int threads = fixed_grid_warps_per_block * 32;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    k1_full_grid64_kernel<<<rows, threads, 0, stream>>>(
        r.data_ptr<float>(),
        mixed_output.data_ptr<double>(),
        rows,
        parameters64
    );
    k1_mixed_finalize_kernel<<<(rows + 127) / 128, 128, 0, stream>>>(
        mixed_output.data_ptr<double>(), output.data_ptr<float>(), rows
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<torch::Tensor> radial_integrals(
    torch::Tensor r,
    torch::Tensor r_grid,
    torch::Tensor log_grid,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d,
    int64_t radial_interval_limit,
    int64_t theta_interval_limit,
    double epsrel,
    double nc,
    int64_t nf,
    double minimum_r,
    double ksub,
    double coupling_b0,
    double coupling_lambda_squared,
    double coupling_scale_numerator,
    double coupling_log_mu0_term
) {
    TORCH_CHECK(r.is_cuda(), "r must be on CUDA");
    TORCH_CHECK(r.scalar_type() == torch::kFloat32, "r must be float32");
    TORCH_CHECK(r.dim() == 1 && r.is_contiguous(), "r must be contiguous and one-dimensional");
    TORCH_CHECK(
        radial_interval_limit > 0 && radial_interval_limit <= maximum_intervals,
        "invalid radial interval limit"
    );
    TORCH_CHECK(
        theta_interval_limit > 0 && theta_interval_limit <= maximum_intervals,
        "invalid theta interval limit"
    );
    const std::vector<torch::Tensor> spline_tensors = {r_grid, log_grid, a, b, c, d};
    for (const auto& tensor : spline_tensors) {
        TORCH_CHECK(tensor.is_cuda(), "spline tensors must be on CUDA");
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, "spline tensors must be float32");
        TORCH_CHECK(tensor.dim() == 1 && tensor.is_contiguous(), "spline tensors must be contiguous");
        TORCH_CHECK(tensor.device() == r.device(), "all inputs must use one CUDA device");
    }
    TORCH_CHECK(r_grid.numel() == log_grid.numel(), "r and log grids must have equal lengths");
    TORCH_CHECK(a.numel() + 1 == r_grid.numel(), "invalid spline coefficient length");
    TORCH_CHECK(
        b.numel() == a.numel() && c.numel() == a.numel() && d.numel() == a.numel(),
        "spline coefficients must have equal lengths"
    );

    auto output = torch::empty_like(r);
    auto output_error = torch::empty_like(r);
    auto output_evaluations = torch::empty(r.sizes(), r.options().dtype(torch::kInt64));
    auto output_converged = torch::empty(r.sizes(), r.options().dtype(torch::kBool));
    const int grid_points = static_cast<int>(r_grid.numel());
    int search_steps = 0;
    for (int values = grid_points - 1; values > 0; values >>= 1) {
        ++search_steps;
    }
    K1Parameters parameters{
        r_grid.data_ptr<float>(),
        log_grid.data_ptr<float>(),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        d.data_ptr<float>(),
        grid_points,
        search_steps,
        static_cast<float>(nc),
        static_cast<float>(nf),
        static_cast<float>(minimum_r),
        static_cast<float>(ksub),
        static_cast<float>(coupling_b0),
        static_cast<float>(coupling_lambda_squared),
        static_cast<float>(coupling_scale_numerator),
        static_cast<float>(coupling_log_mu0_term),
    };
    const int rows = static_cast<int>(r.numel());
    auto radial_lower = torch::empty({rows, maximum_intervals}, r.options());
    auto radial_upper = torch::empty({rows, maximum_intervals}, r.options());
    auto radial_estimate = torch::empty({rows, maximum_intervals}, r.options());
    auto radial_error = torch::empty({rows, maximum_intervals}, r.options());
    auto radial_count = torch::empty({rows}, r.options().dtype(torch::kInt32));
    auto radial_split = torch::empty({rows}, r.options().dtype(torch::kInt32));
    auto parent_finished = torch::zeros({rows}, r.options().dtype(torch::kBool));
    auto task_output = torch::empty({rows, radial_tasks_per_parent}, r.options());
    auto task_counter = torch::empty({1}, r.options().dtype(torch::kInt32));
    constexpr int task_threads = task_warps_per_block * 32;
    constexpr int controller_threads = warps_per_block * 32;
    const int all_task_blocks =
        (rows * radial_tasks_per_parent + task_warps_per_block - 1) / task_warps_per_block;
    int multiprocessors = 0;
    C10_CUDA_CHECK(
        cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, r.get_device())
    );
    const int task_blocks = std::min(all_task_blocks, 12 * multiprocessors);
    const int controller_blocks = (rows + warps_per_block - 1) / warps_per_block;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int iteration = 0; iteration < radial_interval_limit; ++iteration) {
        C10_CUDA_CHECK(cudaMemsetAsync(task_counter.data_ptr<int>(), 0, sizeof(int), stream));
        const int work_items = rows * (iteration == 0 ? 21 : radial_tasks_per_parent);
        radial_tasks_kernel<<<task_blocks, task_threads, 0, stream>>>(
            r.data_ptr<float>(),
            radial_lower.data_ptr<float>(),
            radial_upper.data_ptr<float>(),
            radial_split.data_ptr<int>(),
            parent_finished.data_ptr<bool>(),
            task_output.data_ptr<float>(),
            task_counter.data_ptr<int>(),
            work_items,
            rows,
            iteration,
            static_cast<int>(theta_interval_limit),
            static_cast<float>(epsrel),
            parameters
        );
        radial_controller_kernel<<<
            controller_blocks,
            controller_threads,
            0,
            stream>>>(
            task_output.data_ptr<float>(),
            radial_lower.data_ptr<float>(),
            radial_upper.data_ptr<float>(),
            radial_estimate.data_ptr<float>(),
            radial_error.data_ptr<float>(),
            radial_count.data_ptr<int>(),
            radial_split.data_ptr<int>(),
            parent_finished.data_ptr<bool>(),
            output.data_ptr<float>(),
            output_error.data_ptr<float>(),
            output_evaluations.data_ptr<int64_t>(),
            output_converged.data_ptr<bool>(),
            rows,
            iteration,
            static_cast<int>(radial_interval_limit),
            static_cast<float>(epsrel),
            parameters
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, output_error, output_evaluations, output_converged};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("bessel_values", &bessel_values, "K1 J1/I1 CUDA check");
    module.def("theta_integrals", &theta_integrals, "Persistent adaptive K1 theta integrals");
    module.def("fixed_grid_integrals", &fixed_grid_integrals, "Fixed-grid K1 integrals");
    module.def(
        "mixed_fixed_grid_integrals",
        &mixed_fixed_grid_integrals,
        "Full-float64 fixed-grid K1 integral"
    );
    module.def("radial_integrals", &radial_integrals, "Persistent adaptive K1 radial integrals");
}
