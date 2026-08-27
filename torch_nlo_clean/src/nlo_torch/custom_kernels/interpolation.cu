#include <vector>

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

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

__device__ __forceinline__ float interpolation_value(
    const float* amplitude,
    int index,
    int row,
    bool force_positive
) {
    const float value = amplitude[index];
    if (row == 0) {
        float interpolation = value > 1.0f ? 1.0f : value;
        if (force_positive && interpolation < 0.0f) {
            interpolation = 0.0f;
        }
        return interpolation;
    }
    float interpolation = subtract_rn(1.0f, value);
    interpolation = interpolation < 0.0f ? 0.0f : interpolation;
    if (force_positive && interpolation > 1.0f) {
        interpolation = 1.0f;
    }
    return interpolation;
}

__device__ __forceinline__ float interval_width(const float* log_grid, int index) {
    return subtract_rn(log_grid[index + 1], log_grid[index]);
}

__device__ __forceinline__ float diagonal_value(const float* log_grid, int index) {
    return multiply_rn(
        2.0f,
        add_rn(interval_width(log_grid, index), interval_width(log_grid, index + 1))
    );
}

__device__ __forceinline__ float right_hand_side_value(
    const float* log_grid,
    const float* log_amplitude,
    int index,
    int amplitude_offset
) {
    const float right_slope = divide_rn(
        subtract_rn(
            log_amplitude[amplitude_offset + index + 2],
            log_amplitude[amplitude_offset + index + 1]
        ),
        interval_width(log_grid, index + 1)
    );
    const float left_slope = divide_rn(
        subtract_rn(
            log_amplitude[amplitude_offset + index + 1],
            log_amplitude[amplitude_offset + index]
        ),
        interval_width(log_grid, index)
    );
    return multiply_rn(6.0f, subtract_rn(right_slope, left_slope));
}

__global__ void evolution_loglog_splines_kernel(
    const float* grid,
    const float* amplitude,
    float* log_grid,
    float* log_amplitude,
    float* coefficient_a,
    float* coefficient_b,
    float* coefficient_c,
    float* coefficient_d,
    float* upper,
    float* solution,
    int grid_points,
    bool force_positive
) {
    const int interior_points = grid_points - 2;
    const int intervals = grid_points - 1;
    for (int index = threadIdx.x; index < grid_points; index += blockDim.x) {
        log_grid[index] = logf(grid[index]);
        for (int row = 0; row < 2; ++row) {
            const float value = interpolation_value(amplitude, index, row, force_positive);
            log_amplitude[row * grid_points + index] =
                value > 0.0f ? logf(value) : -40.0f;
        }
    }
    __syncthreads();

    const int row = threadIdx.x;
    if (row < 2) {
        const int amplitude_offset = row * grid_points;
        const int interior_offset = row * interior_points;
        if (interior_points == 1) {
            solution[interior_offset] = divide_rn(
                right_hand_side_value(log_grid, log_amplitude, 0, amplitude_offset),
                diagonal_value(log_grid, 0)
            );
        } else {
            const float first_diagonal = diagonal_value(log_grid, 0);
            upper[interior_offset] =
                divide_rn(interval_width(log_grid, 1), first_diagonal);
            solution[interior_offset] = divide_rn(
                right_hand_side_value(log_grid, log_amplitude, 0, amplitude_offset),
                first_diagonal
            );
            for (int index = 1; index < interior_points; ++index) {
                const float off_diagonal = interval_width(log_grid, index);
                const float denominator = subtract_rn(
                    diagonal_value(log_grid, index),
                    multiply_rn(off_diagonal, upper[interior_offset + index - 1])
                );
                if (index < interior_points - 1) {
                    upper[interior_offset + index] =
                        divide_rn(interval_width(log_grid, index + 1), denominator);
                }
                solution[interior_offset + index] = divide_rn(
                    subtract_rn(
                        right_hand_side_value(
                            log_grid,
                            log_amplitude,
                            index,
                            amplitude_offset
                        ),
                        multiply_rn(
                            off_diagonal,
                            solution[interior_offset + index - 1]
                        )
                    ),
                    denominator
                );
            }
            for (int index = interior_points - 2; index >= 0; --index) {
                solution[interior_offset + index] = subtract_rn(
                    solution[interior_offset + index],
                    multiply_rn(
                        upper[interior_offset + index],
                        solution[interior_offset + index + 1]
                    )
                );
            }
        }
    }
    __syncthreads();

    for (int item = threadIdx.x; item < 2 * intervals; item += blockDim.x) {
        const int coefficient_row = item / intervals;
        const int index = item - coefficient_row * intervals;
        const int amplitude_offset = coefficient_row * grid_points;
        const int interior_offset = coefficient_row * interior_points;
        const int coefficient_offset = coefficient_row * intervals;
        const float width = interval_width(log_grid, index);
        const float y_left = log_amplitude[amplitude_offset + index];
        const float y_right = log_amplitude[amplitude_offset + index + 1];
        const float second_left = index == 0 ? 0.0f : solution[interior_offset + index - 1];
        const float second_right =
            index == intervals - 1 ? 0.0f : solution[interior_offset + index];
        coefficient_a[coefficient_offset + index] = y_left;
        coefficient_b[coefficient_offset + index] = subtract_rn(
            divide_rn(subtract_rn(y_right, y_left), width),
            multiply_rn(
                multiply_rn(
                    width,
                    add_rn(multiply_rn(2.0f, second_left), second_right)
                ),
                1.0f / 6.0f
            )
        );
        coefficient_c[coefficient_offset + index] = divide_rn(second_left, 2.0f);
        coefficient_d[coefficient_offset + index] = divide_rn(
            subtract_rn(second_right, second_left),
            multiply_rn(6.0f, width)
        );
    }
}

__device__ __forceinline__ double interpolation_value64(
    const float* amplitude,
    int index,
    int row,
    bool force_positive
) {
    const double value = static_cast<double>(amplitude[index]);
    if (row == 0) {
        double interpolation = value > 1.0 ? 1.0 : value;
        if (force_positive && interpolation < 0.0) {
            interpolation = 0.0;
        }
        return interpolation;
    }
    double interpolation = 1.0 - value;
    interpolation = interpolation < 0.0 ? 0.0 : interpolation;
    if (force_positive && interpolation > 1.0) {
        interpolation = 1.0;
    }
    return interpolation;
}

__device__ __forceinline__ double interval_width64(const double* log_grid, int index) {
    return log_grid[index + 1] - log_grid[index];
}

__device__ __forceinline__ double diagonal_value64(const double* log_grid, int index) {
    return 2.0 * (interval_width64(log_grid, index) + interval_width64(log_grid, index + 1));
}

__device__ __forceinline__ double right_hand_side_value64(
    const double* log_grid,
    const double* log_amplitude,
    int index,
    int amplitude_offset
) {
    const double right_slope =
        (log_amplitude[amplitude_offset + index + 2] -
         log_amplitude[amplitude_offset + index + 1]) /
        interval_width64(log_grid, index + 1);
    const double left_slope =
        (log_amplitude[amplitude_offset + index + 1] -
         log_amplitude[amplitude_offset + index]) /
        interval_width64(log_grid, index);
    return 6.0 * (right_slope - left_slope);
}

__global__ void evolution_loglog_splines64_kernel(
    const float* grid,
    const float* amplitude,
    double* log_grid,
    double* log_amplitude,
    double* coefficient_a,
    double* coefficient_b,
    double* coefficient_c,
    double* coefficient_d,
    double* upper,
    double* solution,
    int grid_points,
    bool force_positive
) {
    const int interior_points = grid_points - 2;
    const int intervals = grid_points - 1;
    for (int index = threadIdx.x; index < grid_points; index += blockDim.x) {
        log_grid[index] = log(static_cast<double>(grid[index]));
        for (int row = 0; row < 2; ++row) {
            const double value = interpolation_value64(amplitude, index, row, force_positive);
            log_amplitude[row * grid_points + index] = value > 0.0 ? log(value) : -40.0;
        }
    }
    __syncthreads();

    const int row = threadIdx.x;
    if (row < 2) {
        const int amplitude_offset = row * grid_points;
        const int interior_offset = row * interior_points;
        if (interior_points == 1) {
            solution[interior_offset] = right_hand_side_value64(
                log_grid, log_amplitude, 0, amplitude_offset
            ) / diagonal_value64(log_grid, 0);
        } else {
            const double first_diagonal = diagonal_value64(log_grid, 0);
            upper[interior_offset] = interval_width64(log_grid, 1) / first_diagonal;
            solution[interior_offset] = right_hand_side_value64(
                log_grid, log_amplitude, 0, amplitude_offset
            ) / first_diagonal;
            for (int index = 1; index < interior_points; ++index) {
                const double off_diagonal = interval_width64(log_grid, index);
                const double denominator = diagonal_value64(log_grid, index) -
                    off_diagonal * upper[interior_offset + index - 1];
                if (index < interior_points - 1) {
                    upper[interior_offset + index] =
                        interval_width64(log_grid, index + 1) / denominator;
                }
                solution[interior_offset + index] =
                    (right_hand_side_value64(
                         log_grid, log_amplitude, index, amplitude_offset
                     ) -
                     off_diagonal * solution[interior_offset + index - 1]) /
                    denominator;
            }
            for (int index = interior_points - 2; index >= 0; --index) {
                solution[interior_offset + index] -=
                    upper[interior_offset + index] * solution[interior_offset + index + 1];
            }
        }
    }
    __syncthreads();

    for (int item = threadIdx.x; item < 2 * intervals; item += blockDim.x) {
        const int coefficient_row = item / intervals;
        const int index = item - coefficient_row * intervals;
        const int amplitude_offset = coefficient_row * grid_points;
        const int interior_offset = coefficient_row * interior_points;
        const int coefficient_offset = coefficient_row * intervals;
        const double width = interval_width64(log_grid, index);
        const double y_left = log_amplitude[amplitude_offset + index];
        const double y_right = log_amplitude[amplitude_offset + index + 1];
        const double second_left =
            index == 0 ? 0.0 : solution[interior_offset + index - 1];
        const double second_right =
            index == intervals - 1 ? 0.0 : solution[interior_offset + index];
        coefficient_a[coefficient_offset + index] = y_left;
        coefficient_b[coefficient_offset + index] =
            (y_right - y_left) / width - width * (2.0 * second_left + second_right) / 6.0;
        coefficient_c[coefficient_offset + index] = second_left / 2.0;
        coefficient_d[coefficient_offset + index] =
            (second_right - second_left) / (6.0 * width);
    }
}

}  // namespace

std::vector<torch::Tensor> evolution_loglog_splines(
    torch::Tensor grid,
    torch::Tensor amplitude,
    bool force_positive
) {
    TORCH_CHECK(grid.is_cuda(), "grid must be on CUDA");
    TORCH_CHECK(grid.scalar_type() == torch::kFloat32, "grid must be float32");
    TORCH_CHECK(grid.dim() == 1 && grid.is_contiguous(), "grid must be contiguous and one-dimensional");
    TORCH_CHECK(grid.numel() >= 3, "a cubic spline requires at least three points");
    TORCH_CHECK(amplitude.is_cuda(), "amplitude must be on CUDA");
    TORCH_CHECK(amplitude.scalar_type() == torch::kFloat32, "amplitude must be float32");
    TORCH_CHECK(
        amplitude.dim() == 1 && amplitude.is_contiguous() && amplitude.sizes() == grid.sizes(),
        "amplitude must be contiguous and match the grid"
    );
    TORCH_CHECK(grid.device() == amplitude.device(), "inputs must use one CUDA device");

    const c10::cuda::CUDAGuard device_guard(grid.device());
    const int grid_points = static_cast<int>(grid.numel());
    const int interior_points = grid_points - 2;
    const int intervals = grid_points - 1;
    auto log_grid = torch::empty_like(grid);
    auto log_amplitude = torch::empty({2, grid_points}, grid.options());
    auto coefficients = torch::empty({4, 2, intervals}, grid.options());
    auto upper = torch::empty({2, interior_points}, grid.options());
    auto solution = torch::empty_like(upper);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    evolution_loglog_splines_kernel<<<1, 128, 0, stream>>>(
        grid.data_ptr<float>(),
        amplitude.data_ptr<float>(),
        log_grid.data_ptr<float>(),
        log_amplitude.data_ptr<float>(),
        coefficients[0].data_ptr<float>(),
        coefficients[1].data_ptr<float>(),
        coefficients[2].data_ptr<float>(),
        coefficients[3].data_ptr<float>(),
        upper.data_ptr<float>(),
        solution.data_ptr<float>(),
        grid_points,
        force_positive
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {log_grid, coefficients[0], coefficients[1], coefficients[2], coefficients[3]};
}

std::vector<torch::Tensor> evolution_loglog_splines_mixed(
    torch::Tensor grid,
    torch::Tensor amplitude,
    bool force_positive
) {
    TORCH_CHECK(grid.is_cuda(), "grid must be on CUDA");
    TORCH_CHECK(grid.scalar_type() == torch::kFloat32, "grid must be float32");
    TORCH_CHECK(
        grid.dim() == 1 && grid.is_contiguous(),
        "grid must be contiguous and one-dimensional"
    );
    TORCH_CHECK(grid.numel() >= 3, "a cubic spline requires at least three points");
    TORCH_CHECK(amplitude.is_cuda(), "amplitude must be on CUDA");
    TORCH_CHECK(amplitude.scalar_type() == torch::kFloat32, "amplitude must be float32");
    TORCH_CHECK(
        amplitude.dim() == 1 && amplitude.is_contiguous() && amplitude.sizes() == grid.sizes(),
        "amplitude must be contiguous and match the grid"
    );
    TORCH_CHECK(grid.device() == amplitude.device(), "inputs must use one CUDA device");

    const c10::cuda::CUDAGuard device_guard(grid.device());
    const int grid_points = static_cast<int>(grid.numel());
    const int interior_points = grid_points - 2;
    const int intervals = grid_points - 1;
    auto log_grid32 = torch::empty_like(grid);
    auto log_amplitude32 = torch::empty({2, grid_points}, grid.options());
    auto coefficients32 = torch::empty({4, 2, intervals}, grid.options());
    auto upper32 = torch::empty({2, interior_points}, grid.options());
    auto solution32 = torch::empty_like(upper32);
    auto double_options = grid.options().dtype(torch::kFloat64);
    auto log_grid64 = torch::empty(grid.sizes(), double_options);
    auto log_amplitude64 = torch::empty({2, grid_points}, double_options);
    auto coefficients64 = torch::empty({4, 2, intervals}, double_options);
    auto upper64 = torch::empty({2, interior_points}, double_options);
    auto solution64 = torch::empty_like(upper64);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    evolution_loglog_splines_kernel<<<1, 128, 0, stream>>>(
        grid.data_ptr<float>(),
        amplitude.data_ptr<float>(),
        log_grid32.data_ptr<float>(),
        log_amplitude32.data_ptr<float>(),
        coefficients32[0].data_ptr<float>(),
        coefficients32[1].data_ptr<float>(),
        coefficients32[2].data_ptr<float>(),
        coefficients32[3].data_ptr<float>(),
        upper32.data_ptr<float>(),
        solution32.data_ptr<float>(),
        grid_points,
        force_positive
    );
    evolution_loglog_splines64_kernel<<<1, 128, 0, stream>>>(
        grid.data_ptr<float>(),
        amplitude.data_ptr<float>(),
        log_grid64.data_ptr<double>(),
        log_amplitude64.data_ptr<double>(),
        coefficients64[0].data_ptr<double>(),
        coefficients64[1].data_ptr<double>(),
        coefficients64[2].data_ptr<double>(),
        coefficients64[3].data_ptr<double>(),
        upper64.data_ptr<double>(),
        solution64.data_ptr<double>(),
        grid_points,
        force_positive
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        log_grid32,
        coefficients32[0],
        coefficients32[1],
        coefficients32[2],
        coefficients32[3],
        log_grid64,
        coefficients64[0],
        coefficients64[1],
        coefficients64[2],
        coefficients64[3],
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "evolution_loglog_splines",
        &evolution_loglog_splines,
        "Build the N and S log-log spline coefficients"
    );
    module.def(
        "evolution_loglog_splines_mixed",
        &evolution_loglog_splines_mixed,
        "Build float32 and float64 N and S log-log spline coefficients"
    );
}
