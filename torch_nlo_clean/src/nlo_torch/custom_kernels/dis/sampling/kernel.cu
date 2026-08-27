#include <cmath>
#include <limits>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int dimensions = 5;
constexpr int threads = 256;
constexpr float pi = 3.14159265358979323846F;

__global__ void dis_sobol_endpoint_map_kernel(
    const float* __restrict__ points,
    const float* __restrict__ edges,
    float* __restrict__ first,
    float* __restrict__ second,
    float* __restrict__ first_weight,
    float* __restrict__ second_weight,
    int64_t samples,
    int bins,
    float maxr,
    bool fold_angle
) {
    const int64_t sample = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sample >= samples) {
        return;
    }

    float source[dimensions];
    float inverse_density = 1.0F;
#pragma unroll
    for (int dimension = 0; dimension < dimensions; ++dimension) {
        const int64_t offset = sample * dimensions + dimension;
        const float scaled = points[offset] * bins;
        const int bin = min(static_cast<int>(scaled), bins - 1);
        const float fraction = scaled - bin;
        const int edge_offset = dimension * (bins + 1) + bin;
        const float left = edges[edge_offset];
        const float width = edges[edge_offset + 1] - left;
        source[dimension] = left + width * fraction;
        inverse_density *= bins * width;
        first[offset] = source[dimension];
        second[offset] = source[dimension];
    }

    const float parent = maxr * source[2];
    const float radial_fraction = source[3];
    const float angle_fraction = fold_angle ? 0.5F * source[4] : source[4];
    const float angle = 2.0F * pi * angle_fraction;
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const float closer_boundary = cosine > 0.0F
        ? parent / (2.0F * cosine)
        : INFINITY;

    const float first_limit = fminf(maxr, closer_boundary);
    first[sample * dimensions + 3] = radial_fraction * first_limit / maxr;
    first[sample * dimensions + 4] = angle_fraction;
    first_weight[sample] = inverse_density * first_limit / maxr;

    const float disk_boundary = parent * cosine + sqrtf(fmaxf(
        maxr * maxr - parent * parent * sine * sine,
        0.0F
    ));
    const float second_limit = fmaxf(fminf(disk_boundary, closer_boundary), 0.0F);
    const float second_radius = radial_fraction * second_limit;
    const float global_x = parent - second_radius * cosine;
    const float global_y = second_radius * sine;
    const float global_radius = sqrtf(global_x * global_x + global_y * global_y);
    float global_angle = atan2f(global_y, global_x);
    if (global_angle < 0.0F) {
        global_angle += 2.0F * pi;
    }
    second[sample * dimensions + 3] = global_radius / maxr;
    second[sample * dimensions + 4] = global_angle / (2.0F * pi);
    second_weight[sample] = inverse_density * second_radius * second_limit /
        (fmaxf(global_radius, std::numeric_limits<float>::min()) * maxr);
}

std::vector<torch::Tensor> dis_sobol_endpoint_map(
    torch::Tensor points,
    torch::Tensor edges,
    double maxr,
    bool fold_angle
) {
    TORCH_CHECK(
        points.is_cuda() && points.scalar_type() == torch::kFloat32 &&
            points.dim() == 2 && points.size(1) == dimensions && points.is_contiguous(),
        "DIS Sobol points must be contiguous CUDA float32 [samples, 5]"
    );
    TORCH_CHECK(
        edges.is_cuda() && edges.scalar_type() == torch::kFloat32 &&
            edges.dim() == 2 && edges.size(0) == dimensions && edges.is_contiguous() &&
            edges.get_device() == points.get_device(),
        "DIS Sobol edges must be contiguous CUDA float32 [5, bins + 1]"
    );
    TORCH_CHECK(edges.size(1) >= 3, "DIS Sobol importance grids require at least two bins");
    TORCH_CHECK(maxr > 0.0 && std::isfinite(maxr), "DIS Sobol maxr must be positive and finite");

    auto first = torch::empty_like(points);
    auto second = torch::empty_like(points);
    auto first_weight = torch::empty({points.size(0)}, points.options());
    auto second_weight = torch::empty({points.size(0)}, points.options());
    if (points.size(0) == 0) {
        return {first, first_weight, second, second_weight};
    }

    const int blocks = (points.size(0) + threads - 1) / threads;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dis_sobol_endpoint_map_kernel<<<blocks, threads, 0, stream>>>(
        points.data_ptr<float>(),
        edges.data_ptr<float>(),
        first.data_ptr<float>(),
        second.data_ptr<float>(),
        first_weight.data_ptr<float>(),
        second_weight.data_ptr<float>(),
        points.size(0),
        edges.size(1) - 1,
        static_cast<float>(maxr),
        fold_angle
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {first, first_weight, second, second_weight};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "dis_sobol_endpoint_map",
        &dis_sobol_endpoint_map,
        "Map a frozen DIS Sobol grid into paired endpoint sectors"
    );
}
