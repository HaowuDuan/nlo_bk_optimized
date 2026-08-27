#include <algorithm>
#include <functional>
#include <limits>
#include <mutex>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <curand_kernel.h>
#include <torch/extension.h>

namespace {

constexpr int dimensions = 4;
constexpr int bins = 32;
constexpr int block_width = 8;
constexpr int block_height = 16;
constexpr int values_per_load = 4;
constexpr int moment_threads = 512;
constexpr int adaptation_threads = 128;
constexpr int adaptation_items_per_thread = 15;
constexpr int random_threads = 256;
constexpr int random_values_per_thread = 4;

__device__ __forceinline__ float multiply_rn(float left, float right) {
    return __fmul_rn(left, right);
}

__device__ __forceinline__ float add_rn(float left, float right) {
    return __fadd_rn(left, right);
}

__device__ __forceinline__ float subtract_rn(float left, float right) {
    return __fsub_rn(left, right);
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

C10_LAUNCH_BOUNDS_2(random_threads, 4)
__global__ void vegas_random_kernel(
    int64_t* __restrict__ bin_index,
    float* __restrict__ random,
    int64_t values,
    at::PhiloxCudaState integer_state,
    at::PhiloxCudaState uniform_state
) {
    const auto [integer_seed, integer_offset] = at::cuda::philox::unpack(integer_state);
    const auto [uniform_seed, uniform_offset] = at::cuda::philox::unpack(uniform_state);
    const int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t integer_generator;
    curandStatePhilox4_32_10_t uniform_generator;
    curand_init(integer_seed, thread, integer_offset, &integer_generator);
    curand_init(uniform_seed, thread, uniform_offset, &uniform_generator);

    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t rounded_values =
        ((values - 1) / (stride * random_values_per_thread) + 1) *
        stride * random_values_per_thread;
    for (int64_t linear = thread; linear < rounded_values;
         linear += stride * random_values_per_thread) {
        const uint4 integer_values = curand4(&integer_generator);
        const float4 uniform_values = curand_uniform4(&uniform_generator);
#pragma unroll
        for (int item = 0; item < random_values_per_thread; ++item) {
            const int64_t output = linear + stride * item;
            if (output < values) {
                bin_index[output] = static_cast<int64_t>((&integer_values.x)[item] % bins);
                const float value = (&uniform_values.x)[item];
                random[output] = value == 1.0F ? 0.0F : value;
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ float vegas_weight(
    const float* __restrict__ value,
    const float* __restrict__ width,
    const float* __restrict__ volume,
    int sample,
    int64_t sample_stride,
    int64_t dimension_stride
) {
    const int64_t width_base = static_cast<int64_t>(sample) * sample_stride;
    const float width_0 = multiply_rn(32.0F, width[width_base]);
    const float width_1 = multiply_rn(32.0F, width[width_base + dimension_stride]);
    const float width_2 = multiply_rn(32.0F, width[width_base + 2 * dimension_stride]);
    const float width_3 = multiply_rn(32.0F, width[width_base + 3 * dimension_stride]);
    const float inverse_density = multiply_rn(
        multiply_rn(multiply_rn(width_0, width_1), width_2), width_3
    );
    return multiply_rn(
        multiply_rn(value[sample], volume[0]), inverse_density
    );
}

template <int channels>
__device__ __forceinline__ void reduce_moments(float (&value)[channels], float* shared) {
    const int shared_base = threadIdx.x * channels;
#pragma unroll
    for (int channel = 0; channel < channels; ++channel) {
        shared[shared_base + channel] = value[channel];
    }

    for (int offset = moment_threads / 2; offset >= 32; offset >>= 1) {
        __syncthreads();
        if (threadIdx.x < offset) {
#pragma unroll
            for (int channel = 0; channel < channels; ++channel) {
                value[channel] = add_rn(
                    value[channel], shared[(threadIdx.x + offset) * channels + channel]
                );
                shared[shared_base + channel] = value[channel];
            }
        }
    }

    __syncthreads();
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int channel = 0; channel < channels; ++channel) {
            value[channel] = add_rn(
                value[channel], __shfl_down_sync(0xffffffffU, value[channel], offset)
            );
        }
    }
}

__global__ void vegas_weighted_moments_kernel(
    const float* __restrict__ value,
    const float* __restrict__ width,
    const float* __restrict__ volume,
    float* __restrict__ total,
    float* __restrict__ total_square,
    float* __restrict__ absolute_weight,
    int samples,
    int64_t sample_stride,
    int64_t dimension_stride,
    bool initialize
) {
    const int channel = blockIdx.x;
    float moments[values_per_load] = {};
    int vector_index = threadIdx.x;
    while (vector_index * values_per_load + values_per_load - 1 < samples) {
#pragma unroll
        for (int lane = 0; lane < values_per_load; ++lane) {
            const int sample = vector_index * values_per_load + lane;
            const float weighted = vegas_weight(
                value, width, volume, sample, sample_stride, dimension_stride
            );
            if (channel == 0) {
                absolute_weight[sample] = fabsf(weighted);
            }
            const float moment = channel == 0 ? weighted : multiply_rn(weighted, weighted);
            moments[lane] = add_rn(
                moments[lane], moment
            );
        }
        vector_index += moment_threads;
    }

    const int tail_start = samples - samples % values_per_load;
    const int tail_sample = tail_start + threadIdx.x;
    if (tail_sample < samples) {
        const float weighted = vegas_weight(
            value, width, volume, tail_sample, sample_stride, dimension_stride
        );
        if (channel == 0) {
            absolute_weight[tail_sample] = fabsf(weighted);
        }
        const float moment = channel == 0 ? weighted : multiply_rn(weighted, weighted);
        moments[0] = add_rn(moments[0], moment);
    }

    float reduced[1] = {moments[0]};
#pragma unroll
    for (int lane = 1; lane < values_per_load; ++lane) {
        reduced[0] = add_rn(reduced[0], moments[lane]);
    }

    extern __shared__ float shared[];
    reduce_moments(reduced, shared);
    if (threadIdx.x == 0) {
        float* output = channel == 0 ? total : total_square;
        output[0] = add_rn(initialize ? 0.0F : output[0], reduced[0]);
    }
}

__global__ void vegas_weighted_value_moments_kernel(
    const float* __restrict__ weighted_value,
    float* __restrict__ total,
    float* __restrict__ total_square,
    float* __restrict__ absolute_weight,
    int samples,
    bool initialize
) {
    const int channel = blockIdx.x;
    float moments[values_per_load] = {};
    int vector_index = threadIdx.x;
    while (vector_index * values_per_load + values_per_load - 1 < samples) {
#pragma unroll
        for (int lane = 0; lane < values_per_load; ++lane) {
            const int sample = vector_index * values_per_load + lane;
            const float weighted = weighted_value[sample];
            if (channel == 0) {
                absolute_weight[sample] = fabsf(weighted);
            }
            const float moment = channel == 0 ? weighted : multiply_rn(weighted, weighted);
            moments[lane] = add_rn(moments[lane], moment);
        }
        vector_index += moment_threads;
    }

    const int tail_start = samples - samples % values_per_load;
    const int tail_sample = tail_start + threadIdx.x;
    if (tail_sample < samples) {
        const float weighted = weighted_value[tail_sample];
        if (channel == 0) {
            absolute_weight[tail_sample] = fabsf(weighted);
        }
        const float moment = channel == 0 ? weighted : multiply_rn(weighted, weighted);
        moments[0] = add_rn(moments[0], moment);
    }

    float reduced[1] = {moments[0]};
#pragma unroll
    for (int lane = 1; lane < values_per_load; ++lane) {
        reduced[0] = add_rn(reduced[0], moments[lane]);
    }

    extern __shared__ float shared[];
    reduce_moments(reduced, shared);
    if (threadIdx.x == 0) {
        float* output = channel == 0 ? total : total_square;
        output[0] = add_rn(initialize ? 0.0F : output[0], reduced[0]);
    }
}

__global__ void vegas_adapt_edges_kernel(
    const float* __restrict__ edges,
    const float* __restrict__ histogram,
    const float* __restrict__ bin_count,
    float* __restrict__ output
) {
    const int dimension = blockIdx.x;
    const int row = dimension * bins;
    __shared__ float smoothed_values[bins];
    if (threadIdx.x < bins) {
        const int bin = threadIdx.x;
        const int left_bin = max(bin - 1, 0);
        const int right_bin = min(bin + 1, bins - 1);
        const float left_importance = divide_rn(
            histogram[row + left_bin], fmaxf(bin_count[row + left_bin], 1.0F)
        );
        const float center_importance = divide_rn(
            histogram[row + bin], fmaxf(bin_count[row + bin], 1.0F)
        );
        const float right_importance = divide_rn(
            histogram[row + right_bin], fmaxf(bin_count[row + right_bin], 1.0F)
        );
        const float interior = divide_full(
            add_rn(add_rn(left_importance, center_importance), right_importance), 3.0F
        );
        const float first = divide_full(add_rn(center_importance, right_importance), 2.0F);
        const float last = divide_full(add_rn(left_importance, center_importance), 2.0F);
        float smoothed = bin == 0 ? first : interior;
        smoothed = bin == bins - 1 ? last : smoothed;
        smoothed_values[bin] = add_rn(smoothed, 1.1920928955078125e-7F);
    }
    __syncthreads();

    using AdaptationLoad = cub::BlockLoad<
        float,
        adaptation_threads,
        adaptation_items_per_thread,
        cub::BLOCK_LOAD_WARP_TRANSPOSE
    >;
    using AdaptationScan = cub::BlockScan<
        float,
        adaptation_threads,
        cub::BLOCK_SCAN_WARP_SCANS
    >;
    union AdaptationStorage {
        typename AdaptationLoad::TempStorage load;
        typename AdaptationScan::TempStorage scan;
    };
    __shared__ AdaptationStorage scan_storage;
    float cumulative[adaptation_items_per_thread];
    AdaptationLoad(scan_storage.load).Load(
        smoothed_values,
        cumulative,
        bins,
        smoothed_values[0]
    );
    __syncthreads();
    float block_aggregate;
    AdaptationScan(scan_storage.scan).InclusiveScan(
        cumulative,
        cumulative,
        std::plus<float>{},
        block_aggregate
    );

    __shared__ float cumulative_values[bins + 1];
    if (threadIdx.x == 0) {
        cumulative_values[0] = 0.0F;
    }
#pragma unroll
    for (int item = 0; item < adaptation_items_per_thread; ++item) {
        const int position = threadIdx.x * adaptation_items_per_thread + item;
        if (position < bins) {
            cumulative_values[position + 1] = cumulative[item];
        }
    }
    __syncthreads();

    const int position = threadIdx.x;
    if (position > bins) {
        return;
    }
    const int edge_row = dimension * (bins + 1);
    if (position == bins) {
        output[edge_row + position] = 1.0F;
        return;
    }
    const float total = cumulative_values[bins];
    const float step = divide_rn(total, static_cast<float>(bins));
    const float from_start = multiply_rn(step, static_cast<float>(position));
    const float from_end = __fmaf_rn(
        -step, static_cast<float>(bins - position), total
    );
    const float target = position < (bins + 1) / 2 ? from_start : from_end;

    int lower = 0;
    int upper = bins + 1;
#pragma unroll
    for (int search = 0; search < 6; ++search) {
        const int middle = (lower + upper) / 2;
        if (cumulative_values[middle] <= target) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    const int interval = max(0, min(bins - 1, lower - 1));
    const float fraction = divide_rn(
        subtract_rn(target, cumulative_values[interval]),
        smoothed_values[interval]
    );
    const float left_edge = edges[edge_row + interval];
    const float right_edge = edges[edge_row + interval + 1];
    float updated = add_rn(
        left_edge, multiply_rn(fraction, subtract_rn(right_edge, left_edge))
    );
    updated = position == 0 ? 0.0F : updated;
    output[edge_row + position] = updated;
}

__global__ void vegas_estimate_variance_kernel(
    const float* __restrict__ total,
    const float* __restrict__ total_square,
    float* __restrict__ estimates,
    float* __restrict__ variances,
    int output_index,
    float inverse_samples,
    float inverse_sample_degrees,
    float minimum_variance
) {
    const float estimate = multiply_rn(total[0], inverse_samples);
    const float centered_square = subtract_rn(
        total_square[0], multiply_rn(multiply_rn(total[0], total[0]), inverse_samples)
    );
    const float sample_variance = multiply_rn(
        fmaxf(centered_square, 0.0F), inverse_sample_degrees
    );
    const float variance = fmaxf(
        multiply_rn(sample_variance, inverse_samples), minimum_variance
    );
    estimates[output_index] = estimate;
    variances[output_index] = variance;
}

__global__ void vegas_combine_estimates_kernel(
    const float* __restrict__ estimates,
    const float* __restrict__ variances,
    float* __restrict__ output,
    int iterations
) {
    float weights[4] = {};
    float weighted_estimates[4] = {};
    for (int iteration = 0; iteration < iterations; ++iteration) {
        weights[iteration] = reciprocal_rn(variances[iteration]);
        weighted_estimates[iteration] = multiply_rn(
            weights[iteration], estimates[iteration]
        );
    }
    const float total_weight = add_rn(
        add_rn(weights[0], weights[2]), add_rn(weights[1], weights[3])
    );
    const float weighted_total = add_rn(
        add_rn(weighted_estimates[0], weighted_estimates[2]),
        add_rn(weighted_estimates[1], weighted_estimates[3])
    );
    output[0] = divide_rn(weighted_total, total_weight);
    output[1] = __fsqrt_rn(reciprocal_rn(total_weight));
}

__global__ void vegas_store_and_combine_kernel(
    const float* __restrict__ total,
    const float* __restrict__ total_square,
    float* __restrict__ estimates,
    float* __restrict__ variances,
    float* __restrict__ output,
    int output_index,
    float inverse_samples,
    float inverse_sample_degrees,
    float minimum_variance
) {
    const float estimate = multiply_rn(total[0], inverse_samples);
    const float centered_square = subtract_rn(
        total_square[0], multiply_rn(multiply_rn(total[0], total[0]), inverse_samples)
    );
    const float sample_variance = multiply_rn(
        fmaxf(centered_square, 0.0F), inverse_sample_degrees
    );
    const float variance = fmaxf(
        multiply_rn(sample_variance, inverse_samples), minimum_variance
    );
    estimates[output_index] = estimate;
    variances[output_index] = variance;

    float weights[4] = {};
    float weighted_estimates[4] = {};
    for (int iteration = 0; iteration <= output_index; ++iteration) {
        weights[iteration] = reciprocal_rn(variances[iteration]);
        weighted_estimates[iteration] = multiply_rn(
            weights[iteration], estimates[iteration]
        );
    }
    const float total_weight = add_rn(
        add_rn(weights[0], weights[2]), add_rn(weights[1], weights[3])
    );
    const float weighted_total = add_rn(
        add_rn(weighted_estimates[0], weighted_estimates[2]),
        add_rn(weighted_estimates[1], weighted_estimates[3])
    );
    output[0] = divide_rn(weighted_total, total_weight);
    output[1] = __fsqrt_rn(reciprocal_rn(total_weight));
}

__global__ void vegas_initialize_kernel(
    const float* __restrict__ bounds,
    float* __restrict__ edges,
    float* __restrict__ bounds_lower,
    float* __restrict__ bounds_width,
    float* __restrict__ volume,
    int* __restrict__ semaphores
) {
    for (int edge = threadIdx.x; edge < dimensions * (bins + 1); edge += blockDim.x) {
        edges[edge] = multiply_rn(static_cast<float>(edge % (bins + 1)), 1.0F / bins);
    }
    if (threadIdx.x < dimensions) {
        const int dimension = threadIdx.x;
        const float lower = bounds[dimension * 2];
        bounds_lower[dimension] = lower;
        bounds_width[dimension] = subtract_rn(bounds[dimension * 2 + 1], lower);
    }
    if (threadIdx.x < dimensions + 1) {
        semaphores[threadIdx.x] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        volume[0] = multiply_rn(
            multiply_rn(bounds_width[0], bounds_width[2]),
            multiply_rn(bounds_width[1], bounds_width[3])
        );
    }
}

template <int channels>
__device__ __forceinline__ void reduce_block_y(
    float (&value)[channels][values_per_load],
    float* shared
) {
    const int shared_base =
        ((threadIdx.y * block_width + threadIdx.x) * channels) * values_per_load;
#pragma unroll
    for (int channel = 0; channel < channels; ++channel) {
#pragma unroll
        for (int lane = 0; lane < values_per_load; ++lane) {
            shared[shared_base + channel * values_per_load + lane] = value[channel][lane];
        }
    }

    for (int offset = block_height / 2; offset > 0; offset >>= 1) {
        __syncthreads();
        if (threadIdx.y < offset) {
            const int other_base = shared_base + offset * block_width * channels * values_per_load;
#pragma unroll
            for (int channel = 0; channel < channels; ++channel) {
#pragma unroll
                for (int lane = 0; lane < values_per_load; ++lane) {
                    value[channel][lane] = add_rn(
                        value[channel][lane],
                        shared[other_base + channel * values_per_load + lane]
                    );
                    shared[shared_base + channel * values_per_load + lane] = value[channel][lane];
                }
            }
        }
    }
}

template <bool accumulate_weighted_moments>
__global__ void vegas_histogram_kernel(
    const int64_t* __restrict__ bin_index,
    const float* __restrict__ sample_weight,
    float* __restrict__ histogram,
    float* __restrict__ bin_count,
    float* __restrict__ partial_histogram,
    float* __restrict__ partial_count,
    float* __restrict__ total,
    float* __restrict__ total_square,
    float* __restrict__ partial_moments,
    int* __restrict__ semaphores,
    int samples,
    int ctas_per_dimension,
    bool initialize
) {
    extern __shared__ float shared[];
    const int dimension = blockIdx.z;
    const int linear_thread = threadIdx.y * block_width + threadIdx.x;

    if constexpr (accumulate_weighted_moments) {
        if (blockIdx.y == 0) {
            // The first block of each dimension evaluates one quarter of the old
            // 512-thread moment reduction. The last quarter combines them in the
            // same pairwise order, so fusing the launches does not change rounding.
            const int virtual_thread = dimension * block_width * block_height + linear_thread;
            float moments[2][values_per_load] = {};
            int vector_index = virtual_thread;
            while (vector_index * values_per_load + values_per_load - 1 < samples) {
#pragma unroll
                for (int lane = 0; lane < values_per_load; ++lane) {
                    const float weighted = sample_weight[vector_index * values_per_load + lane];
                    moments[0][lane] = add_rn(moments[0][lane], weighted);
                    moments[1][lane] = add_rn(
                        moments[1][lane], multiply_rn(weighted, weighted)
                    );
                }
                vector_index += moment_threads;
            }

            const int tail_start = samples - samples % values_per_load;
            const int tail_sample = tail_start + virtual_thread;
            if (tail_sample < samples) {
                const float weighted = sample_weight[tail_sample];
                moments[0][0] = add_rn(moments[0][0], weighted);
                moments[1][0] = add_rn(
                    moments[1][0], multiply_rn(weighted, weighted)
                );
            }

#pragma unroll
            for (int channel = 0; channel < 2; ++channel) {
                float reduced = moments[channel][0];
#pragma unroll
                for (int lane = 1; lane < values_per_load; ++lane) {
                    reduced = add_rn(reduced, moments[channel][lane]);
                }
                partial_moments[channel * moment_threads + virtual_thread] = reduced;
            }

            __threadfence();
            __syncthreads();
            __shared__ bool is_last_moment_block;
            if (linear_thread == 0) {
                const int previous = atomicAdd(semaphores + dimensions, 1);
                is_last_moment_block = previous == dimensions - 1;
            }
            __syncthreads();
            if (is_last_moment_block) {
                float reduced[2];
#pragma unroll
                for (int channel = 0; channel < 2; ++channel) {
                    const float* channel_partials = partial_moments + channel * moment_threads;
                    reduced[channel] = add_rn(
                        add_rn(
                            channel_partials[linear_thread],
                            channel_partials[linear_thread + 256]
                        ),
                        add_rn(
                            channel_partials[linear_thread + 128],
                            channel_partials[linear_thread + 384]
                        )
                    );
                    shared[linear_thread * 2 + channel] = reduced[channel];
                }
                for (int offset = 64; offset >= 32; offset >>= 1) {
                    __syncthreads();
                    if (linear_thread < offset) {
#pragma unroll
                        for (int channel = 0; channel < 2; ++channel) {
                            reduced[channel] = add_rn(
                                reduced[channel], shared[(linear_thread + offset) * 2 + channel]
                            );
                            shared[linear_thread * 2 + channel] = reduced[channel];
                        }
                    }
                }
                __syncthreads();
                for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
                    for (int channel = 0; channel < 2; ++channel) {
                        reduced[channel] = add_rn(
                            reduced[channel],
                            __shfl_down_sync(0xffffffffU, reduced[channel], offset)
                        );
                    }
                }
                if (linear_thread == 0) {
                    total[0] = add_rn(initialize ? 0.0F : total[0], reduced[0]);
                    total_square[0] = add_rn(
                        initialize ? 0.0F : total_square[0], reduced[1]
                    );
                    semaphores[dimensions] = 0;
                }
            }
            __syncthreads();
        }
    }

    const int first_bin = threadIdx.x * values_per_load;
    const int input_stride = block_height * ctas_per_dimension;
    int sample = threadIdx.y + blockIdx.y * block_height;

    float thread_values[2][values_per_load][values_per_load] = {};
    while (sample + (values_per_load - 1) * input_stride < samples) {
#pragma unroll
        for (int input = 0; input < values_per_load; ++input) {
            const int input_sample = sample + input * input_stride;
            const int64_t label = bin_index[input_sample * dimensions + dimension];
            const float input_weight = sample_weight[input_sample];
            const float weight = accumulate_weighted_moments ? fabsf(input_weight) : input_weight;
#pragma unroll
            for (int lane = 0; lane < values_per_load; ++lane) {
                const bool matches = label == first_bin + lane;
                thread_values[0][input][lane] = add_rn(
                    thread_values[0][input][lane], matches ? weight : 0.0F
                );
                thread_values[1][input][lane] = add_rn(
                    thread_values[1][input][lane], matches ? 1.0F : 0.0F
                );
            }
        }
        sample += input_stride * values_per_load;
    }

#pragma unroll
    for (int input = 0; input < values_per_load; ++input) {
        if (sample >= samples) {
            break;
        }
        const int64_t label = bin_index[sample * dimensions + dimension];
        const float input_weight = sample_weight[sample];
        const float weight = accumulate_weighted_moments ? fabsf(input_weight) : input_weight;
#pragma unroll
        for (int lane = 0; lane < values_per_load; ++lane) {
            const bool matches = label == first_bin + lane;
            thread_values[0][input][lane] = add_rn(
                thread_values[0][input][lane], matches ? weight : 0.0F
            );
            thread_values[1][input][lane] = add_rn(
                thread_values[1][input][lane], matches ? 1.0F : 0.0F
            );
        }
        sample += input_stride;
    }

    float value[2][values_per_load];
#pragma unroll
    for (int channel = 0; channel < 2; ++channel) {
#pragma unroll
        for (int lane = 0; lane < values_per_load; ++lane) {
            value[channel][lane] = thread_values[channel][0][lane];
#pragma unroll
            for (int input = 1; input < values_per_load; ++input) {
                value[channel][lane] = add_rn(
                    value[channel][lane], thread_values[channel][input][lane]
                );
            }
        }
    }

    reduce_block_y(value, shared);
    if (threadIdx.y == 0) {
        const int partial_base =
            (dimension * ctas_per_dimension + blockIdx.y) * bins + first_bin;
#pragma unroll
        for (int lane = 0; lane < values_per_load; ++lane) {
            partial_histogram[partial_base + lane] = value[0][lane];
            partial_count[partial_base + lane] = value[1][lane];
        }
    }

    __threadfence();
    __syncthreads();
    __shared__ bool is_last_block;
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        const int previous = atomicAdd(semaphores + dimension, 1);
        is_last_block = previous == ctas_per_dimension - 1;
    }
    __syncthreads();
    if (!is_last_block) {
        return;
    }
    __threadfence();

#pragma unroll
    for (int channel = 0; channel < 2; ++channel) {
#pragma unroll
        for (int lane = 0; lane < values_per_load; ++lane) {
            value[channel][lane] = 0.0F;
        }
    }
    for (int partial = threadIdx.y; partial < ctas_per_dimension; partial += block_height) {
        const int partial_base = (dimension * ctas_per_dimension + partial) * bins + first_bin;
#pragma unroll
        for (int lane = 0; lane < values_per_load; ++lane) {
            value[0][lane] = add_rn(value[0][lane], partial_histogram[partial_base + lane]);
            value[1][lane] = add_rn(value[1][lane], partial_count[partial_base + lane]);
        }
    }
    reduce_block_y(value, shared);
    if (threadIdx.y == 0) {
        const int output_base = dimension * bins + first_bin;
#pragma unroll
        for (int lane = 0; lane < values_per_load; ++lane) {
            histogram[output_base + lane] = add_rn(
                initialize ? 0.0F : histogram[output_base + lane], value[0][lane]
            );
            bin_count[output_base + lane] = add_rn(
                initialize ? 0.0F : bin_count[output_base + lane], value[1][lane]
            );
        }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        semaphores[dimension] = 0;
    }
}

void vegas_histogram(
    torch::Tensor bin_index,
    torch::Tensor absolute_weight,
    torch::Tensor histogram,
    torch::Tensor bin_count,
    torch::Tensor semaphores,
    bool initialize
) {
    TORCH_CHECK(bin_index.is_cuda(), "bin_index must be on CUDA");
    TORCH_CHECK(bin_index.scalar_type() == torch::kInt64, "bin_index must be int64");
    TORCH_CHECK(
        bin_index.dim() == 2 && bin_index.size(1) == dimensions && bin_index.is_contiguous(),
        "bin_index must be a contiguous [samples, 4] matrix"
    );
    TORCH_CHECK(absolute_weight.is_cuda(), "absolute_weight must be on CUDA");
    TORCH_CHECK(
        absolute_weight.scalar_type() == torch::kFloat32 && absolute_weight.dim() == 1 &&
            absolute_weight.is_contiguous(),
        "absolute_weight must be contiguous float32"
    );
    TORCH_CHECK(
        absolute_weight.numel() == bin_index.size(0),
        "bin_index and absolute_weight must have equal sample counts"
    );
    const std::vector<torch::Tensor> outputs = {histogram, bin_count};
    for (const auto& output : outputs) {
        TORCH_CHECK(output.is_cuda(), "histogram outputs must be on CUDA");
        TORCH_CHECK(
            output.scalar_type() == torch::kFloat32 && output.sizes() == torch::IntArrayRef({4, 32}) &&
                output.is_contiguous(),
            "histogram outputs must be contiguous float32 [4, 32] matrices"
        );
        TORCH_CHECK(output.device() == bin_index.device(), "all tensors must use one CUDA device");
    }
    TORCH_CHECK(
        absolute_weight.device() == bin_index.device(), "all tensors must use one CUDA device"
    );
    TORCH_CHECK(
        semaphores.is_cuda() && semaphores.scalar_type() == torch::kInt32 &&
            semaphores.dim() == 1 && semaphores.numel() >= dimensions &&
            semaphores.is_contiguous(),
        "Vegas histogram semaphores must contain at least four contiguous CUDA int32 values"
    );
    TORCH_CHECK(
        semaphores.device() == bin_index.device(), "all tensors must use one CUDA device"
    );

    const int samples = static_cast<int>(absolute_weight.numel());
    TORCH_CHECK(samples > 0, "Vegas histogram requires at least one sample");
    const int values_per_thread = (samples + block_height - 1) / block_height;
    int ctas_per_dimension = 1;
    if (values_per_thread >= 256) {
        const auto* properties = at::cuda::getCurrentDeviceProperties();
        const int blocks_per_sm = properties->maxThreadsPerMultiProcessor /
            (block_width * block_height);
        const int target_blocks = properties->multiProcessorCount * blocks_per_sm;
        const int ctas_for_minimum_work = (values_per_thread + 15) / 16;
        const int ctas_for_maximum_work = (values_per_thread + 255) / 256;
        ctas_per_dimension = std::max(
            std::min(target_blocks, ctas_for_minimum_work), ctas_for_maximum_work
        );
    }

    auto partial_histogram = torch::empty(
        {dimensions, ctas_per_dimension, bins}, absolute_weight.options()
    );
    auto partial_count = torch::empty_like(partial_histogram);
    const dim3 block(block_width, block_height);
    const dim3 grid(1, ctas_per_dimension, dimensions);
    constexpr int shared_bytes =
        block_width * block_height * 2 * values_per_load * sizeof(float);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    vegas_histogram_kernel<false><<<grid, block, shared_bytes, stream>>>(
        bin_index.data_ptr<int64_t>(),
        absolute_weight.data_ptr<float>(),
        histogram.data_ptr<float>(),
        bin_count.data_ptr<float>(),
        partial_histogram.data_ptr<float>(),
        partial_count.data_ptr<float>(),
        nullptr,
        nullptr,
        nullptr,
        semaphores.data_ptr<int>(),
        samples,
        ctas_per_dimension,
        initialize
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> vegas_random(
    int64_t samples,
    torch::Tensor reference,
    at::Generator generator
) {
    TORCH_CHECK(
        reference.is_cuda() && reference.scalar_type() == torch::kFloat32,
        "Vegas random generation requires a CUDA float32 reference tensor"
    );
    TORCH_CHECK(samples > 0, "Vegas random generation requires a positive sample count");
    TORCH_CHECK(
        samples <= std::numeric_limits<int32_t>::max() / dimensions,
        "Vegas random generation sample count is too large"
    );
    TORCH_CHECK(
        generator.device().is_cuda(),
        "Vegas random generation requires a CUDA generator"
    );

    const int64_t values = samples * dimensions;
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    const uint32_t blocks_per_sm = properties->maxThreadsPerMultiProcessor / random_threads;
    const uint32_t maximum_blocks = properties->multiProcessorCount * blocks_per_sm;
    const uint32_t requested_blocks = static_cast<uint32_t>(
        (values + random_threads - 1) / random_threads
    );
    const uint32_t blocks = std::min(maximum_blocks, requested_blocks);
    const uint64_t counter_offset =
        ((values - 1) /
             (static_cast<int64_t>(random_threads) * blocks * random_values_per_thread) +
         1) *
        4;

    auto* cuda_generator = generator.get<at::CUDAGeneratorImpl>();
    at::PhiloxCudaState integer_state;
    at::PhiloxCudaState uniform_state;
    {
        std::lock_guard<std::mutex> lock(cuda_generator->mutex_);
        integer_state = cuda_generator->philox_cuda_state(counter_offset);
        uniform_state = cuda_generator->philox_cuda_state(counter_offset);
    }

    auto bin_index = torch::empty(
        {samples, dimensions}, reference.options().dtype(torch::kInt64)
    );
    auto random = torch::empty({samples, dimensions}, reference.options());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    vegas_random_kernel<<<blocks, random_threads, 0, stream>>>(
        bin_index.data_ptr<int64_t>(),
        random.data_ptr<float>(),
        values,
        integer_state,
        uniform_state
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {bin_index, random};
}

void vegas_weighted_histogram(
    torch::Tensor bin_index,
    torch::Tensor weighted_value,
    torch::Tensor total,
    torch::Tensor total_square,
    torch::Tensor histogram,
    torch::Tensor bin_count,
    torch::Tensor semaphores,
    bool initialize
) {
    TORCH_CHECK(
        bin_index.is_cuda() && bin_index.scalar_type() == torch::kInt64 &&
            bin_index.dim() == 2 && bin_index.size(1) == dimensions && bin_index.is_contiguous(),
        "bin_index must be a contiguous CUDA int64 [samples, 4] matrix"
    );
    TORCH_CHECK(
        weighted_value.is_cuda() && weighted_value.scalar_type() == torch::kFloat32 &&
            weighted_value.dim() == 1 && weighted_value.is_contiguous() &&
            weighted_value.numel() == bin_index.size(0),
        "weighted values must be contiguous CUDA float32 with the matching sample count"
    );
    const std::vector<torch::Tensor> scalars = {total, total_square};
    for (const auto& scalar : scalars) {
        TORCH_CHECK(
            scalar.is_cuda() && scalar.scalar_type() == torch::kFloat32 && scalar.numel() == 1 &&
                scalar.is_contiguous(),
            "Vegas moment outputs must be contiguous CUDA float32 scalars"
        );
        TORCH_CHECK(scalar.device() == bin_index.device(), "all tensors must use one CUDA device");
    }
    const std::vector<torch::Tensor> histograms = {histogram, bin_count};
    for (const auto& output : histograms) {
        TORCH_CHECK(
            output.is_cuda() && output.scalar_type() == torch::kFloat32 &&
                output.sizes() == torch::IntArrayRef({dimensions, bins}) &&
                output.is_contiguous(),
            "Vegas histograms must be contiguous CUDA float32 [4, 32] matrices"
        );
        TORCH_CHECK(output.device() == bin_index.device(), "all tensors must use one CUDA device");
    }
    TORCH_CHECK(
        weighted_value.device() == bin_index.device(), "all tensors must use one CUDA device"
    );
    TORCH_CHECK(
        semaphores.is_cuda() && semaphores.scalar_type() == torch::kInt32 &&
            semaphores.dim() == 1 && semaphores.numel() >= dimensions + 1 &&
            semaphores.is_contiguous() && semaphores.device() == bin_index.device(),
        "weighted Vegas histograms require at least five contiguous CUDA int32 semaphores"
    );

    const int samples = static_cast<int>(weighted_value.numel());
    TORCH_CHECK(
        samples > 0 && samples <= 65536,
        "weighted Vegas batches must contain 1-65536 samples"
    );
    const int values_per_thread = (samples + block_height - 1) / block_height;
    int ctas_per_dimension = 1;
    if (values_per_thread >= 256) {
        const auto* properties = at::cuda::getCurrentDeviceProperties();
        const int blocks_per_sm = properties->maxThreadsPerMultiProcessor /
            (block_width * block_height);
        const int target_blocks = properties->multiProcessorCount * blocks_per_sm;
        const int ctas_for_minimum_work = (values_per_thread + 15) / 16;
        const int ctas_for_maximum_work = (values_per_thread + 255) / 256;
        ctas_per_dimension = std::max(
            std::min(target_blocks, ctas_for_minimum_work), ctas_for_maximum_work
        );
    }

    auto partial_histogram = torch::empty(
        {dimensions, ctas_per_dimension, bins}, weighted_value.options()
    );
    auto partial_count = torch::empty_like(partial_histogram);
    auto partial_moments = torch::empty({2, moment_threads}, weighted_value.options());
    const dim3 block(block_width, block_height);
    const dim3 grid(1, ctas_per_dimension, dimensions);
    constexpr int shared_bytes =
        block_width * block_height * 2 * values_per_load * sizeof(float);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    vegas_histogram_kernel<true><<<grid, block, shared_bytes, stream>>>(
        bin_index.data_ptr<int64_t>(),
        weighted_value.data_ptr<float>(),
        histogram.data_ptr<float>(),
        bin_count.data_ptr<float>(),
        partial_histogram.data_ptr<float>(),
        partial_count.data_ptr<float>(),
        total.data_ptr<float>(),
        total_square.data_ptr<float>(),
        partial_moments.data_ptr<float>(),
        semaphores.data_ptr<int>(),
        samples,
        ctas_per_dimension,
        initialize
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor vegas_accumulate_moments(
    torch::Tensor value,
    torch::Tensor width,
    torch::Tensor volume,
    torch::Tensor total,
    torch::Tensor total_square,
    bool initialize
) {
    TORCH_CHECK(value.is_cuda(), "value must be on CUDA");
    TORCH_CHECK(
        value.scalar_type() == torch::kFloat32 && value.dim() == 1 && value.is_contiguous(),
        "value must be contiguous float32"
    );
    TORCH_CHECK(width.is_cuda(), "width must be on CUDA");
    TORCH_CHECK(
        width.scalar_type() == torch::kFloat32 && width.dim() == 2 && width.size(1) == dimensions,
        "width must be a float32 [samples, 4] matrix"
    );
    TORCH_CHECK(width.size(0) == value.size(0), "value and width must have equal sample counts");
    const std::vector<torch::Tensor> scalars = {volume, total, total_square};
    for (const auto& scalar : scalars) {
        TORCH_CHECK(scalar.is_cuda(), "Vegas moment scalars must be on CUDA");
        TORCH_CHECK(
            scalar.scalar_type() == torch::kFloat32 && scalar.numel() == 1 &&
                scalar.is_contiguous(),
            "Vegas moments require contiguous float32 scalars"
        );
        TORCH_CHECK(scalar.device() == value.device(), "all tensors must use one CUDA device");
    }
    TORCH_CHECK(
        width.device() == value.device(), "all tensors must use one CUDA device"
    );

    const int samples = static_cast<int>(value.numel());
    TORCH_CHECK(samples > 0, "Vegas weighting requires at least one sample");
    TORCH_CHECK(
        samples <= 65536,
        "fused Vegas moments support batches of at most 65536 samples"
    );
    auto absolute_weight = torch::empty_like(value);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int shared_bytes = moment_threads * sizeof(float);
    vegas_weighted_moments_kernel<<<2, moment_threads, shared_bytes, stream>>>(
        value.data_ptr<float>(),
        width.data_ptr<float>(),
        volume.data_ptr<float>(),
        total.data_ptr<float>(),
        total_square.data_ptr<float>(),
        absolute_weight.data_ptr<float>(),
        samples,
        width.stride(0),
        width.stride(1),
        initialize
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return absolute_weight;
}

torch::Tensor vegas_accumulate_weighted_moments(
    torch::Tensor weighted_value,
    torch::Tensor total,
    torch::Tensor total_square,
    bool initialize
) {
    TORCH_CHECK(weighted_value.is_cuda(), "weighted_value must be on CUDA");
    TORCH_CHECK(
        weighted_value.scalar_type() == torch::kFloat32 && weighted_value.dim() == 1 &&
            weighted_value.is_contiguous(),
        "weighted_value must be contiguous float32"
    );
    const std::vector<torch::Tensor> scalars = {total, total_square};
    for (const auto& scalar : scalars) {
        TORCH_CHECK(scalar.is_cuda(), "Vegas moment scalars must be on CUDA");
        TORCH_CHECK(
            scalar.scalar_type() == torch::kFloat32 && scalar.numel() == 1 &&
                scalar.is_contiguous(),
            "Vegas moments require contiguous float32 scalars"
        );
        TORCH_CHECK(
            scalar.device() == weighted_value.device(), "all tensors must use one CUDA device"
        );
    }

    const int samples = static_cast<int>(weighted_value.numel());
    TORCH_CHECK(samples > 0, "Vegas moments require at least one sample");
    TORCH_CHECK(
        samples <= 65536,
        "fused Vegas moments support batches of at most 65536 samples"
    );
    auto absolute_weight = torch::empty_like(weighted_value);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int shared_bytes = moment_threads * sizeof(float);
    vegas_weighted_value_moments_kernel<<<2, moment_threads, shared_bytes, stream>>>(
        weighted_value.data_ptr<float>(),
        total.data_ptr<float>(),
        total_square.data_ptr<float>(),
        absolute_weight.data_ptr<float>(),
        samples,
        initialize
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return absolute_weight;
}

torch::Tensor vegas_adapt_edges(
    torch::Tensor edges,
    torch::Tensor histogram,
    torch::Tensor bin_count
) {
    const std::vector<torch::Tensor> tensors = {edges, histogram, bin_count};
    for (const auto& tensor : tensors) {
        TORCH_CHECK(tensor.is_cuda(), "Vegas adaptation tensors must be on CUDA");
        TORCH_CHECK(
            tensor.scalar_type() == torch::kFloat32 && tensor.is_contiguous(),
            "Vegas adaptation tensors must be contiguous float32"
        );
        TORCH_CHECK(tensor.device() == edges.device(), "all tensors must use one CUDA device");
    }
    TORCH_CHECK(
        edges.sizes() == torch::IntArrayRef({dimensions, bins + 1}),
        "Vegas edges must be a [4, 33] matrix"
    );
    TORCH_CHECK(
        histogram.sizes() == torch::IntArrayRef({dimensions, bins}) &&
            bin_count.sizes() == histogram.sizes(),
        "Vegas histograms must be matching [4, 32] matrices"
    );

    auto output = torch::empty_like(edges);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    vegas_adapt_edges_kernel<<<dimensions, adaptation_threads, 0, stream>>>(
        edges.data_ptr<float>(),
        histogram.data_ptr<float>(),
        bin_count.data_ptr<float>(),
        output.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor vegas_estimate_variance(
    torch::Tensor total,
    torch::Tensor total_square,
    int64_t samples
) {
    const std::vector<torch::Tensor> tensors = {total, total_square};
    for (const auto& tensor : tensors) {
        TORCH_CHECK(
            tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32 && tensor.numel() == 1 &&
                tensor.is_contiguous(),
            "Vegas totals must be contiguous CUDA float32 scalars"
        );
        TORCH_CHECK(tensor.device() == total.device(), "all tensors must use one CUDA device");
    }
    TORCH_CHECK(samples > 1, "Vegas variance requires at least two samples");

    auto output = torch::empty({2}, total.options());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    vegas_estimate_variance_kernel<<<1, 1, 0, stream>>>(
        total.data_ptr<float>(),
        total_square.data_ptr<float>(),
        output.data_ptr<float>(),
        output.data_ptr<float>() + 1,
        0,
        static_cast<float>(1.0 / static_cast<double>(samples)),
        static_cast<float>(1.0 / static_cast<double>(samples - 1)),
        std::numeric_limits<float>::min()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

void vegas_store_estimate_variance(
    torch::Tensor total,
    torch::Tensor total_square,
    int64_t samples,
    torch::Tensor estimates,
    torch::Tensor variances,
    int64_t output_index
) {
    const std::vector<torch::Tensor> scalars = {total, total_square};
    for (const auto& tensor : scalars) {
        TORCH_CHECK(
            tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32 && tensor.numel() == 1 &&
                tensor.is_contiguous(),
            "Vegas totals must be contiguous CUDA float32 scalars"
        );
        TORCH_CHECK(tensor.device() == total.device(), "all tensors must use one CUDA device");
    }
    const std::vector<torch::Tensor> buffers = {estimates, variances};
    for (const auto& tensor : buffers) {
        TORCH_CHECK(
            tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32 && tensor.dim() == 1 &&
                tensor.is_contiguous(),
            "Vegas estimate buffers must be contiguous CUDA float32 vectors"
        );
        TORCH_CHECK(tensor.device() == total.device(), "all tensors must use one CUDA device");
    }
    TORCH_CHECK(samples > 1, "Vegas variance requires at least two samples");
    TORCH_CHECK(
        estimates.numel() == variances.numel() && output_index >= 0 &&
            output_index < estimates.numel(),
        "Vegas estimate output index is outside the matched buffers"
    );

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    vegas_estimate_variance_kernel<<<1, 1, 0, stream>>>(
        total.data_ptr<float>(),
        total_square.data_ptr<float>(),
        estimates.data_ptr<float>(),
        variances.data_ptr<float>(),
        output_index,
        static_cast<float>(1.0 / static_cast<double>(samples)),
        static_cast<float>(1.0 / static_cast<double>(samples - 1)),
        std::numeric_limits<float>::min()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor vegas_combine_estimates(
    torch::Tensor estimates,
    torch::Tensor variances,
    int64_t iterations
) {
    const std::vector<torch::Tensor> tensors = {estimates, variances};
    for (const auto& tensor : tensors) {
        TORCH_CHECK(
            tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32 && tensor.dim() == 1 &&
                tensor.is_contiguous(),
            "Vegas estimates must be contiguous CUDA float32 vectors"
        );
        TORCH_CHECK(
            tensor.device() == estimates.device(), "all tensors must use one CUDA device"
        );
    }
    TORCH_CHECK(
        estimates.numel() == variances.numel() && iterations >= 1 && iterations <= 4 &&
            iterations <= estimates.numel(),
        "Vegas combination requires one to four populated estimates and variances"
    );

    auto output = torch::empty({2}, estimates.options());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    vegas_combine_estimates_kernel<<<1, 1, 0, stream>>>(
        estimates.data_ptr<float>(),
        variances.data_ptr<float>(),
        output.data_ptr<float>(),
        iterations
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor vegas_store_and_combine(
    torch::Tensor total,
    torch::Tensor total_square,
    int64_t samples,
    torch::Tensor estimates,
    torch::Tensor variances,
    int64_t output_index
) {
    const std::vector<torch::Tensor> scalars = {total, total_square};
    for (const auto& tensor : scalars) {
        TORCH_CHECK(
            tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32 && tensor.numel() == 1 &&
                tensor.is_contiguous(),
            "Vegas totals must be contiguous CUDA float32 scalars"
        );
        TORCH_CHECK(tensor.device() == total.device(), "all tensors must use one CUDA device");
    }
    const std::vector<torch::Tensor> buffers = {estimates, variances};
    for (const auto& tensor : buffers) {
        TORCH_CHECK(
            tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32 && tensor.dim() == 1 &&
                tensor.is_contiguous(),
            "Vegas estimate buffers must be contiguous CUDA float32 vectors"
        );
        TORCH_CHECK(tensor.device() == total.device(), "all tensors must use one CUDA device");
    }
    TORCH_CHECK(samples > 1, "Vegas variance requires at least two samples");
    TORCH_CHECK(
        estimates.numel() == variances.numel() && estimates.numel() <= 4 && output_index >= 0 &&
            output_index < estimates.numel(),
        "Vegas combined output index is outside one to four matched buffers"
    );

    auto output = torch::empty({2}, total.options());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    vegas_store_and_combine_kernel<<<1, 1, 0, stream>>>(
        total.data_ptr<float>(),
        total_square.data_ptr<float>(),
        estimates.data_ptr<float>(),
        variances.data_ptr<float>(),
        output.data_ptr<float>(),
        output_index,
        static_cast<float>(1.0 / static_cast<double>(samples)),
        static_cast<float>(1.0 / static_cast<double>(samples - 1)),
        std::numeric_limits<float>::min()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<torch::Tensor> vegas_initialize(torch::Tensor bounds) {
    TORCH_CHECK(
        bounds.is_cuda() && bounds.scalar_type() == torch::kFloat32 &&
            bounds.sizes() == torch::IntArrayRef({dimensions, 2}) && bounds.is_contiguous(),
        "Vegas initialization requires contiguous CUDA float32 [4, 2] bounds"
    );

    auto edges = torch::empty({dimensions, bins + 1}, bounds.options());
    auto bounds_lower = torch::empty({dimensions}, bounds.options());
    auto bounds_width = torch::empty({dimensions}, bounds.options());
    auto volume = torch::empty({}, bounds.options());
    auto semaphores = torch::empty(
        {dimensions + 1}, bounds.options().dtype(torch::kInt32)
    );
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    vegas_initialize_kernel<<<1, 128, 0, stream>>>(
        bounds.data_ptr<float>(),
        edges.data_ptr<float>(),
        bounds_lower.data_ptr<float>(),
        bounds_width.data_ptr<float>(),
        volume.data_ptr<float>(),
        semaphores.data_ptr<int>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {edges, bounds_lower, bounds_width, volume, semaphores};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("vegas_random", &vegas_random, "Fused Vegas integer and uniform random values");
    module.def("vegas_histogram", &vegas_histogram, "Deterministic fused Vegas histogram");
    module.def(
        "vegas_weighted_histogram",
        &vegas_weighted_histogram,
        "Fused weighted moments and deterministic Vegas histogram"
    );
    module.def(
        "vegas_accumulate_moments",
        &vegas_accumulate_moments,
        "Fused four-dimensional Vegas weights and moments"
    );
    module.def(
        "vegas_accumulate_weighted_moments",
        &vegas_accumulate_weighted_moments,
        "Fused moments for preweighted Vegas samples"
    );
    module.def("vegas_adapt_edges", &vegas_adapt_edges, "Fused deterministic Vegas adaptation");
    module.def(
        "vegas_estimate_variance",
        &vegas_estimate_variance,
        "Fused Vegas estimate and variance"
    );
    module.def(
        "vegas_store_estimate_variance",
        &vegas_store_estimate_variance,
        "Store fused Vegas estimate and variance"
    );
    module.def(
        "vegas_combine_estimates",
        &vegas_combine_estimates,
        "Fused inverse-variance combination"
    );
    module.def(
        "vegas_store_and_combine",
        &vegas_store_and_combine,
        "Store and combine one Vegas estimate"
    );
    module.def("vegas_initialize", &vegas_initialize, "Fused Vegas state initialization");
}
