#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)          \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';     \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                   \
    } while (0)

constexpr int ROWS = 4096;
constexpr int COLS = 1024;
constexpr int THREADS = 256;

// ------------------------------------------------------------
// CPU reference implementation
// ------------------------------------------------------------
void softmax_cpu(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* rowIn = input + r * cols;
        float* rowOut = output + r * cols;

        float maxVal = -INFINITY;
        for (int c = 0; c < cols; ++c) {
            maxVal = std::max(maxVal, rowIn[c]);
        }

        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            sum += std::exp(rowIn[c] - maxVal);
        }

        for (int c = 0; c < cols; ++c) {
            rowOut[c] = std::exp(rowIn[c] - maxVal) / sum;
        }
    }
}

// ------------------------------------------------------------
// GPU reduction helper: reduce shared values to one maximum
// ------------------------------------------------------------
__device__ float block_reduce_max(float value, float* shared) {
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    return shared[0];
}

// ------------------------------------------------------------
// GPU reduction helper: reduce shared values to one sum
// ------------------------------------------------------------
__device__ float block_reduce_sum(float value, float* shared) {
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    return shared[0];
}

// ------------------------------------------------------------
// Unfused softmax: kernel 1 computes row maximum
// ------------------------------------------------------------
__global__ void row_max_kernel(
    const float* input,
    float* rowMax,
    int rows,
    int cols) {

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) return;

    __shared__ float shared[THREADS];

    const float* rowIn = input + row * cols;
    float localMax = -INFINITY;

    for (int c = tid; c < cols; c += blockDim.x) {
        localMax = fmaxf(localMax, rowIn[c]);
    }

    const float maxVal = block_reduce_max(localMax, shared);

    if (tid == 0) {
        rowMax[row] = maxVal;
    }
}

// ------------------------------------------------------------
// Unfused softmax: kernel 2 computes row sum of exponentials
// ------------------------------------------------------------
__global__ void row_sum_kernel(
    const float* input,
    const float* rowMax,
    float* rowSum,
    int rows,
    int cols) {

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) return;

    __shared__ float shared[THREADS];

    const float* rowIn = input + row * cols;
    const float maxVal = rowMax[row];

    float localSum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        localSum += expf(rowIn[c] - maxVal);
    }

    const float sumVal = block_reduce_sum(localSum, shared);

    if (tid == 0) {
        rowSum[row] = sumVal;
    }
}

// ------------------------------------------------------------
// Unfused softmax: kernel 3 normalizes each row
// ------------------------------------------------------------
__global__ void normalize_kernel(
    const float* input,
    const float* rowMax,
    const float* rowSum,
    float* output,
    int rows,
    int cols) {

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) return;

    const float* rowIn = input + row * cols;
    float* rowOut = output + row * cols;

    const float maxVal = rowMax[row];
    const float sumVal = rowSum[row];

    for (int c = tid; c < cols; c += blockDim.x) {
        rowOut[c] = expf(rowIn[c] - maxVal) / sumVal;
    }
}

// ------------------------------------------------------------
// Fused softmax: max, exponential sum, and normalization
// all performed in one kernel launch per row
// ------------------------------------------------------------
__global__ void fused_softmax_kernel(
    const float* input,
    float* output,
    int rows,
    int cols) {

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows) return;

    __shared__ float shared[THREADS];

    const float* rowIn = input + row * cols;
    float* rowOut = output + row * cols;

    // Step 1: maximum
    float localMax = -INFINITY;
    for (int c = tid; c < cols; c += blockDim.x) {
        localMax = fmaxf(localMax, rowIn[c]);
    }

    const float maxVal = block_reduce_max(localMax, shared);

    // Step 2: sum of exponentials
    float localSum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        localSum += expf(rowIn[c] - maxVal);
    }

    const float sumVal = block_reduce_sum(localSum, shared);

    // Step 3: normalize
    for (int c = tid; c < cols; c += blockDim.x) {
        rowOut[c] = expf(rowIn[c] - maxVal) / sumVal;
    }
}

// ------------------------------------------------------------
// Timing and correctness helpers
// ------------------------------------------------------------
float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

float max_abs_error(
    const std::vector<float>& a,
    const std::vector<float>& b) {

    float maxError = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        maxError = std::max(maxError, std::fabs(a[i] - b[i]));
    }
    return maxError;
}

int main() {
    const size_t count = static_cast<size_t>(ROWS) * COLS;
    const size_t bytes = count * sizeof(float);

    std::cout << "Softmax Kernel Fusion Experiment\n";
    std::cout << "Rows: " << ROWS << ", Columns: " << COLS << "\n\n";

    // Generate deterministic input.
    std::vector<float> h_input(count);
    std::vector<float> h_cpu(count);
    std::vector<float> h_unfused(count);
    std::vector<float> h_fused(count);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    for (float& value : h_input) {
        value = dist(rng);
    }

    // CPU reference and timing.
    const auto cpuStart = std::chrono::high_resolution_clock::now();
    softmax_cpu(h_input.data(), h_cpu.data(), ROWS, COLS);
    const auto cpuStop = std::chrono::high_resolution_clock::now();

    const double cpu_ms =
        std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count();

    // Allocate GPU memory.
    float *d_input = nullptr, *d_unfused = nullptr, *d_fused = nullptr;
    float *d_rowMax = nullptr, *d_rowSum = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_unfused, bytes));
    CUDA_CHECK(cudaMalloc(&d_fused, bytes));
    CUDA_CHECK(cudaMalloc(&d_rowMax, ROWS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rowSum, ROWS * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm up GPU kernels.
    row_max_kernel<<<ROWS, THREADS>>>(d_input, d_rowMax, ROWS, COLS);
    row_sum_kernel<<<ROWS, THREADS>>>(d_input, d_rowMax, d_rowSum, ROWS, COLS);
    normalize_kernel<<<ROWS, THREADS>>>(
        d_input, d_rowMax, d_rowSum, d_unfused, ROWS, COLS);
    fused_softmax_kernel<<<ROWS, THREADS>>>(d_input, d_fused, ROWS, COLS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time the unfused GPU implementation.
    CUDA_CHECK(cudaEventRecord(start));
    row_max_kernel<<<ROWS, THREADS>>>(d_input, d_rowMax, ROWS, COLS);
    row_sum_kernel<<<ROWS, THREADS>>>(d_input, d_rowMax, d_rowSum, ROWS, COLS);
    normalize_kernel<<<ROWS, THREADS>>>(
        d_input, d_rowMax, d_rowSum, d_unfused, ROWS, COLS);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const float unfused_ms = elapsed_ms(start, stop);

    // Time the fused GPU implementation.
    CUDA_CHECK(cudaEventRecord(start));
    fused_softmax_kernel<<<ROWS, THREADS>>>(d_input, d_fused, ROWS, COLS);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const float fused_ms = elapsed_ms(start, stop);

    // Copy results back.
    CUDA_CHECK(cudaMemcpy(
        h_unfused.data(), d_unfused, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        h_fused.data(), d_fused, bytes, cudaMemcpyDeviceToHost));

    const float unfused_error = max_abs_error(h_cpu, h_unfused);
    const float fused_error = max_abs_error(h_cpu, h_fused);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CPU time:                 " << cpu_ms << " ms\n";
    std::cout << "GPU unfused (3 kernels):  " << unfused_ms << " ms\n";
    std::cout << "GPU fused (1 kernel):    " << fused_ms << " ms\n\n";

    std::cout << "Unfused max abs error: " << unfused_error << "\n";
    std::cout << "Fused max abs error:   " << fused_error << "\n\n";

    if (fused_ms > 0.0f) {
        std::cout << "Unfused / fused speedup: "
                  << unfused_ms / fused_ms << "x\n";
    }

    // Save results for plotting.
    std::ofstream csv("results.csv");
    csv << "method,time_ms,speedup\n";
    csv << "CPU," << cpu_ms << ",1\n";
    csv << "GPU_Unfused," << unfused_ms << ","
        << (unfused_ms > 0.0f ? cpu_ms / unfused_ms : 0.0f) << "\n";
    csv << "GPU_Fused," << fused_ms << ","
        << (fused_ms > 0.0f ? cpu_ms / fused_ms : 0.0f) << "\n";
    csv.close();

    std::cout << "\nSaved timing results to results.csv\n";

    // Cleanup.
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_unfused));
    CUDA_CHECK(cudaFree(d_fused));
    CUDA_CHECK(cudaFree(d_rowMax));
    CUDA_CHECK(cudaFree(d_rowSum));

    return 0;
}