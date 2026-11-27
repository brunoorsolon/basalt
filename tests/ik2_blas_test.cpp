// Inkling's relative-bias FP32 and dense FP16 products against CPU double sums.
#include "ik2_hip_compat.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do { const auto status = (call); if (status != 0) { \
    std::fprintf(stderr, "%s returned %d\n", #call, (int) status); std::exit(1); } } while (0)

template <class T>
void compare(cublasHandle_t blas, int n, int columns, int k, bool half) {
    std::vector<T> weights((size_t) n * k), inputs((size_t) columns * k);
    std::vector<double> w(weights.size()), x(inputs.size());
    for (size_t i = 0; i < w.size(); ++i) {
        const float value = std::sin((float) i * .137f) * .7f;
        weights[i] = static_cast<T>(value);
        w[i] = static_cast<double>(weights[i]);
    }
    for (size_t i = 0; i < x.size(); ++i) {
        const float value = std::cos((float) i * .219f) * .8f;
        inputs[i] = static_cast<T>(value);
        x[i] = static_cast<double>(inputs[i]);
    }
    T *dw = nullptr, *dx = nullptr;
    float* dy = nullptr;
    CHECK(cudaMalloc(&dw, weights.size() * sizeof(T)));
    CHECK(cudaMalloc(&dx, inputs.size() * sizeof(T)));
    CHECK(cudaMalloc(&dy, (size_t) n * columns * sizeof(float)));
    CHECK(cudaMemcpy(dw, weights.data(), weights.size() * sizeof(T), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx, inputs.data(), inputs.size() * sizeof(T), cudaMemcpyHostToDevice));
    const float one = 1, zero = 0;
    CHECK(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, columns, k, &one,
        dw, half ? CUDA_R_16F : CUDA_R_32F, k, dx, half ? CUDA_R_16F : CUDA_R_32F, k,
        &zero, dy, CUDA_R_32F, n, half ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_PEDANTIC,
        CUBLAS_GEMM_DEFAULT));
    std::vector<float> result((size_t) n * columns);
    CHECK(cudaMemcpy(result.data(), dy, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double worst = 0;
    for (int c = 0; c < columns; ++c) for (int r = 0; r < n; ++r) {
        double reference = 0;
        for (int j = 0; j < k; ++j) reference += w[(size_t) r * k + j] * x[(size_t) c * k + j];
        const double error = std::abs(result[(size_t) c * n + r] - reference);
        const double tolerance = 2e-4 + 2e-5 * std::abs(reference);
        if (!std::isfinite(result[(size_t) c * n + r]) || error > tolerance) {
            std::fprintf(stderr, "BLAS %s [%d,%d] %.9g != %.12g (error %.3g > %.3g)\n",
                         half ? "FP16" : "FP32", c, r, result[(size_t) c * n + r], reference, error, tolerance);
            std::exit(1);
        }
        worst = std::max(worst, error);
    }
    std::printf("PASS BLAS %s n=%d columns=%d k=%d max_abs_error=%.9g\n", half ? "FP16" : "FP32", n, columns, k, worst);
    CHECK(cudaFree(dw)); CHECK(cudaFree(dx)); CHECK(cudaFree(dy));
}

int main() {
    int devices = 0;
    CHECK(cudaGetDeviceCount(&devices));
    if (devices != 1) return 2;
    cublasHandle_t blas{};
    CHECK(cublasCreate(&blas));
    CHECK(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
    compare<float>(blas, 1024, 24 * 20, 16, false);
    compare<_Float16>(blas, 64, 8, 2304, true);
    CHECK(hipblasDestroy(blas));
}
