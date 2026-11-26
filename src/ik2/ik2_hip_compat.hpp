#pragma once
// Inkling-only CUDA-shaped APIs; keep the qualified Qwen runtime shim unchanged.
#ifdef STRATA_USE_HIP
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define cudaCpuDeviceId hipCpuDeviceId
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaMallocManaged hipMallocManaged
#define cudaMemPrefetchAsync hipMemPrefetchAsync
#define cudaDevAttrConcurrentManagedAccess hipDeviceAttributeConcurrentManagedAccess
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#if hipblasVersionMajor == 0
#define cublasSgemv rocblas_sgemv
#define CUBLAS_COMPUTE_32F_PEDANTIC rocblas_datatype_f32_r
#else
#define cublasSgemv hipblasSgemv
#define CUBLAS_COMPUTE_32F_PEDANTIC HIPBLAS_COMPUTE_32F_PEDANTIC
#endif
#endif
