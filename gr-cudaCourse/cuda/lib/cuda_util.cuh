#ifndef LIB_CUDA_UTIL_CUH_
#define LIB_CUDA_UTIL_CUH_

#include <complex>
#include <cuda/std/complex>
#include <kernels/cuda/wrapper.hpp>
#include <utility>

/**
 * @brief Get the start index for this thread.
 * Note that the variables used in this function are "magic" hidden variables
 * exist inside of all __global__ and __device__ functions.
 */
__device__ __forceinline__ int get_start() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

/**
 * @brief Get the stride for this thread
 * Generally the same for all threads.
 * If we have more data to process than threads, each thread will process
 * multiple datapoints.  It is most efficient if each thread interleaves
 * the data to process rather than doing a block.
 * E.g. We have 30 points and 10 threads
 *   Thread 0: Process data 0,10,20
 *   Thread 1: Process data 1,11,21,
 *   ...
 * Note that the variables used in this function are "magic" hidden variables
 * exist inside of all __global__ and __device__ functions.
 */
__device__ __forceinline__ int get_stride() {
    return blockDim.x * gridDim.x;
}

#endif // LIB_CUDA_UTIL_CUH
