#include <kernels/cuda/wrapper.hpp>
#include <stdexcept>
#include <string>

// Wrappers around cuda allocation functions.  None of these functions are
// meant to be called directly by user code.

namespace cuda {
namespace detail {

/**
 * Turn a cuda error code into a runtime error.
 */
void check_for_error(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(cudaGetErrorName(error)) + ": " + std::string(cudaGetErrorString(error)));
    }
}

/**
 * Allocate a buffer in cuda unified memory with `n` bytes.  Throws on failure.
 */
void* _unified_allocate(size_t n) {
    void* p;
    auto error = cudaMallocManaged(&p, n);
    check_for_error(error);
    return p;
}

/**
 * Allocate a buffer in cuda device memory with `n` bytes.  Throws on failure.
 */
void* _device_allocate(size_t n) {
    void* p;
    auto error = cudaMalloc(&p, n);
    check_for_error(error);
    return p;
}

/**
 * Allocate a pinned host buffer (dma) with `n` bytes.  Throws on failure.
 */
void* _dma_allocate(size_t n) {
    void* p;
    auto error = cudaMallocHost(&p, n);
    check_for_error(error);
    return p;
}

/**
 * Free memory that was allocated on the device or in unified memory.
 */
void _deallocate(void* p) {
    if (p) {
        // Can't throw from the destructor.  So we have to ignore the error.
        auto error = cudaFree(p);
    }
}

/**
 * Free a pinned host buffer (dma)
 */
void _dma_deallocate(void* p) {
    if (p) {
        // Can't throw from the destructor.  So we have to ignore the error.
        auto error = cudaFreeHost(p);
    }
}

}  // namespace detail

void memcpy(void* dst, const void* src, size_t count, const CudaStream& stream, cudaMemcpyKind kind) {
    // Note that we call the Async memcpy func.  There is a sync version, but it isn't
    // overly useful if we care about performance.
    auto error = cudaMemcpyAsync(dst, src, count, ::cudaMemcpyKind(kind), stream.get());
    if (error != 0)
        throw std::runtime_error(cudaGetErrorString(error));
}

CudaStream::CudaStream() {
    cudaStream_t temp;
    auto error = cudaStreamCreate(&temp);
    if (error != 0)
        throw std::runtime_error(cudaGetErrorString(error));
    _stream = std::shared_ptr<CUstream_st>(temp, [](CUstream_st* p) { cudaStreamDestroy(p); });
}

void CudaStream::sync() const {
    // A cuda_stream_t = 0 (i.e. nullptr) represents the default stream.
    // Call cudaDeviceSyncronize in that case.
    if (_stream.get() == nullptr)
        cudaDeviceSynchronize();
    else
        cudaStreamSynchronize(_stream.get());
}

}  // namespace cuda
