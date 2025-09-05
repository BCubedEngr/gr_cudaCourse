#ifndef INCLUDE_KERNELS_CUDA_WRAPPER_HPP_
#define INCLUDE_KERNELS_CUDA_WRAPPER_HPP_

/******************************************************************************
 * Create allocators for cuda buffers.  This allows us to have an std::vector
 * with cuda data, rather than managing our own memory.
 *
 * This file contains custom allocators and other code needed to be able to use
 * an std::vector that allocates memory in cuda memory.  The most commonly used
 * class here would be the b3::cuda::vector<T>/
 *
 * Note: We do need to be careful, cuda memory is not exactly the same as CPU
 * memory.  For example, on the host I can write `data[i] = 4.0;`.  If the
 * buffer is allocated on the GPU, then we can't directly access individual
 * elements.  The above code would cause a segfault (or other program error).
 * We also can't construct objects in device memory, such as std::complex<float>.
 *
 * We need to operate on cuda memory in blocks.  We can use b3::cuda::copy to
 * move data between the host and gpu.
 *
 * Note 2: Because of these limitations, not all constructors work for these vectors
 * and not all vector functions work.  It is still an extremely useful construct.  At
 * some point it may be worthwhile to write our own cuda vector class rather than
 * using std::vector, but that would be a lot of effort to write and test.
 */

#include <cstddef>
#include <vector>
#include <complex>
#include <cstdint>
#include <memory>

struct CUstream_st;

namespace cuda {

static constexpr float TWO_PI = 2.0 * 3.1415927f;

// Used by cuda::memcpy.  Generally not useful to developers.
enum cudaMemcpyKind {
    cudaMemcpyHostToHost,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
    cudaMemcpyDefault
};

// Hide the cuda stuff in here.  They wrap the cuda functions.
// They will be compiled while the header will be free from cuda so that we can
// use it anywhere.
int get_cuda_compiled_version();
int get_cuda_driver_version();

// Hide the cuda stuff in here.  They wrap the cuda functions.
// They will be compiled while the header will be free from cuda so that we can
// use it anywhere.
namespace detail {
void* _unified_allocate(size_t n);
void* _device_allocate(size_t n);
void* _dma_allocate(size_t n);
void _deallocate(void* p);
void _dma_deallocate(void* p);
}  // namespace detail

/**
 * Allocate memory in cuda unified memory.  We should be able to use this
 * just like a normal std::vector.  However, I have found that sometimes I
 * get a memory error when doing this.
 */
template <class T>
class unified_alloc {
   public:
    using value_type = T;
    auto allocate(size_t n) {
        return static_cast<T*>(detail::_unified_allocate(n * sizeof(T)));
    }
    // We can't construct individual objects in device memory.  It should work in
    // unified memory, but sometimes it doesn't.  So we don't construct
    // anything when initializing.  This means
    // that some constructors like cuda::vector<T>(10, 5) won't work as expected.
    template <class U, class... Args>
    void construct(U*, Args&&...) {}
    auto deallocate(T* p, size_t) {
        detail::_deallocate(p);
    }
};

/**
 * Allocate memory in cuda device memory.
 *
 * You cannot access this memory directly outside of a cuda kernel.  In host
 * code, you have to use b3::cuda::copy to copy the data into host memory.
 * This is much more limited than unified memory, without too many benefits.
 */
template <class T>
class device_alloc {
   public:
    using value_type = T;
    auto allocate(size_t n) {
        return static_cast<T*>(detail::_device_allocate(n * sizeof(T)));
    }
    // We can't construct individual objects in device memory.  This means
    // that some constructors like device_vector<T>(10, 5) won't work as expected.
    template <class U, class... Args>
    void construct(U*, Args&&...) {}

    auto deallocate(T* p, size_t) {
        detail::_deallocate(p);
    }
};

/**
 * Allocate pinned host memory.
 *
 * This allocator pins host memory in a way that can improve performance when
 * transferring large chunks of data between the host and the GPU.  Unlike the
 * other buffers, these are normal host vectors, and all std::vector functions
 * work just fine with them.
 *
 * These buffers use dma.
 */
template <class T>
class dma_alloc {
   public:
    using value_type = T;
    auto allocate(size_t n) {
        return static_cast<T*>(detail::_dma_allocate(n * sizeof(T)));
    }
    auto deallocate(T* p, size_t) {
        detail::_dma_deallocate(p);
    }
};


/*
RAII Wrapper around a cudaStream_t.  Note that it is really a pointer to a
CUstream_t and we use a smart pointer so that it is destroyed when everyone
is done with it.

A cudaStream_t allows us to launch multiple cuda kernels in parallel.  This is
important especially in multithreaded applications because we don't want to
wait on one thread to finish before another can start.  All cuda kernels should
have a stream associated with them.  If we have multiple kernels that are running
in the same thread - in order, they should share the same cuda stream.
*/
struct CudaStream {
    CudaStream();
    CudaStream(nullptr_t)
        : _stream(nullptr) {}
    CUstream_st* get() const noexcept {
        return _stream.get();
    }
    /**
     * Block until all kernels on this stream have finished.
     */
    void sync() const;

   private:
    std::shared_ptr<CUstream_st> _stream = nullptr;
};


/**
 * @brief Cuda Version of memcpy.
 * @param dst The output address to start copying into.
 * @param src The input address to start copying from.
 * @param count The number of bytes to copy.
 * @param stream The cuda stream that the work will be done on.  Default nullptr(the default stream)
 * @param kind The direction of the copy.  Default cudaMemcpyDefault (usually does what is wanted)
 */
void memcpy(void* dst,
            const void* src,
            size_t count,
            const CudaStream& stream = nullptr,
            cudaMemcpyKind kind = cudaMemcpyDefault);


/**
 * We have to configure a few parameters when we launch a kernel.  The block
 * size tells us how many "threads" to use in a grid block and the grid_size
 * tells us how many of those blocks to use.
 * Struct to configure how a cuda kernel is called.  Although the kernel params
 * can be quite complicated, this struct allows us to cover the 95% use case.
 * These parameters map directly to the cuda documentation.
 */
struct kernel_params {
    uint32_t grid_size = 0;
    uint32_t block_size = 0;
    // These are only useful if we want to have a multi dimension kernel.  This
    // is rare but happens
    uint32_t grid_size_y = 1;
    uint32_t block_size_y = 1;
};

/**
 * @brief Pick a good set of kernel params.
 * If you don't like it, you can write your own function for determining the values.
 * @param problem_size The number of elements to be processed.
 */
inline kernel_params default_kernel_params(size_t problem_size, uint32_t block_size = 256) {
    // Empirically, this works pretty well for a wide range of problems.
    auto grid_size = std::ceil(static_cast<double>(problem_size) / static_cast<double>(block_size));
    return {static_cast<uint32_t>(grid_size), block_size};
}


// These are important so that we can copy/transfer allocators.
template <class T, class U>
bool operator==(const unified_alloc<T>&, const unified_alloc<U>&) {
    return true;
}
template <class T, class U>
bool operator!=(const unified_alloc<T>&, const unified_alloc<U>&) {
    return false;
}

template <class T, class U>
bool operator==(const device_alloc<T>&, const device_alloc<U>&) {
    return true;
}
template <class T, class U>
bool operator!=(const device_alloc<T>&, const device_alloc<U>&) {
    return false;
}

template <class T, class U>
bool operator==(const dma_alloc<T>&, const dma_alloc<U>&) {
    return true;
}
template <class T, class U>
bool operator!=(const dma_alloc<T>&, const dma_alloc<U>&) {
    return false;
}

// Create names for the vectors to make it easier to use.
// b3::cuda::vector
template <class T>
using vector = std::vector<T, unified_alloc<T>>;

// b3::cuda::device_vector
template <class T>
using device_vector = std::vector<T, device_alloc<T>>;

// b3::cuda::dma_vector
template <class T>
using dma_vector = std::vector<T, dma_alloc<T>>;

}  // namespace cuda

#endif  // INCLUDE_KERNELS_CUDA_WRAPPER_HPP_
