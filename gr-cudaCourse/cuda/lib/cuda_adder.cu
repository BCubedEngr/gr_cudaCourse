#include <kernels/cuda/wrapper.hpp>
#include <stdexcept>
#include "./cuda_util.cuh"

using namespace std; // Require when including "cuda/std/complex"

// Wrappers around cuda allocation functions.  None of these functions are
// meant to be called directly by user code.

namespace cuda {
namespace kernels {

/**
 * @brief Add two arrays together.
 * @param out The output data
 * @param in1 The first input array
 * @param in2 The second input array
 */
__global__ void adder_kernel(int32_t* out, int32_t* in1, int32_t* in2, size_t size) {
    for (size_t i = get_start(); i < size; i += get_stride()) {
        out[i] = in1[i] + in2[i];
    }
}

void vector_add(int32_t* out, size_t out_size,
                int32_t* in1, size_t in1_size,
                int32_t* in2, size_t in2_size,
                const kernel_params& params,
                CudaStream& stream) {
    // Assume same input/output size
    if (in1_size != out_size) {
        throw runtime_error("In size and out size don't match");
    }
    if (in1_size != in2_size) {
        throw runtime_error("In sizes don't match");
    }
    adder_kernel<<<params.grid_size, params.block_size, 0, stream.get()>>>(out, in1, in2, out_size);
}

} // namespace kernels
} // namespace cuda
