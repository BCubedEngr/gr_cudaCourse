#include <kernels/cuda/wrapper.hpp>

namespace cuda {
namespace kernels {

/**
 * @brief Add two vectors on the GPU.
 * @param out output buffer
 * @param in1 first input buffer.
 * @param in2 second input buffer.
 * @param params The kernel parameters to use.
 * @param stream The CudaStream to run the kernel on.
 */
void vector_add(int32_t* out, size_t out_size,
                int32_t* in1, size_t in1_size,
                int32_t* in2, size_t in2_size,
                const kernel_params& params,
                CudaStream& stream);

} // namespace kernels
} // namespace cuda
