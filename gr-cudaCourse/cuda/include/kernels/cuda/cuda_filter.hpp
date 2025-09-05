#include <kernels/cuda/wrapper.hpp>
#include <complex>

namespace cuda {
namespace kernels {

float decimate_filterN(std::complex<float>* out, size_t out_size,
                       const std::complex<float>* in, size_t in_size,
                       const float* taps, size_t taps_size,
                       size_t decimate_factor,
                       const kernel_params& params,
                       CudaStream& stream);

} // namespace kernels
} // namespace cuda
