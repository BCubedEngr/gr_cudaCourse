#include <kernels/cuda/wrapper.hpp>
#include <complex>

namespace cuda {
namespace kernels {

/**
 * @brief Tune the input array
 * @param out The output data
 * @param in The input data
 * @param size Input/Output array size
 * @param freq Input Frequency (Frequency (Hz) * 2 * PI) / (SampleRate (Samples/Sec))
 * @param phase Input Phase (Phase (radians))
 */
float vector_tune(std::complex<float>* out, size_t out_size,
                  const std::complex<float>* in, size_t in_size,
                  float freq,
                  float phase,
                  const kernel_params& params,
                  CudaStream& stream);

} // namespace kernels
} // namespace cuda
