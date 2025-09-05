#include <kernels/cuda/wrapper.hpp>
#include <stdexcept>
#include <complex>
#include <cuda/std/complex>
#include "./cuda_util.cuh"

using namespace std; // Require when including "cuda/std/complex"

// Wrappers around cuda allocation functions.  None of these functions are
// meant to be called directly by user code.

namespace cuda {
namespace kernels {

/**
 * @brief Tune the input array
 * @param out The output data
 * @param in The input data
 * @param size Input/Output array size
 * @param freq Input Frequency ( (Frequency (Hz)) / (SampleRate (Samples/Sec)) ) * 2 * PI
 * @param phase Input Phase (Phase (radians))
 */
__global__ void tuner_kernel(cuda::std::complex<float>* out, const cuda::std::complex<float>* in, size_t size, float freq, float phase) {
    for (size_t i = get_start(); i < size; i =+ get_stride()) {
        // Compute input arg
        auto carg = freq * i + phase;

        // Get real & imag components
        cuda::std::complex<float> _out;
        float* val = reinterpret_cast<float*>(&_out);
        sincosf(carg, val + 1, val);

        // Generate output
        out[i] = in[i] * _out;
    }
}

float vector_tune(complex<float>* out, size_t out_size,
                  const complex<float>* in, size_t in_size,
                  float freq,
                  float phase,
                  const kernel_params& params,
                  CudaStream& stream) {
    if (in_size != out_size) {
        throw runtime_error("In size and out size don't match");
    }

    // Convert to gpu pointer types
    cuda::std::complex<float>* g_out = reinterpret_cast<cuda::std::complex<float>* >(out);
    const cuda::std::complex<float>* g_in =  reinterpret_cast<const cuda::std::complex<float>* >(in);

    tuner_kernel<<<params.grid_size, params.block_size, 0, stream.get()>>>(g_out, g_in, out_size, freq, phase);

    return std::fmod(freq * in_size + phase, cuda::TWO_PI);
}

} //namespace kernels
} //namespace cuda
