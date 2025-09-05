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
 * @brief Filter and decimate the input array
 * @param out The output data
 * @param in The input data
 * @param size Input/Output array size
 * @param taps Input taps
 * @param dec Decimation Factor
 */
__global__ void decimating_fir_filter_kernel(cuda::std::complex<float>* out, const cuda::std::complex<float>* in, size_t size, const float* taps, size_t taps_size, size_t dec) {

    for (size_t i = get_start(); i < size * dec; i += get_stride()) {
	auto out_i = (i / dec);
	auto taps_start = (i % dec);
	float* value = reinterpret_cast<float*>(&out[out_i]);
	for (size_t j = taps_start; j < taps_size; j += dec) {
            atomicAdd(value, in[out_i * dec + j].real() * taps[j] );
            atomicAdd(value + 1, in[out_i * dec + j].imag() * taps[j] );
	}
    }
}

float decimate_filterN(complex<float>* out, size_t out_size,
                       const complex<float>* in, size_t in_size,
                       const float* taps, size_t taps_size,
                       size_t decimate_factor,
                       const kernel_params& params,
                       CudaStream& stream) {
    if (in_size < (decimate_factor * (out_size - 1)) + taps_size) {
        printf("in_size = %li, out_size = %li, taps_size = %li, D = %li : expected %li < %li",
                               in_size,
                               out_size,
                               taps_size,
                               decimate_factor,
                               in_size,
                               (out_size * (decimate_factor - 1)) + taps_size);
        throw runtime_error("Not enough input data");
    }

    // Convert to gpu pointer types
    cuda::std::complex<float>* g_out = reinterpret_cast<cuda::std::complex<float>* >(out);
    const cuda::std::complex<float>* g_in =  reinterpret_cast<const cuda::std::complex<float>* >(in);
    const float* g_taps = reinterpret_cast<const float* >(taps);

    cudaMemsetAsync(g_out, 0, out_size * sizeof(std::complex<float>), stream.get());
    decimating_fir_filter_kernel<<<params.grid_size, params.block_size, 0, stream.get()>>>(g_out, g_in, out_size, g_taps, taps_size, decimate_factor);

    return out_size;
}

} //namespace kernels
} //namespace cuda
