#include <gtest/gtest.h>
#include <kernels/cuda/wrapper.hpp>
#include <kernels/cuda/cuda_filter.hpp>
#include "./test_helpers.h"

using test_types = ::testing::Types<std::complex<float> >;

template <class Type>
class GPUFilterDec : public ::testing::Test {
};

TYPED_TEST_SUITE(GPUFilterDec, test_types);

/**
 * @test Cuda Filter & Decimate
 * @brief This test executes a Cuda Filter & Decimate Kernel, comparing results
 * with known truth data
 */
TYPED_TEST(GPUFilterDec, cuda_filter_dec) {
    
    // Allocate vectors
    std::string input_pass_file = "unfiltered_in_pass.fc32";
    std::string output_dec1_pass_file = "filtered_dec1_out_pass.fc32";
    std::string low_pass_filter_taps_file = "low_pass_filter_taps.f32";

    std::vector<std::complex<float>> _input_pass_vector;
    cuda::vector<std::complex<float>> output_dec1_pass_vector;
    std::vector<std::complex<float>> output_dec1_pass_check;
    cuda::vector<float> filter_taps;

    // Read in data-files
    TestHelpers::read_data_file(&_input_pass_vector, input_pass_file);
    TestHelpers::read_data_file(&output_dec1_pass_check, output_dec1_pass_file);
    TestHelpers::read_taps_file(&filter_taps, low_pass_filter_taps_file);

    // Reverse order of Filter Taps
    std::reverse(filter_taps.begin(), filter_taps.end());

    // Set size of Cuda Output Vectors
    output_dec1_pass_vector.resize(_input_pass_vector.size());

    // Resize input vectors
    cuda::vector<std::complex<float>> input_pass_vector(_input_pass_vector.size() + 2 * (filter_taps.size() - 1), 0);
    std::copy(_input_pass_vector.begin(), _input_pass_vector.end(), input_pass_vector.begin() + filter_taps.size() - 1);

    // Execute the Kernel
    // Lowpass, signal in passband, Decimation = 1
    cuda::CudaStream stream;
    size_t decimation_factor = 1;
    auto params = cuda::default_kernel_params(input_pass_vector.size());
    cuda::kernels::decimate_filterN(output_dec1_pass_vector.data(), output_dec1_pass_vector.size(),
                                    input_pass_vector.data(), input_pass_vector.size(),
                                    filter_taps.data(), filter_taps.size(),
                                    decimation_factor,
                                    params,
                                    stream);

    // Copy data from Device back to Host memory
    std::vector<std::complex<float>> test_dec1_pass_vector(output_dec1_pass_check.size(), 0);
    cuda::memcpy(test_dec1_pass_vector.data(), output_dec1_pass_vector.data(), output_dec1_pass_vector.size() * sizeof(std::complex<float>), stream);
    stream.sync();

    // Compare results
    double atol = 1e-5;
    double rtol = 1e-5;
    for (size_t i = 0; i < test_dec1_pass_vector.size(); i++) {
         auto max_diff = std::max(atol, rtol * std::abs<float>(output_dec1_pass_check[i]));
         EXPECT_NEAR(test_dec1_pass_vector[i].real(), output_dec1_pass_check[i].real(), max_diff);
         EXPECT_NEAR(test_dec1_pass_vector[i].imag(), output_dec1_pass_check[i].imag(), max_diff);
    }

    // Test with different decimation factors
    auto dec_low = 2;
    auto dec_high = 20;
    params = cuda::default_kernel_params(input_pass_vector.size());
    for (auto dec = dec_low; dec <= dec_high; dec++) {
	// Allocate host memory
	auto outcheck_pass_n = std::vector<std::complex<float>>((_input_pass_vector.size() + dec - 1) / dec);
	for (size_t j = 0, k = 0; j < output_dec1_pass_check.size(); j += dec, k++) {
             outcheck_pass_n[k] = output_dec1_pass_check[j];
	}

	// Allocate device memory
	cuda::vector<std::complex<float>> output_decN_pass_vector(outcheck_pass_n.size(), 0);
        stream.sync();

	// Run the kernel
	cuda::kernels::decimate_filterN(output_decN_pass_vector.data(), output_decN_pass_vector.size(),
			               input_pass_vector.data(), input_pass_vector.size(),
				       filter_taps.data(), filter_taps.size(),
				       dec,
				       params,
				       stream);
	stream.sync();

        // Copy data from Device back to Host memory
	std::vector<std::complex<float>> test_pass_vector(outcheck_pass_n.size(), 0);
        cuda::memcpy(test_pass_vector.data(), output_decN_pass_vector.data(), output_decN_pass_vector.size() * sizeof(std::complex<float>), stream);
	stream.sync();

        // Compare Results
        for (size_t i = 0; i < output_decN_pass_vector.size(); i++) {
             auto max_diff = std::max(atol, rtol * std::abs<float>(outcheck_pass_n[i]));
             EXPECT_NEAR(test_pass_vector[i].real(), outcheck_pass_n[i].real(), max_diff);
             EXPECT_NEAR(test_pass_vector[i].imag(), outcheck_pass_n[i].imag(), max_diff);
        }
    }
}
