#include <gtest/gtest.h>
#include <kernels/cuda/wrapper.hpp>
#include <kernels/cuda/cuda_tune.hpp>
#include "./test_helpers.h"

using test_types = ::testing::Types<std::complex<float> >;

template <class Type>
class GPUTuner : public ::testing::Test {
   public:
    size_t input_size = 1000;
};

TYPED_TEST_SUITE(GPUTuner, test_types);

/**
 * @test Cuda Tuner
 * @brief This test executes a Cuda Tuner Kernel, comparing results
 * with known truth data
 */
TYPED_TEST(GPUTuner, cuda_tuner) {
    // Allocate host vectors
    std::string inputFile = "cuda_tuner_input.fc32"; // 32k Samples @ 32k Samples per Second; Cosine Waveform; 1kHz Frequency; 1.0 Amplitude; 0.0 Phase (radians)
    std::string outputFile = "cuda_tuner_output.fc32"; // 32k Samples @ 32k Samples per Second; Cosine Waveform; 2kHz Frequency; 1.0 Amplitude; 0.741 Phase (radians)
    
    cuda::vector<std::complex<float>> input_vector;
    cuda::vector<std::complex<float>> output_vector;
    std::vector<std::complex<float>> out_check;

    TestHelpers::read_data_file(&input_vector, inputFile);
    TestHelpers::read_data_file(&out_check, outputFile);
    output_vector.resize(out_check.size());

    cuda::CudaStream stream;

    // Complete the tuning operation over two kernel executions
    size_t N = input_vector.size() / 2;
    float sampleRate = 32000.0;
    float freq = 1000;
    float phase = 0.741;
    float fArg = (freq / sampleRate) * cuda::TWO_PI;
    auto params = cuda::default_kernel_params(input_vector.size());
    phase = cuda::kernels::vector_tune(output_vector.data(), N,
	  	                       input_vector.data(), N,
			               fArg,
			               phase,
			               params,
			               stream);
    stream.sync();
 
    cuda::kernels::vector_tune(&output_vector[N], N,
                               &input_vector[N], N,
                               fArg,
                               phase,
                               params,
                               stream);
    stream.sync();
    
    // Compare results
    double atol = 5e-4;
    double rtol = 5e-4;
    for (size_t i = 0; i < output_vector.size(); i++) {
        auto max_diff = std::max(atol, rtol * std::abs<float>(out_check[i]));
        EXPECT_NEAR(output_vector[i].real(), out_check[i].real(), max_diff);
        EXPECT_NEAR(output_vector[i].imag(), out_check[i].imag(), max_diff);
    }

                std::ofstream output("wot_out.fc32", std::ios::out | std::ios::binary);

            output.write(reinterpret_cast<const char*>(output_vector.data()), output_vector.size() * sizeof(std::complex<float>));
            output.close();

}

/**
 * @test GRC Cuda Tuner
 * @brief This test compares the results from the cudaTunerTest.py
 * with known truth data
 */
TYPED_TEST(GPUTuner, grc_cuda_tuner) {

}
