#include <gtest/gtest.h>
#include <kernels/cuda/wrapper.hpp>
#include <kernels/cuda/cuda_adder.hpp>

#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>
#include <random>
#include <limits>
#include <type_traits>

using test_types_alloc = ::testing::Types<char, int32_t, float, std::complex<float> >;
using test_types_memcpy = ::testing::Types<char, int32_t, float >;
using test_adder = ::testing::Types<int32_t >;

template <class Type>
class GPUAllocFixture : public ::testing::Test {
   public:
    size_t input_size = 1000;
};

template <class Type>
class GPUMemcpy : public ::testing::Test {
  public:
   size_t input_size = 1000;
};

template <class Type>
class GPUAdder : public ::testing::Test {
  public:
   size_t input_size = 1000;
};

TYPED_TEST_SUITE(GPUAllocFixture, test_types_alloc);
TYPED_TEST_SUITE(GPUMemcpy, test_types_memcpy);
TYPED_TEST_SUITE(GPUAdder, test_adder);

/**
 * @test cuda::vector
 * @brief This test checks the cuda::vector constructor, resize,
 * & basic operator functionality
 */
TYPED_TEST(GPUAllocFixture, unified_vector) {
    // Call constructures & resize testing
    cuda::vector<TypeParam> empty;
    cuda::vector<TypeParam> full(this->input_size);
    full.resize(2 * full.size());
    full = cuda::vector<TypeParam>(this->input_size, TypeParam(0));
}

/**
 * @test cuda::device_vector
 * @brief This test checks the cuda::device_vector constructor, resize,
 * & basic operator functionality
 */
TYPED_TEST(GPUAllocFixture, device_vector) {
    // Make sure that we can call various constructors and resize.
    cuda::device_vector<TypeParam> empty;
    cuda::device_vector<TypeParam> full(this->input_size);
    full.resize(2 * full.size());
    full = cuda::device_vector<TypeParam>(this->input_size, TypeParam(0));
}

/**
 * @test cuda::dma_vector
 * @brief This test checks the cuda::dma_vector constructor, resize,
 * & basic operator functionality
 */
TYPED_TEST(GPUAllocFixture, dma_vector) {
    // Make sure that we can call various constructors and resize.
    cuda::dma_vector<TypeParam> empty;
    cuda::dma_vector<TypeParam> full(this->input_size);
    full.resize(2 * full.size());
    full = cuda::dma_vector<TypeParam>(this->input_size, TypeParam(0));
}

/**
 * @test cuda::memcpy, std::vector, cuda::vector
 * @brief This test copies host data to the device, then copies
 * data from the device back to the host.
 * 
 * Host memory allocated with `std::vector`, device memory allocated with `cuda::vector`.
 * Data transfer accomplished with `cuda::memcpy`
 */
TYPED_TEST(GPUMemcpy, vector_memcpy) {
    // Allocate host vectors
    std::vector<TypeParam> host1_vector(this->input_size);
    std::vector<TypeParam> host2_vector(this->input_size);

    // Generate random input values
    TypeParam upper_bound = std::numeric_limits<TypeParam>::max();
    TypeParam lower_bound = std::numeric_limits<TypeParam>::min();
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;

    // Set generated data on host vector
    for (size_t i = 0; i < this->input_size; i++) {
        host1_vector[i] = static_cast<TypeParam>(unif(re));
    }

    // Allocate device vectors
    cuda::vector<TypeParam> device_vector(this->input_size);

    // Copy host1 to Device
    cuda::CudaStream stream;
    cuda::memcpy(device_vector.data(), host1_vector.data(), host1_vector.size() * sizeof(TypeParam), stream);

    // Copy Device to host2
    cuda::memcpy(host2_vector.data(), device_vector.data(), device_vector.size() * sizeof(TypeParam), stream);

    // Compare results
    stream.sync();
    const double atol = 1e-5;
    for (size_t index = 0; index < host1_vector.size(); index++) {
        EXPECT_NEAR(host1_vector[index], host2_vector[index], atol);
    }
}

/**
 * @test cuda::memcpy, cuda::dma_vector, cuda::vector
 * @brief This test copies host data to the device, then copies
 * data from the device back to the host.
 * 
 * Host memory allocated with `cuda::dma_vector`, device memory allocated with `cuda::vector`.
 * Data transfer accomplished with `cuda::memcpy`
 */
TYPED_TEST(GPUMemcpy, dma_memcpy) {
    // Allocate host vectors
    cuda::dma_vector<TypeParam> host1_vector(this->input_size);
    cuda::dma_vector<TypeParam> host2_vector(this->input_size);

    // Generate random input values
    TypeParam upper_bound = std::numeric_limits<TypeParam>::max();
    TypeParam lower_bound = std::numeric_limits<TypeParam>::min();
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;

    // Set generated data on host vector
    for (size_t i = 0; i < this->input_size; i++) {
        host1_vector[i] = static_cast<TypeParam>(unif(re));
    }

    // Allocate device vectors
    cuda::vector<TypeParam> device_vector(this->input_size);

    // Copy host1 to Device
    cuda::CudaStream stream;
    cuda::memcpy(device_vector.data(), host1_vector.data(), host1_vector.size() * sizeof(TypeParam), stream);

    // Copy Device to host2
    cuda::memcpy(host2_vector.data(), device_vector.data(), device_vector.size() * sizeof(TypeParam), stream);

    // Compare results
    stream.sync();
    const double atol = 1e-5;
    for (size_t index = 0; index < host1_vector.size(); index++) {
        EXPECT_NEAR(host1_vector[index], host2_vector[index], atol);
    }
}

/**
 * @test cuda::memcpy, cuda::dma_vector, cuda::device_vector
 * @brief This test copies host data to the device, then copies
 * data from the device back to the host.
 * 
 * Host memory allocated with `cuda::dma_vector`, device memory allocated with `cuda::device_vector`.
 * Data transfer accomplished with `cuda::memcpy`
 */
TYPED_TEST(GPUMemcpy, device_memcpy) {
    // Allocate host vectors
    cuda::dma_vector<TypeParam> host1_vector(this->input_size);
    cuda::dma_vector<TypeParam> host2_vector(this->input_size);

    // Generate random input values
    TypeParam upper_bound = std::numeric_limits<TypeParam>::max();
    TypeParam lower_bound = std::numeric_limits<TypeParam>::min();
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;

    // Set generated data on host vector
    for (size_t i = 0; i < this->input_size; i++) {
        host1_vector[i] = static_cast<TypeParam>(unif(re));
    }

    // Allocate device vectors
    cuda::device_vector<TypeParam> device_vector(this->input_size);

    // Copy host1 to Device
    cuda::CudaStream stream;
    cuda::memcpy(device_vector.data(), host1_vector.data(), host1_vector.size() * sizeof(TypeParam), stream);

    // Copy Device to host2
    cuda::memcpy(host2_vector.data(), device_vector.data(), host2_vector.size() * sizeof(TypeParam), stream);

    // Compare results
    stream.sync();
    const double atol = 1e-5;
    for (size_t index = 0; index < host1_vector.size(); index++) {
        EXPECT_NEAR(host1_vector[index], host2_vector[index], atol);
    }
}

/**
 * @test cuda::kernel:adder
 * @brief This tests the Adder kernel
 */
TYPED_TEST(GPUAdder, adder_kernel) {
    // Allocate host vectors
    std::vector<TypeParam> host_input1_vector(this->input_size);
    std::vector<TypeParam> host_input2_vector(this->input_size);
    std::vector<TypeParam> host_output_vector(this->input_size);
    std::vector<TypeParam> out_check(this->input_size);

    // Set generated data on host input & output_check vectors
    TypeParam gen_data = -(static_cast<TypeParam>((this->input_size) / 2));
    for (size_t i = 0; i < this->input_size; i++) {
        host_input1_vector[i] = gen_data;
        host_input2_vector[i] = gen_data + 1;
        out_check[i] = host_input1_vector[i] + host_input2_vector[i];
	gen_data++;
    }

    // Allocate device vectors
    cuda::device_vector<TypeParam> device_input1_vector(this->input_size);
    cuda::device_vector<TypeParam> device_input2_vector(this->input_size);
    cuda::device_vector<TypeParam> device_output_vector(this->input_size);

    // Copy host vectors to Device
    cuda::CudaStream stream;
    cuda::memcpy(device_input1_vector.data(), host_input1_vector.data(), host_input1_vector.size() * sizeof(TypeParam), stream);
    cuda::memcpy(device_input2_vector.data(), host_input2_vector.data(), host_input2_vector.size() * sizeof(TypeParam), stream);

    // Execute kernel
    auto params = cuda::default_kernel_params(host_input1_vector.size());
    cuda::kernels::vector_add(device_output_vector.data(), device_output_vector.size(),
		              device_input1_vector.data(), device_input1_vector.size(),
			      device_input2_vector.data(), device_input2_vector.size(),
			      params,
			      stream);

    // Copy the result on the GPU to the Host
    cuda::memcpy(host_output_vector.data(), device_output_vector.data(), host_output_vector.size() * sizeof(TypeParam), stream);
    stream.sync();

    // Compare results
    const double atol = 1e-5;
    for (size_t index = 0; index < host_output_vector.size(); index++) {
        EXPECT_NEAR(host_output_vector[index], out_check[index], atol);
    }
}
