#ifndef TEST_TEST_HELPERS_H_
#define TEST_TEST_HELPERS_H_

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <fstream>
#include <complex>
#include <kernels/cuda/wrapper.hpp>

class TestHelpers {
    public:
        static void read_data_file(std::vector<std::complex<float>>* samples, std::string file) {
            std::ifstream input(file);
            if (!input.is_open()) {
                std::cout << "COULD NOT OPEN FILE" << std::endl;
                EXPECT_TRUE(false);
            }

            input.seekg(0, std::ios::end);
            size_t fileSize = input.tellg();
            input.seekg(0, std::ios::beg);

            size_t numSamples = fileSize / sizeof(std::complex<float>);
            samples->resize(numSamples);
            input.read(reinterpret_cast<char*>(samples->data()), numSamples * sizeof(std::complex<float>));

            input.close();
        }

        static void read_data_file(cuda::vector<std::complex<float>>* samples, std::string file) {
            std::ifstream input(file);
            if (!input.is_open()) {
                std::cout << "COULD NOT OPEN FILE" << std::endl;
                EXPECT_TRUE(false);
            }

            input.seekg(0, std::ios::end);
            size_t fileSize = input.tellg();
            input.seekg(0, std::ios::beg);

            size_t numSamples = fileSize / sizeof(std::complex<float>);
            samples->resize(numSamples);
            input.read(reinterpret_cast<char*>(samples->data()), numSamples * sizeof(std::complex<float>));

            input.close();
        }

        static void read_taps_file(cuda::vector<float>* samples, std::string file) {
            std::ifstream input(file);
            if (!input.is_open()) {
                std::cout << "COULD NOT OPEN FILE" << std::endl;
                EXPECT_TRUE(false);
            }

            input.seekg(0, std::ios::end);
            size_t fileSize = input.tellg();
            input.seekg(0, std::ios::beg);

            size_t numSamples = fileSize / sizeof(float);
            samples->resize(numSamples);
            input.read(reinterpret_cast<char*>(samples->data()), numSamples * sizeof(float));

            input.close();
        }


        static void write_data_file(cuda::vector<std::complex<float>> samples, std::string file) {
            std::ofstream output(file, std::ios::out | std::ios::binary);
            if (!output.is_open()) {
                std::cout << "COULD NOT OPEN FILE" << std::endl;
                EXPECT_TRUE(false);
            }
            std::vector<std::complex<float>> _samples;
	    for (size_t i = 0; i < samples.size(); i++) {
		//std::cout << " REAL: " << samples[i].real() << " IMAG: " << samples[i].imag();
                _samples.push_back(samples[i]);
	    }
	    output.write(reinterpret_cast<const char*>(_samples.data()), _samples.size() * sizeof(std::complex<float>));
            output.close();
        }

        static void write_data_file(std::vector<std::complex<float>> samples, std::string file) {
            std::ofstream output(file, std::ios::out | std::ios::binary);
            if (!output.is_open()) {
                std::cout << "COULD NOT OPEN FILE" << std::endl;
                EXPECT_TRUE(false);
            }

            output.write(reinterpret_cast<const char*>(samples.data()), samples.size() * sizeof(std::complex<float>));
            output.close();
        }

	static void expect_close(std::vector<std::complex<float>> x, std::vector<std::complex<float>> y, double rtol = 1e-5, double atol = 1e-5) {
            for (size_t i = 0; i < x.size(); i++) {
                auto max_diff = std::max(atol, rtol * std::abs<float>(y[i]));
                EXPECT_NEAR(x[i].real(), y[i].real(), max_diff);
		EXPECT_NEAR(x[i].imag(), y[i].imag(), max_diff);
	    }

	}

};

#endif  // TEST_TEST_HELPERS_H_
