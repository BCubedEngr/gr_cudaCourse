find_package(GTest)

try_run(run_result compile_result ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/bin/check_gpu.cu
        RUN_OUTPUT_VARIABLE have_gpu COMPILE_OUTPUT_VARIABLE rutro)
message(STATUS "${have_gpu}")
if (NOT ${run_result})
  message(STATUS "Running Unit tests")
  SET(unittest_files
          wrapper_alloc_test.cpp
	  cuda_tuner_test.cpp
	  cuda_filter_dec_test.cpp
     )
else()
  message(STATUS "Skipping unit tests")
endif()

foreach(test_file IN ITEMS ${unittest_files})
    get_filename_component(name ${test_file} NAME_WE)
    add_executable(${name} unittests.cpp ${test_file})
    # We don't want to use the CUDA compiler here, because we need to be able to
    # link with just gcc
    set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(${name} gtest gr_cuda_course)
    add_test(
        NAME ${name}
        # Pass in a directory for where the test files are.
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${name} -- ${CMAKE_BINARY_DIR} )
endforeach()

SET(python_files
        cudaTunerTestData.py
    )

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_tuner_input.fc32 ${CMAKE_CURRENT_BINARY_DIR}/cuda_tuner_output.fc32
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cudaTunerTestData.py
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cudaTunerTestData.py
    COMMENT "Generating Cuda Tune test data with Python Script ..."
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tune_filter_dec_out.fc32 ${CMAKE_CURRENT_BINARY_DIR}/tune_filter_out.fc32 ${CMAKE_CURRENT_BINARY_DIR}/low_pass_filter_taps.f32
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cudaFilterDecimateTestData.py
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cudaTunerTestData.py
    COMMENT "Generating Cuda Filter test data with Python Script ..."
)
add_custom_target(
    generate_test_data ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cuda_tuner_output.fc32 ${CMAKE_CURRENT_BINARY_DIR}/cuda_tuner_input.fc32 ${CMAKE_CURRENT_BINARY_DIR}/tune_filter_dec_out.fc32 ${CMAKE_CURRENT_BINARY_DIR}/tune_filter_out.fc32 ${CMAKE_CURRENT_BINARY_DIR}/low_pass_filter_taps.f32
)

