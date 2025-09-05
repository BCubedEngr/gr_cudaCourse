list(APPEND cuda_sources
    wrapper.cu
    cuda_tune.cu
    cuda_adder.cu
    cuda_filter.cu
)

add_library(gr_cuda_course SHARED "${cuda_sources}")

target_include_directories(gr_cuda_course PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

install(TARGETS gr_cuda_course DESTINATION lib )
install(TARGETS ${PROJECT_NAME} EXPORT gnuradio-cudaCourse-export)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
