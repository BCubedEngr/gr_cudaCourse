########################################################################
# Project setup
########################################################################

cmake_minimum_required(VERSION 3.8.0)
project(gr_cuda_course LANGUAGES CUDA CXX)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_C_STANDARD 11)
SET(CMAKE_CUDA_STANDARD 17)
SET(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_testing()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads=1")

SET(PROJECT_WARNINGS
    -Werror all-warnings # Convert Warnings to errors
    -Wall
    -Wextra # reasonable and standard
    -Wshadow # warn the user if a variable declaration shadows one from a parent context
    -Wnon-virtual-dtor # warn the user if a class with virtual functions has a non-virtual destructor. This helps
                       # catch hard to track down memory errors
    #-Wold-style-cast # warn for c-style casts
    -Wcast-align # warn for potential performance problem casts
    -Wunused # warn on anything being unused
    -Woverloaded-virtual # warn if you overload (not override) a virtual function
    # Don't do -Wpedantic.  Nvcc generates intermediate code that doesn't pass
    #-Wpedantic # warn if non-standard C++ is used
    -Wconversion # warn on type conversions that may lose data
    -Wsign-conversion # warn on sign conversions
    -Wnull-dereference # warn if a null dereference is detected
    -Wformat=1 # warn on security issues around functions that format output (ie printf)
    -Wno-unknown-pragmas # ignore IDE, GCC/CLANG specific pragmas
    -Wimplicit-fallthrough # Warns when case statements fall-through.)
    -Wmisleading-indentation # warn if indentation implies blocks where blocks do not exist
    -Wduplicated-cond # warn if if / else chain has duplicated conditions
    -Wduplicated-branches # warn if if / else branches have duplicated code
    -Wlogical-op # warn about logical operations being used where bitwise were probably wanted
    -Wuseless-cast # warn if you perform a cast to the same type
    -Wno-maybe-uninitialized # false positives if asan is enabled: https://gcc.gnu.org/bugzilla//show_bug.cgi?id=1056h6
)


########################################################################
# Add subdirectories
########################################################################

add_subdirectory(include/kernels/cuda)
add_subdirectory(lib)
add_subdirectory(bin)
add_subdirectory(test)
