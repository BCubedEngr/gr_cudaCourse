find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_CUDACOURSE gnuradio-cudaCourse)

FIND_PATH(
    GR_CUDACOURSE_INCLUDE_DIRS
    NAMES gnuradio/cudaCourse/api.h
    HINTS $ENV{CUDACOURSE_DIR}/include
        ${PC_CUDACOURSE_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_CUDACOURSE_LIBRARIES
    NAMES gnuradio-cudaCourse
    HINTS $ENV{CUDACOURSE_DIR}/lib
        ${PC_CUDACOURSE_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-cudaCourseTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_CUDACOURSE DEFAULT_MSG GR_CUDACOURSE_LIBRARIES GR_CUDACOURSE_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_CUDACOURSE_LIBRARIES GR_CUDACOURSE_INCLUDE_DIRS)
