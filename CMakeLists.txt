# SPDX-License-Identifier: BSD-3-Clause

project (exrperf)

find_package(Threads REQUIRED)

add_subdirectory(ext/openexr)
add_executable(exrperf src/main/cpp/main.cpp)
target_include_directories(exrperf PRIVATE ext/cxxopts)
target_link_libraries(exrperf OpenEXR::OpenEXR Threads::Threads)

if(WIN32 AND (BUILD_SHARED_LIBS OR OPENEXR_BUILD_BOTH_STATIC_SHARED))
  target_compile_definitions(exrperf PRIVATE OPENEXR_DLL)
endif()