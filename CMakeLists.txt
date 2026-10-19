cmake_minimum_required(VERSION 3.18)
project(timestral_cpp LANGUAGES C CXX)

set(TIMESTRAL_VERSION "0.0.1" CACHE STRING "Version string returned by ts::version()")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(TIMESTRAL_BUILD_TESTS  "Build ctest targets" OFF)
option(TIMESTRAL_BUILD_CLI    "Build timestral-cli" ON)
option(TIMESTRAL_SHARED       "Build libtimestral as shared" OFF)
option(TIMESTRAL_GGML_CUDA    "Forward GGML_CUDA" OFF)
option(TIMESTRAL_GGML_METAL   "Forward GGML_METAL" OFF)
option(TIMESTRAL_GGML_VULKAN  "Forward GGML_VULKAN" OFF)

# TIMESTRAL_GGML_<X>=ON turns the ggml backend on. When it is OFF, a direct
# -DGGML_<X>=ON from the caller is kept; ggml's own default applies otherwise.
foreach(_be CUDA METAL VULKAN)
  if(TIMESTRAL_GGML_${_be})
    set(GGML_${_be} ON CACHE BOOL "" FORCE)
  endif()
endforeach()

# Native code generation and tinyBLAS SGEMM on by default; a caller can override.
if(NOT DEFINED GGML_NATIVE)
  set(GGML_NATIVE ON CACHE BOOL "ggml: optimize the build for the current system" FORCE)
endif()
if(NOT DEFINED GGML_LLAMAFILE)
  set(GGML_LLAMAFILE ON CACHE BOOL "ggml: use LLAMAFILE (tinyBLAS SGEMM)" FORCE)
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/CMakeLists.txt")
  message(FATAL_ERROR "third_party/ggml is missing: run `git submodule update --init`")
endif()
# Static ggml with PIC: tests and the future shared library carry ggml inside, so `ldd`
# shows no external libggml.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(third_party/ggml)

set(TIMESTRAL_SRC
    src/common.cpp
    src/backend.cpp
    src/model_loader.cpp
    src/scaler.cpp
    src/layers.cpp
    src/encoder.cpp
    src/head.cpp
    src/predict_block.cpp
    src/rope.cpp)

# PyTorch materializes multiply/add separately at the scaler float32 boundaries.

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(src/scaler.cpp src/layers.cpp src/head.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

if(TIMESTRAL_SHARED)
  add_library(timestral SHARED ${TIMESTRAL_SRC})
else()
  add_library(timestral STATIC ${TIMESTRAL_SRC})
endif()
set_target_properties(timestral PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(timestral PUBLIC include PRIVATE src)
target_compile_definitions(timestral PRIVATE TIMESTRAL_VERSION="${TIMESTRAL_VERSION}")
target_link_libraries(timestral PUBLIC ggml)

if(TIMESTRAL_BUILD_CLI AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/cli/CMakeLists.txt")
  add_subdirectory(examples/cli)
endif()
if(TIMESTRAL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
