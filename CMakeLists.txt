cmake_minimum_required(VERSION 3.28)
project(Gyre VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GYRE_BUILD_TESTS "Build tests" ON)
option(GYRE_ENABLE_TUI "Build gyre-tui (FTXUI)" ON)
option(GYRE_BUILD_EXAMPLE_BINS "Build gyre-charlm / gyre-ga-onemax example binaries" ON)
option(GYRE_ENABLE_VULKAN "Vulkan backend (future; unimplemented)" OFF)
option(GYRE_ENABLE_ASAN "Sanitize tests" OFF)
option(GYRE_ENABLE_OPENMP "Multi-thread CPU kernels (matmul/bmm)" ON)

add_library(gyre STATIC
  src/gyre/log.cpp
  src/gyre/device.cpp
  src/gyre/tensor.cpp
  src/gyre/storage.cpp
  src/gyre/io/safetensors.cpp
  src/gyre/io/compress_probe.cpp
  src/gyre/io/wpack.cpp
  src/gyre/io/codec.cpp
  src/gyre/ops.cpp
  src/gyre/rng.cpp
  src/gyre/module.cpp
  src/gyre/nn/layers.cpp
  src/gyre/nn/tokenize.cpp
  src/gyre/nn/bpe.cpp
  src/gyre/nn/tiktoken.cpp
  src/gyre/nn/unigram.cpp
  src/gyre/nn/pretok.cpp
  src/gyre/nn/sp_model.cpp
  src/gyre/nn/attention.cpp
  src/gyre/nn/transformer.cpp
  src/gyre/nn/gqa.cpp
  src/gyre/nn/moe.cpp
  src/gyre/nn/grok.cpp
  src/gyre/optim.cpp
  src/gyre/ga/population.cpp
  src/gyre/checkpoint.cpp
  src/gyre/data.cpp
  src/gyre/train/loop.cpp
  src/gyre/adapt/agent.cpp
  src/gyre/export/onnx.cpp
)
target_include_directories(gyre PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
  )
  set(JSON_BuildTests OFF CACHE INTERNAL "")
  set(JSON_Install OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(nlohmann_json)
endif()
# Header-only: include path only so install(EXPORT) does not require exporting nlohmann.
if(TARGET nlohmann_json::nlohmann_json)
  get_target_property(_nlohmann_inc nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories(gyre PRIVATE ${_nlohmann_inc})
elseif(DEFINED nlohmann_json_SOURCE_DIR)
  target_include_directories(gyre PRIVATE ${nlohmann_json_SOURCE_DIR}/include)
endif()
target_include_directories(gyre PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/gyre)

if(MSVC)
  target_compile_options(gyre PRIVATE /W4 /permissive-)
else()
  target_compile_options(gyre PRIVATE -Wall -Wextra)
endif()

if(GYRE_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(gyre PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(gyre PUBLIC GYRE_OPENMP=1)
    message(STATUS "Gyre: OpenMP enabled")
  else()
    message(WARNING "Gyre: OpenMP requested but not found; kernels stay single-threaded")
  endif()
endif()

add_library(gyre-app-jobs STATIC apps/common/jobs.cpp)
target_include_directories(gyre-app-jobs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/apps/common
                                        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/gyre)
if(TARGET nlohmann_json::nlohmann_json)
  get_target_property(_nlohmann_inc nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories(gyre-app-jobs PRIVATE ${_nlohmann_inc})
elseif(DEFINED nlohmann_json_SOURCE_DIR)
  target_include_directories(gyre-app-jobs PRIVATE ${nlohmann_json_SOURCE_DIR}/include)
endif()
target_link_libraries(gyre-app-jobs PUBLIC gyre)

add_executable(gyre-cli apps/cli/main.cpp)
target_link_libraries(gyre-cli PRIVATE gyre-app-jobs)

if(GYRE_BUILD_EXAMPLE_BINS)
  add_executable(gyre-ga-onemax examples/ga_onemax.cpp)
  target_link_libraries(gyre-ga-onemax PRIVATE gyre)
  add_executable(gyre-charlm examples/charlm.cpp)
  target_link_libraries(gyre-charlm PRIVATE gyre)
endif()

if(GYRE_ENABLE_TUI)
  include(FetchContent)
  FetchContent_Declare(
    ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
    GIT_TAG v5.0.0
  )
  FetchContent_MakeAvailable(ftxui)
  add_executable(gyre-tui apps/tui/main.cpp)
  target_link_libraries(gyre-tui PRIVATE gyre-app-jobs ftxui::screen ftxui::dom ftxui::component)
endif()

include(CMakePackageConfigHelpers)
install(TARGETS gyre EXPORT GyreTargets ARCHIVE DESTINATION lib)
install(DIRECTORY include/gyre DESTINATION include)
install(EXPORT GyreTargets FILE GyreTargets.cmake NAMESPACE gyre:: DESTINATION lib/cmake/Gyre)
configure_package_config_file(
  cmake/GyreConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/GyreConfig.cmake
  INSTALL_DESTINATION lib/cmake/Gyre
)

if(GYRE_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_executable(gyre-tests
    tests/tensor_test.cpp
    tests/ops_test.cpp
    tests/grok_ops_test.cpp
    tests/grok_attn_test.cpp
    tests/grok_moe_test.cpp
    tests/grok_lm_test.cpp
    tests/mmap_test.cpp
    tests/safetensors_test.cpp
    tests/compress_probe_test.cpp
    tests/wpack_test.cpp
    tests/grok_io_test.cpp
    tests/ga_test.cpp
    tests/nn_test.cpp
    tests/tiktoken_test.cpp
    tests/ckpt_test.cpp
    tests/adapt_grid_test.cpp
  )
  target_link_libraries(gyre-tests PRIVATE gyre GTest::gtest_main)
  target_compile_definitions(gyre-tests PRIVATE GYRE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  if(GYRE_ENABLE_ASAN AND NOT MSVC)
    target_compile_options(gyre-tests PRIVATE -fsanitize=address)
    target_link_options(gyre-tests PRIVATE -fsanitize=address)
  endif()
  include(GoogleTest)
  gtest_discover_tests(gyre-tests)
endif()
