cmake_minimum_required(VERSION 3.18)
project(StakMesh VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -fsanitize=address,undefined")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
endif()

find_package(Threads REQUIRED)

# ── stakmesh_comm ─────────────────────────────────────────────────────────────
# Pure networking + collective algorithms. Zero dependency on StakML -
# operates on raw float*. This is what Phase 1 is about, and it's fully
# testable on its own (see tests/test_ring_allreduce.cpp).
add_library(stakmesh_comm INTERFACE)
target_include_directories(stakmesh_comm INTERFACE include)
target_link_libraries(stakmesh_comm INTERFACE Threads::Threads)

if(WIN32)
    target_link_libraries(stakmesh_comm INTERFACE ws2_32)
endif()

# ── StakML dependency (needed for stakmesh_dist and the integration test) ────
# Fetched at configure time so you don't have to manage a submodule by hand.
# Set STAKMESH_STAKML_SOURCE_DIR to point at a local checkout instead (handy
# while you're actively developing both repos side by side).
option(STAKMESH_BUILD_DIST "Build the StakML integration layer (needs StakML)" ON)

if(STAKMESH_BUILD_DIST)
    include(FetchContent)
    # EXCLUDE_FROM_ALL: we only want StakML's header-only `stakml` interface
    # target, not its own tests/examples/benchmarks as part of OUR default
    # build. (Needs CMake >= 3.28, which is what this file requires.)
    if(DEFINED STAKMESH_STAKML_SOURCE_DIR)
        FetchContent_Declare(stakml SOURCE_DIR ${STAKMESH_STAKML_SOURCE_DIR} EXCLUDE_FROM_ALL)
    else()
        FetchContent_Declare(
            stakml
            GIT_REPOSITORY https://github.com/STAKXX002/StakML.git
            GIT_TAG main
            EXCLUDE_FROM_ALL
        )
    endif()
    FetchContent_MakeAvailable(stakml)

    # ── stakmesh_dist ─────────────────────────────────────────────────────────
    # The StakML-aware glue (DistributedContext::sync_gradients). Depends on
    # both stakmesh_comm and stakml.
    add_library(stakmesh_dist INTERFACE)
    target_include_directories(stakmesh_dist INTERFACE include)
    target_link_libraries(stakmesh_dist INTERFACE stakmesh_comm stakml)
endif()

# ── Tests ────────────────────────────────────────────────────────────────────
enable_testing()

add_executable(test_ring_allreduce tests/test_ring_allreduce.cpp)
target_link_libraries(test_ring_allreduce PRIVATE stakmesh_comm)
add_test(NAME ring_allreduce COMMAND test_ring_allreduce)

add_executable(test_broadcast tests/test_broadcast.cpp)
target_link_libraries(test_broadcast PRIVATE stakmesh_comm)
add_test(NAME broadcast COMMAND test_broadcast)

if(STAKMESH_BUILD_DIST)
    add_executable(test_distributed_integration tests/test_distributed_integration.cpp)
    target_link_libraries(test_distributed_integration PRIVATE stakmesh_dist)
    add_test(NAME distributed_integration COMMAND test_distributed_integration)

    add_executable(test_distributed_training_loop tests/test_distributed_training_loop.cpp)
    target_link_libraries(test_distributed_training_loop PRIVATE stakmesh_dist)
    add_test(NAME distributed_training_loop COMMAND test_distributed_training_loop)

    add_executable(test_distributed_sequential tests/test_distributed_sequential.cpp)
    target_link_libraries(test_distributed_sequential PRIVATE stakmesh_dist)
    add_test(NAME distributed_sequential COMMAND test_distributed_sequential)

    # Real Phase 2 example. Not run by ctest (needs MNIST data files + a
    # second machine) -- built so it's caught by CI/compile checks, and
    # ready to run directly: see examples/mnist_distributed.cpp for usage.
    add_executable(mnist_distributed examples/mnist_distributed.cpp)
    target_link_libraries(mnist_distributed PRIVATE stakmesh_dist)
endif()
