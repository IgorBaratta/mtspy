cmake_minimum_required(VERSION 3.12)
project(mtspy_cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set c++ flags
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# find packages and dependencies
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/mtspy/cpp/cmake/modules")

find_package(Python3 REQUIRED Interpreter Development)
find_package(pybind11 CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})
if (NOT pybind11_FOUND)
    add_subdirectory("${PROJECT_SOURCE_DIR}/third-party/pybind11")
endif()

if(NOT ENV{EIGEN3_INCLUDE_DIR})
    set(EIGEN3_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/third-party/eigen")
endif()

# Check for required packages
find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3.9 REQUIRED)
pybind11_add_module(mtspy_cpp mtspy/cpp/main.cpp)

target_include_directories(mtspy_cpp SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(mtspy_cpp PUBLIC OpenMP::OpenMP_CXX)