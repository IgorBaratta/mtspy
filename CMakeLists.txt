cmake_minimum_required(VERSION 3.10)
project(mtspy_cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/mtspy/cpp/cmake/modules")
add_subdirectory("${PROJECT_SOURCE_DIR}/third-party/pybind11")
set(EIGEN3_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/third-party/eigen")

# Check for required packages
find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
pybind11_add_module(mtspy_cpp mtspy/cpp/main.cpp)

target_include_directories(mtspy_cpp SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(mtspy_cpp PUBLIC OpenMP::OpenMP_CXX)