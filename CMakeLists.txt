cmake_minimum_required(VERSION 3.10)
project(mtspy_cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
add_subdirectory("${PROJECT_SOURCE_DIR}/third-party/pybind11")

# Check for required packages
find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
pybind11_add_module(mtspy_cpp cpp/main.cpp)

target_include_directories(mtspy_cpp SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(mtspy_cpp PUBLIC OpenMP::OpenMP_CXX)