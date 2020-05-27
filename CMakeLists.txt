cmake_minimum_required(VERSION 3.10)
project(cmake_example)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Check for required packages
find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
add_subdirectory(pybind11)
pybind11_add_module(cmake_example src/main.cpp)

target_include_directories(cmake_example SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(cmake_example PUBLIC OpenMP::OpenMP_CXX)