cmake_minimum_required(VERSION 3.10)
project(mtspy_cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
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
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# find packages and dependencies
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/mtspy/cpp/cmake/modules")

find_package(pybind11 CONFIG HINTS $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT} QUIET)
if (NOT pybind11_FOUND)
    add_subdirectory("${PROJECT_SOURCE_DIR}/third-party/pybind11")
    set(pybind11_FOUND)
endif()

# Check for required packages
find_package(OpenMP REQUIRED)
pybind11_add_module(mtspy_cpp mtspy/cpp/main.cpp)

target_link_libraries(mtspy_cpp PRIVATE OpenMP::OpenMP_CXX)

# Set compiler specific flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    message(STATUS "Setting Intel compiler specific flags")
    set(FLAGS_INTEL_DEFAULT "-xAVX -m64 -parallel -qopenmp -qopt-report=2 -qopt-report-phase=vec -qopt-report=2 -qopt-report-phase=par")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS_INTEL_DEFAULT}")
endif()


# Find optional packages
find_package(Eigen3 3.3.7 NO_MODULE)
if (TARGET Eigen3::Eigen)
    add_definitions(-DUSE_EIGEN_BACKEND)
    message(STATUS "Eigen backend is available.")
    target_link_libraries(mtspy_cpp PUBLIC Eigen3::Eigen)
endif()