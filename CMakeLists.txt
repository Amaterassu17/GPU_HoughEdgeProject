# FILEPATH: /user/2/gennusoa/CGPU-Project/GPU_HoughEdgeProject/CMakeLists.txt

cmake_minimum_required(VERSION 3.12)
project(GPU_HoughEdgeProject)

# Set the C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

# Add the include directories for the headers in the libs folder
include_directories(libs)

# Add the source files to be compiled
add_executable(GPU_HoughEdgeProject main.cpp)

# Link the necessary libraries
target_link_libraries(GPU_HoughEdgeProject OpenCL)
