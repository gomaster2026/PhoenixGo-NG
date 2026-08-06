cmake_minimum_required(VERSION 3.5)

project(leelaz VERSION 0.17.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CPU_ONLY "Use CPU only, no OpenCL" OFF)
option(BUILD_TESTS "Build tests" ON)

find_package(Boost REQUIRED COMPONENTS program_options filesystem system)
find_package(ZLIB REQUIRED)

if(NOT USE_CPU_ONLY)
  find_package(OpenCL REQUIRED)
endif()

file(GLOB SOURCES src/*.cpp)

add_executable(leelaz ${SOURCES})
target_link_libraries(leelaz Boost::program_options Boost::filesystem Boost::system ZLIB::ZLIB)

if(NOT USE_CPU_ONLY)
  target_link_libraries(leelaz OpenCL::OpenCL)
  target_compile_definitions(leelaz PRIVATE USE_OPENCL)
endif()

if(BUILD_TESTS)
  add_executable(tests src/tests/tests.cpp)
  target_link_libraries(tests Boost::program_options Boost::filesystem Boost::system ZLIB::ZLIB)
endif()