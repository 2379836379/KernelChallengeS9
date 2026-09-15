cmake_minimum_required(VERSION 3.16)
project(concat_ascendc LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(op_host)
add_subdirectory(op_kernel)
add_custom_target(concat_ascendc ALL DEPENDS concat_host concat_kernel
  COMMENT "Building Concat Ascend C sources")
