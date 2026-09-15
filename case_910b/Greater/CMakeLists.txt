cmake_minimum_required(VERSION 3.16)
project(greater_ascendc LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(op_host)
add_subdirectory(op_kernel)
add_custom_target(greater_ascendc ALL
  DEPENDS greater_host greater_kernel
  COMMENT "Building Greater Ascend C sources")
