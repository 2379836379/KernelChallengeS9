if(NOT DEFINED ASCENDC_CANN_PACKAGE_PATH)
  set(ASCENDC_CANN_PACKAGE_PATH "$ENV{ASCENDC_CANN_PACKAGE_PATH}")
endif()
if(NOT ASCENDC_CANN_PACKAGE_PATH)
  message(FATAL_ERROR "Set ASCENDC_CANN_PACKAGE_PATH to the CANN toolkit root")
endif()
find_path(ASCENDC_INCLUDE_DIR kernel_operator.h
  PATHS
    "${ASCENDC_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfw"
    "${ASCENDC_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc/include"
    "${ASCENDC_CANN_PACKAGE_PATH}/include"
  NO_DEFAULT_PATH)
if(NOT ASCENDC_INCLUDE_DIR)
  message(FATAL_ERROR "kernel_operator.h was not found in ${ASCENDC_CANN_PACKAGE_PATH}")
endif()
add_library(concat_kernel OBJECT concat_kernel.cpp)
target_include_directories(concat_kernel PRIVATE "${ASCENDC_INCLUDE_DIR}")
target_compile_features(concat_kernel PRIVATE cxx_std_17)
