add_library(greater_host OBJECT greater_op.cpp)
target_compile_features(greater_host PRIVATE cxx_std_17)
