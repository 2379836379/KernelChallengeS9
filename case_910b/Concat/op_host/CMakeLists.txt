add_library(concat_host OBJECT concat_op.cpp)
target_compile_features(concat_host PRIVATE cxx_std_17)
