add_library(gpu_wrapper GPUWrapper.cpp GPUWrapper.hpp)

target_link_libraries(gpu_wrapper
                      gpu_operators)