CUDA_ADD_LIBRARY(gpu_wrapper GPUWrapper.cu GPUWrapper.hpp)

target_link_libraries(gpu_wrapper
                      gpu_operators
                      gpu_small)