CUDA_ADD_LIBRARY(gpu_wrapper GPUWrapper.cu GPUWrapper.hpp OPTIONS -G -g)

target_link_libraries(gpu_wrapper
                      gpu_operators
                      gpu_eth)