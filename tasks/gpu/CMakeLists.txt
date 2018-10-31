find_library(gpu_gpuprobe
             ${CMAKE_SOURCE_DIR}/task/gpu/kernels/)
add_library(gpu_gpuprobe STATIC IMPORTED)
set_target_properties(gpu_gpuprobe PROPERTIES
                      IMPORTED_LOCATION "./task/gpu/kernels/libgpu_gpuprobe.a")


add_library(gpu_wrapper GPUWrapper.cpp GPUWrapper.hpp)

set_target_properties(gpu_wrapper
                      PROPERTIES IMPORTED_LINK_INTERFACE_LIBRARIES
                      gpu_gpuprobe)