# looking for cuda package
find_package(CUDA REQUIRED 9.0)

# using separable compilation for dynamic parallelism
SET(CUDA_SEPARABLE_COMPILATION ON)

set(ROOT_DIR "${CMAKE_SOURCE_DIR}/../../../")
include_directories("${ROOT_DIR}/data/")
include_directories("/usr/local/cuda-${CUDA_VERSION}/include/")

# setting release mode flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# nvcc flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")

# generating libararies
#CUDA_ADD_LIBRARY(gpu_kernels kernels.cu kernels_optimized.cu kernels_tile.cu kernels_compressed.cu kernels.cuh OPTIONS -G -g -Xcompiler -fPIC -arch=sm_60 -rdc=true)
#CUDA_ADD_LIBRARY(gpu_small small_data.cu small_data_optimized.cu small_data.cuh OPTIONS -G -g -Xcompiler -fPIC -arch=sm_60 -rdc=true)
CUDA_ADD_LIBRARY(gpu_operators ../../empty.cpp OPTIONS -G -g -Xcompiler -fPIC -arch=sm_60 -rdc=true)
CUDA_ADD_LIBRARY(gpu_eth eth.cu eth.cuh OPTIONS -G -g -Xcompiler -fPIC -arch=sm_60 -rdc=true)
#set_target_properties(gpu_kernels
#                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_eth
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_operators
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(gpu_small
#                      gpu_common
#                      gpu_kernels)
target_link_libraries(gpu_eth
                      gpu_common)
target_link_libraries(gpu_operators
                      gpu_eth
#                      gpu_kernels
#                      gpu_small
                      gpu_common
                      gpu_debug
                      ${CUDA_LIBRARIES})
