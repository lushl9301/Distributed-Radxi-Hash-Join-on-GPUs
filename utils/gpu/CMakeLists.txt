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
CUDA_ADD_LIBRARY(gpu_common common.cu common.cuh OPTIONS -Xcompiler -fPIC -arch=sm_60 -rdc=true)
CUDA_ADD_LIBRARY(gpu_debug debug.cuh OPTIONS -Xcompiler -fPIC -arch=sm_60 -rdc=true)
