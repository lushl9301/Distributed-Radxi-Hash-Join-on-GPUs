#include <cstdio>

/**check_cuda_error(char *, int )
* Function to identify the CUDA error using the error code.
* file  : the file where the error occured.
* line  : the line number where the error occured.
*/
void check_cuda_error(char *file, int line) {

  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {
    printf("%s in %s at line %d", cudaGetErrorString(error), file, line);
    exit(-1);
  }
}