#include "kernels.cuh"
#include "../../utils/gpu/debug.cuh"
#include "../../utils/gpu/common.cuh"
#include "../../utils/gpu/cuda_parameters.hpp"
#include "thrust/device_ptr.h"
#include "thrust/execution_policy.h"
#include "thrust/scan.h"

namespace SD {

/**shared_memory(relation_t * , relation_t * , args_t *, cudaParameters_t )
* Function to join small relations i.e both R and S are smaller than GPU
* memory. The function uses histogram stored in the GPU shared memory.
* hRelR     : host side array for relation R.
* hRelS     : host side array for relation S.
* agrs      : arguments data structure needed for hash join.
* cudaParam : data structure storing the cuda parameters
*/
int shared_memory(relation_t *hRelR, relation_t *hRelS, args_t *args, cudaParameters_t *cudaParam) {

  //setting up the logger variable

  relation_t *relR = (relation_t *) malloc(sizeof(relation_t)); //Device side array for relation R
  relation_t *relS = (relation_t *) malloc(sizeof(relation_t));; //Device side array for relation S

  relation_t *relRn = (relation_t *) malloc(sizeof(relation_t)); //Device side array for partitioned relation R
  relation_t *relSn = (relation_t *) malloc(sizeof(relation_t));; //Device side array for partitioned relation S

  relR->numTuples = hRelR->numTuples;
  relS->numTuples = hRelS->numTuples;

  data *out; //GPU side output buffer
  int *globalPtr; //The global pointer that is used to get the index of the output tuples.

  //allocating memory for output buffer
  cudaMalloc((void **) &out, 2 * relS->numTuples * sizeof(data));

  //allocating memory for the global pointer
  cudaMalloc((void **) &globalPtr, sizeof(data));

  //allocating device memory for storing input data
  cudaMalloc((void **) &relR->id, relR->numTuples * sizeof(data));
  cudaMalloc((void **) &relR->key, relR->numTuples * sizeof(data));
  cudaMalloc((void **) &relS->id, relS->numTuples * sizeof(data));
  cudaMalloc((void **) &relS->key, relS->numTuples * sizeof(data));

  //allocating device memory for storing partitioned data
  relRn->numTuples = relR->numTuples;
  relSn->numTuples = relS->numTuples;

  cudaMalloc((void **) &relRn->id, relRn->numTuples * sizeof(data));
  cudaMalloc((void **) &relRn->key, relRn->numTuples * sizeof(data));
  cudaMalloc((void **) &relSn->id, relSn->numTuples * sizeof(data));
  cudaMalloc((void **) &relSn->key, relSn->numTuples * sizeof(data));

  //declaring device side histogram data
  int *rHisto, *sHisto;
  int *rnHisto, *snHisto; //To allow for histogram update during re-order.

  //allocating device side memory for histogram. An additional entry is required for the last partition.
  cudaMalloc((void **) &rHisto, (args->pCount + 1) * sizeof(int));
  cudaMalloc((void **) &sHisto, (args->pCount + 1) * sizeof(int));

  cudaMalloc((void **) &rnHisto, (args->pCount + 1) * sizeof(int));
  cudaMalloc((void **) &snHisto, (args->pCount + 1) * sizeof(int));

  //setting the global pointer to 0
  cudaMemset(globalPtr, 0, sizeof(int));

  //initializing all histogram entries to 0
  cudaMemset(rHisto, 0, (args->pCount + 1) * sizeof(int));
  cudaMemset(sHisto, 0, (args->pCount + 1) * sizeof(int));
  cudaMemset(rnHisto, 0, (args->pCount + 1) * sizeof(int));
  cudaMemset(snHisto, 0, (args->pCount + 1) * sizeof(int));

  //setting kernel thread dimensions
  cudaParam->gridSize =
      args->pCountL2; //to avoid the histogram buffer overflow. the size of Histo is pCountL1 * pCountL2
  cudaParam->blockSize = MAX_BLOCK_SIZE;

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //copying Key of relation R to GPU for building the histogram
  cudaMemcpyAsync(relR->key, hRelR->key, relR->numTuples * sizeof(int), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  //building histogram for relation R
  histogram_build_L1 << < args->pCountL2, cudaParam->blockSize, args->pCountL1 * sizeof(int), cudaParam->streams[0] >>
      > (relR->key, relR->numTuples, args->pCountL1, rHisto);

  //getting the prefix sum of the level 1 histogram for relation R
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(rHisto),
                         thrust::device_pointer_cast(rHisto) + (args->pCountL1 * cudaParam->gridSize + 1),
                         thrust::device_pointer_cast(rHisto));

  //copying id of relation R to GPU
  cudaMemcpyAsync(relR->id, hRelR->id, relR->numTuples * sizeof(int), cudaMemcpyHostToDevice, cudaParam->streams[1]);

  //making sure that all histogram build and data copy are complete.
  cudaDeviceSynchronize();

  //re-ordering relation R. This is the first level of partitioning
  reorder_L1 << < args->pCountL2, cudaParam->blockSize, args->pCountL1 * sizeof(int), cudaParam->streams[0] >>
      > (relR->key, relR->id, relR->numTuples, args->pCountL1, rHisto, relRn->key, relRn->id);

  //copying Key of relation S to GPU for building the histogram
  cudaMemcpyAsync(relS->key, hRelS->key, relS->numTuples * sizeof(int), cudaMemcpyHostToDevice, cudaParam->streams[2]);

  //building histogram for second level of relation R partitioning
  histogram_build_L2 << < args->pCountL1, cudaParam->blockSize, args->pCountL2 * sizeof(int), cudaParam->streams[0] >>
      > (relRn->key, args->pCountL2, args->pCountL1, rnHisto, rHisto);

  //building histogram for relation S
  histogram_build_L1 << < args->pCountL2, cudaParam->blockSize, args->pCountL1 * sizeof(int), cudaParam->streams[2] >>
      > (relS->key, relS->numTuples, args->pCountL1, sHisto);

  //getting the prefix sum of the level 2 histogram for relation R
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(rnHisto),
                         thrust::device_pointer_cast(rnHisto) + (args->pCountL1 * cudaParam->gridSize + 1),
                         thrust::device_pointer_cast(rnHisto));

  //getting the prefix sum of the level 1 histogram for relation S
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[2]),
                         thrust::device_pointer_cast(sHisto),
                         thrust::device_pointer_cast(sHisto) + (args->pCountL1 * cudaParam->gridSize + 1),
                         thrust::device_pointer_cast(sHisto));

  //copying id of relation S to GPU
  cudaMemcpyAsync(relS->id, hRelS->id, relS->numTuples * sizeof(int), cudaMemcpyHostToDevice, cudaParam->streams[3]);

  //re-ordering relation R. This is the second level of partitioning
  reorder_L2 << < args->pCountL1, cudaParam->blockSize, args->pCountL2 * sizeof(int), cudaParam->streams[0] >>
      > (relRn->key, relRn->id, args->pCountL2, args->pCountL1, rnHisto, rHisto, relR->key, relR->id);

  //making sure the data transfer of id values of relation S is complete before re-ordering the realtion.
  cudaStreamSynchronize(cudaParam->streams[3]);

  //re-ordering relation S. This is the first level of partitioning
  reorder_L1 << < args->pCountL2, cudaParam->blockSize, args->pCountL1 * sizeof(int), cudaParam->streams[2] >>
      > (relS->key, relS->id, relS->numTuples, args->pCountL1, sHisto, relSn->key, relSn->id);

  //building histogram for second level of relation S partitioning
  histogram_build_L2 << < args->pCountL1, cudaParam->blockSize, args->pCountL2 * sizeof(int), cudaParam->streams[2] >>
      > (relSn->key, args->pCountL2, args->pCountL1, snHisto, sHisto);

  //getting the prefix sum of the level 2 histogram for relation S
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[2]),
                         thrust::device_pointer_cast(snHisto),
                         thrust::device_pointer_cast(snHisto) + (args->pCountL1 * cudaParam->gridSize + 1),
                         thrust::device_pointer_cast(snHisto));

  //re-ordering relation S. This is the second level of partitioning
  reorder_L2 << < args->pCountL1, cudaParam->blockSize, args->pCountL2 * sizeof(int), cudaParam->streams[2] >>
      > (relSn->key, relSn->id, args->pCountL2, args->pCountL1, snHisto, sHisto, relS->key, relS->id);

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[2]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  //displaying execution time
  std::cout << "Partition Stage Execution Time for Shared Memory: " << cudaParam->time << " ms" << std::endl;

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //probe kernel invocation. We assume that the data distribution is uniform.
  probe << < std::min(args->pCount, MAX_GRID_SIZE), ceil((float) relS->numTuples / args->pCount), 2
      * ceil((float) relS->numTuples / args->pCount) * sizeof(int), cudaParam->streams[0] >>
      > (relR->key, relR->id, relS->key, relS->id, rnHisto, snHisto, args->pCount, globalPtr, out);

  //copying the results back to the CPU main memory. Assuming a 100% match rate.
  cudaMemcpyAsync(args->hOut[0], out, 2 * relS->numTuples * sizeof(int), cudaMemcpyDeviceToHost, cudaParam->streams[0]);

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[0]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  //checking for any errors during execution
  check_cuda_error((char *) __FILE__, __LINE__);

  //displaying execution time
  std::cout << "Join Stage Execution Time for Shared Memory: " << cudaParam->time << " ms" << std::endl;

  //debug code
  //displayGPUBuffer(rnHisto, args->hOut[0], args->pCount + 1);

  //cleaning up all allocated data

  cudaFree(relR->id);
  cudaFree(relR->key);
  cudaFree(relS->id);
  cudaFree(relS->key);

  cudaFree(relRn->id);
  cudaFree(relRn->key);
  cudaFree(relSn->id);
  cudaFree(relSn->key);

  cudaFree(rHisto);
  cudaFree(sHisto);
  cudaFree(rnHisto);
  cudaFree(snHisto);

  cudaFree(out);
  cudaFree(globalPtr);

  return 0;
}

/**global_memory(relation_t * , relation_t * , args_t *, cudaParameters_t )
* Function to join small relations using GPU global memory.
* hRelR     : host side array for relation R.
* hRelS     : host side array for relation S.
* agrs      : arguments data structure needed for hash join.
* cudaParam : data structure storing the cuda parameters
*/
int global_memory(relation_t *hRelR, relation_t *hRelS, args_t *args, cudaParameters_t *cudaParam) {

  //setting up the logger variable

  relation_t *relR = (relation_t *) malloc(sizeof(relation_t)); //Device side array for relation R
  relation_t *relS = (relation_t *) malloc(sizeof(relation_t));; //Device side array for relation S

  relation_t *relRn = (relation_t *) malloc(sizeof(relation_t)); //Device side array for partitioned relation R
  relation_t *relSn = (relation_t *) malloc(sizeof(relation_t));; //Device side array for partitioned relation S

  relR->numTuples = hRelR->numTuples;
  relS->numTuples = hRelS->numTuples;

  data *out; //GPU side output buffer
  int *globalPtr; //The global pointer that is used to get the index of the output tuples.

  //allocating memory for output buffer
  cudaMalloc((void **) &out, 2 * relS->numTuples * sizeof(data));

  //allocating memory for the global pointer
  cudaMalloc((void **) &globalPtr, sizeof(data));

  //allocating device memory for storing input data
  cudaMalloc((void **) &relR->id, relR->numTuples * sizeof(data));
  cudaMalloc((void **) &relR->key, relR->numTuples * sizeof(data));
  cudaMalloc((void **) &relS->id, relS->numTuples * sizeof(data));
  cudaMalloc((void **) &relS->key, relS->numTuples * sizeof(data));

  //allocating device memory for storing partitioned data
  relRn->numTuples = relR->numTuples;
  relSn->numTuples = relS->numTuples;

  cudaMalloc((void **) &relRn->id, relRn->numTuples * sizeof(data));
  cudaMalloc((void **) &relRn->key, relRn->numTuples * sizeof(data));
  cudaMalloc((void **) &relSn->id, relSn->numTuples * sizeof(data));
  cudaMalloc((void **) &relSn->key, relSn->numTuples * sizeof(data));

  //declaring device side histogram data
  int *rHisto, *sHisto;
  int *rnHisto, *snHisto; //To allow for histogram update during re-order.

  //allocating device side memory for histogram. An additional entry is required for the last partition.
  cudaMalloc((void **) &rHisto, (args->pCount + 1) * sizeof(int));
  cudaMalloc((void **) &sHisto, (args->pCount + 1) * sizeof(int));

  cudaMalloc((void **) &rnHisto, (args->pCount + 1) * sizeof(int));
  cudaMalloc((void **) &snHisto, (args->pCount + 1) * sizeof(int));

  //setting the global pointer to 0
  cudaMemset(globalPtr, 0, sizeof(int));

  //initializing all histogram entries to 0
  cudaMemset(rHisto, 0, (args->pCount + 1) * sizeof(int));
  cudaMemset(sHisto, 0, (args->pCount + 1) * sizeof(int));
  cudaMemset(rnHisto, 0, (args->pCount + 1) * sizeof(int));
  cudaMemset(snHisto, 0, (args->pCount + 1) * sizeof(int));

  //setting kernel thread dimensions
  cudaParam->gridSize =
      args->pCountL2; //to avoid the histogram buffer overflow. the size of Histo is pCountL1 * pCountL2
  cudaParam->blockSize = MAX_BLOCK_SIZE;

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //copying Key of relation R to GPU for building the histogram
  cudaMemcpyAsync(relR->key, hRelR->key, relR->numTuples * sizeof(int), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  //building histogram for relation R
  histogram_build_global << < std::min((int) (relR->numTuples / cudaParam->blockSize), MAX_GRID_SIZE),
      cudaParam->blockSize, 0, cudaParam->streams[0] >> > (relR->key, relR->numTuples, args->pCount, rHisto);

  //getting the prefix sum of the level 1 histogram for relation R
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(rHisto),
                         thrust::device_pointer_cast(rHisto) + (args->pCount + 1),
                         thrust::device_pointer_cast(rHisto));

  //copying id of relation R to GPU
  cudaMemcpyAsync(relR->id, hRelR->id, relR->numTuples * sizeof(int), cudaMemcpyHostToDevice, cudaParam->streams[1]);

  //creating a device side copy of relation R histogram for re-order kernel. Otehrwise re-order kernel will update the histogram making it unusable for the probe kernel.
  cudaMemcpyAsync(rnHisto, rHisto, (args->pCount + 1) * sizeof(int), cudaMemcpyDeviceToDevice, cudaParam->streams[0]);

  //making sure that all histogram build and data copy are complete.
  cudaDeviceSynchronize();

  //re-ordering relation R. This is the first level of partitioning
  reorder_global << < std::min((int) (relR->numTuples / cudaParam->blockSize), MAX_GRID_SIZE), cudaParam->blockSize, 0,
      cudaParam->streams[0] >> > (relR->key, relR->id, relR->numTuples, args->pCount, rHisto, relRn->key, relRn->id);

  //copying Key of relation S to GPU for building the histogram
  cudaMemcpyAsync(relS->key, hRelS->key, relS->numTuples * sizeof(int), cudaMemcpyHostToDevice, cudaParam->streams[2]);

  //building histogram for relation S
  histogram_build_global << < std::min((int) (relS->numTuples / cudaParam->blockSize), MAX_GRID_SIZE),
      cudaParam->blockSize, 0, cudaParam->streams[2] >> > (relS->key, relS->numTuples, args->pCount, sHisto);

  //getting the prefix sum of the level 1 histogram for relation S
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[2]),
                         thrust::device_pointer_cast(sHisto),
                         thrust::device_pointer_cast(sHisto) + (args->pCount + 1),
                         thrust::device_pointer_cast(sHisto));

  //copying id of relation S to GPU
  cudaMemcpyAsync(relS->id, hRelS->id, relS->numTuples * sizeof(int), cudaMemcpyHostToDevice, cudaParam->streams[3]);

  //creating a device side copy of relation S histogram for re-order kernel. Otehrwise re-order kernel will update the histogram making it unusable for the probe kernel.
  cudaMemcpyAsync(snHisto, sHisto, (args->pCount + 1) * sizeof(int), cudaMemcpyDeviceToDevice, cudaParam->streams[2]);

  //making sure the data transfer of id values of relation S is complete before re-ordering the realtion.
  cudaStreamSynchronize(cudaParam->streams[3]);

  //re-ordering relation S. This is the first level of partitioning
  reorder_global << < std::min((int) (relS->numTuples / cudaParam->blockSize), MAX_GRID_SIZE), cudaParam->blockSize, 0,
      cudaParam->streams[2] >> > (relS->key, relS->id, relS->numTuples, args->pCount, sHisto, relSn->key, relSn->id);

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[2]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  //displaying execution time
  std::cout << "Partition Stage Execution Time for Global Memory: " << cudaParam->time << " ms" << std::endl;

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //probe kernel invocation. We assume that the data distribution is uniform.
  probe << < std::min(args->pCount, MAX_GRID_SIZE), ceil((float) relS->numTuples / args->pCount), 2
      * ceil((float) relS->numTuples / args->pCount) * sizeof(int), cudaParam->streams[0] >>
      > (relRn->key, relRn->id, relSn->key, relSn->id, rnHisto, snHisto, args->pCount, globalPtr, out);

  //copying the results back to the CPU main memory. Assuming a 100% match rate.
  cudaMemcpyAsync(args->hOut[0], out, 2 * relS->numTuples * sizeof(int), cudaMemcpyDeviceToHost, cudaParam->streams[0]);

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[0]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  //checking for any errors during execution
  check_cuda_error((char *) __FILE__, __LINE__);

  //displaying execution time
  std::cout << "Join Stage Execution Time for Global Memory: " << cudaParam->time << " ms" << std::endl;

  //debug code
  //displayGPUBuffer(out, args->hOut[0], 2 * relRn->numTuples);

  //cleaning up all allocated data

  cudaFree(relR->id);
  cudaFree(relR->key);
  cudaFree(relS->id);
  cudaFree(relS->key);

  cudaFree(relRn->id);
  cudaFree(relRn->key);
  cudaFree(relSn->id);
  cudaFree(relSn->key);

  cudaFree(rHisto);
  cudaFree(sHisto);
  cudaFree(rnHisto);
  cudaFree(snHisto);

  cudaFree(out);
  cudaFree(globalPtr);

  return 0;
}

/**small_data_sm(relation_t * , relation_t * , args_t * )
* Function to join small relations with high match rate. The histogram is
* stored in the shared memory in this implementation.
* hRelR     : host side array for relation R.
* hRelS     : host side array for relation S.
* agrs      : arguments data structure needed for hash join.
* cudaParam : data structure storing the cuda parameters
*/

}