#include "kernels.cuh"
#include "../../utils/gpu/debug.cuh"
#include "../../utils/gpu/common.cuh"
#include "../../utils/gpu/cuda_parameters.hpp"
#include "thrust/device_ptr.h"
#include "thrust/execution_policy.h"
#include "thrust/scan.h"

namespace SD {

//namepsace containing the optimized functions
namespace OPT {

/**shared_memory(relation_t * , relation_t * , args_t *, cudaParameters_t )
 * Function to join small relations using shared memory. The function has additional
 * optimizations and tuning.
 * hRelR     : host side array for relation R.
 * hRelS     : host side array for relation S.
 * agrs      : arguments data structure needed for hash join.
 * cudaParam : data structure storing the cuda parameters
 */
int shared_memory(relation_t *hRelR, relation_t *hRelS, args_t *args, cudaParameters_t *cudaParam) {

  relation_t *relR = (relation_t *) malloc(sizeof(relation_t)); //Device side array for relation R
  relation_t *relS = (relation_t *) malloc(sizeof(relation_t));; //Device side array for relation S

  relation_t *relRn = (relation_t *) malloc(sizeof(relation_t)); //Device side array for partitioned relation R
  relation_t *relSn = (relation_t *) malloc(sizeof(relation_t));; //Device side array for partitioned relation S

  relR->numTuples = hRelR->numTuples;
  relS->numTuples = hRelS->numTuples;

  data *out[2]; //GPU side output buffer
  int *globalPtr[2]; //The global pointer that is used to get the index of the output tuples.

  //allocating memory for output buffer
  cudaMalloc((void **) &out[0], relS->numTuples * sizeof(data));
  cudaMalloc((void **) &out[1], relS->numTuples * sizeof(data));

  //allocating memory for the global pointer
  cudaMalloc((void **) &globalPtr[0], sizeof(data));
  cudaMalloc((void **) &globalPtr[1], sizeof(data));

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
  cudaMalloc((void **) &rHisto, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMalloc((void **) &sHisto, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));

  cudaMalloc((void **) &rnHisto, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));
  cudaMalloc((void **) &snHisto, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));

  //setting the global pointer to 0
  cudaMemset(globalPtr[0], 0, sizeof(int));
  cudaMemset(globalPtr[1], 0, sizeof(int));

  //initializing all histogram entries to 0
  cudaMemset(rHisto, 0, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMemset(sHisto, 0, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMemset(rnHisto, 0, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));
  cudaMemset(snHisto, 0, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();
  check_cuda_error((char *) __FILE__, __LINE__);

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //copying Key of relation R to GPU for building the histogram
  cudaMemcpyAsync(relR->key, hRelR->key, relR->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  //building histogram for relation R
  histogram_build_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int),
      cudaParam->streams[0] >> > (relR->key, relR->numTuples, args->pCountL1, rHisto);

  //getting the prefix sum of the level 1 histogram for relation R
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(rHisto),
                         thrust::device_pointer_cast(rHisto) + (args->pCountL1 * cudaParam->gridSize + 1),
                         thrust::device_pointer_cast(rHisto));

  //copying id of relation R to GPU
  cudaMemcpyAsync(relR->id, hRelR->id, relR->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[1]);

  //making sure that all histogram build and data copy are complete.
  cudaDeviceSynchronize();

  //re-ordering relation R. This is the first level of partitioning
  reorder_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int), cudaParam->streams[0] >>
      > (relR->key, relR->id, relR->numTuples, args->pCountL1, rHisto, relRn->key, relRn->id);

  //copying Key of relation S to GPU for building the histogram
  cudaMemcpyAsync(relS->key, hRelS->key, relS->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[2]);

  //building histogram for second level of relation R partitioning
  histogram_build_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2
      * sizeof(int), cudaParam->streams[0] >>
      > (relRn->key, args->pCountL2, args->pCountL1, rnHisto, rHisto, cudaParam->gridSize);

  //building histogram for relation S
  histogram_build_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int),
      cudaParam->streams[2] >> > (relS->key, relS->numTuples, args->pCountL1, sHisto);

  //getting the prefix sum of the level 2 histogram for relation R
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(rnHisto),
                         thrust::device_pointer_cast(rnHisto) + (cudaParam->gridSizeMultiplier * args->pCount + 1),
                         thrust::device_pointer_cast(rnHisto));

  //getting the prefix sum of the level 1 histogram for relation S
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[2]),
                         thrust::device_pointer_cast(sHisto),
                         thrust::device_pointer_cast(sHisto) + (args->pCountL1 * cudaParam->gridSize + 1),
                         thrust::device_pointer_cast(sHisto));

  //copying id of relation S to GPU
  cudaMemcpyAsync(relS->id, hRelS->id, relS->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[3]);

  //re-ordering relation R. This is the second level of partitioning
  reorder_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2 * sizeof(int),
      cudaParam->streams[0] >>
          > (relRn->key, relRn->id, args->pCountL2, args->pCountL1, rnHisto, rHisto, relR->key, relR->id, cudaParam->gridSize);

  //making sure the data transfer of id values of relation S is complete before re-ordering the realtion.
  cudaStreamSynchronize(cudaParam->streams[3]);

  //re-ordering relation S. This is the first level of partitioning
  reorder_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int), cudaParam->streams[2] >>
      > (relS->key, relS->id, relS->numTuples, args->pCountL1, sHisto, relSn->key, relSn->id);

  //building histogram for second level of relation S partitioning
  histogram_build_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2
      * sizeof(int), cudaParam->streams[2] >>
      > (relSn->key, args->pCountL2, args->pCountL1, snHisto, sHisto, cudaParam->gridSize);

  //getting the prefix sum of the level 2 histogram for relation S
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[2]),
                         thrust::device_pointer_cast(snHisto),
                         thrust::device_pointer_cast(snHisto) + (cudaParam->gridSizeMultiplier * args->pCount + 1),
                         thrust::device_pointer_cast(snHisto));

  //re-ordering relation S. This is the second level of partitioning
  reorder_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2 * sizeof(int),
      cudaParam->streams[2] >>
          > (relSn->key, relSn->id, args->pCountL2, args->pCountL1, snHisto, sHisto, relS->key, relS->id, cudaParam->gridSize);

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[2]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  //displaying execution time
  std::cout << "Partition Stage Execution Time for Optimized Shared Memory: " << cudaParam->time << " ms" << std::endl;

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //probe kernel invocation. We assume that the data distribution is uniform.
  probe << < std::min(args->pCount / 2, MAX_GRID_SIZE), ceil((float) relS->numTuples / args->pCount), 2
      * ceil((float) relS->numTuples / args->pCount) * sizeof(data), cudaParam->streams[0] >>
      > (relR->key, relR->id, relS->key, relS->id, rnHisto, snHisto, args->pCount / 2, globalPtr[0], out[0], 0);

  //copying the results back to the CPU main memory. Assuming a 100% match rate.
  cudaMemcpyAsync(args->hOut[0], out[0], relS->numTuples * sizeof(data), cudaMemcpyDeviceToHost, cudaParam->streams[0]);

  //second probe kernel invocation
  probe << < std::min(args->pCount / 2, MAX_GRID_SIZE), ceil((float) relS->numTuples / args->pCount), 2
      * ceil((float) relS->numTuples / args->pCount) * sizeof(data), cudaParam->streams[1] >>
      > (relR->key, relR->id, relS->key, relS->id, rnHisto, snHisto, args->pCount, globalPtr[1], out[1], args->pCount
          / 2);

  //copying the second set of results back to the CPU main memory. Assuming a 100% match rate.
  cudaMemcpyAsync(args->hOut[0] + relS->numTuples,
                  out[1],
                  relS->numTuples * sizeof(data),
                  cudaMemcpyDeviceToHost,
                  cudaParam->streams[1]);

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[1]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  //checking for any errors during execution
  cudaDeviceSynchronize();
  check_cuda_error((char *) __FILE__, __LINE__);

  //displaying execution time
  std::cout << "Join Stage Execution Time for Optimized Shared Memory: " << cudaParam->time << " ms" << std::endl;

  //debug code
  //displayGPUBuffer(out[0], args->hOut[0], 10);
  //displayGPUBuffer(out[1], args->hOut[0], 10);

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

  cudaFree(out[0]);
  cudaFree(out[1]);
  cudaFree(globalPtr[0]);
  cudaFree(globalPtr[1]);

  return 0;
}

/**shared_memory(relation_t * , relation_t * , args_t *, cudaParameters_t )
 * Function to join small relations using shared memory. The function has additional
 * optimizations and tuning.
 * hRelR     : host side array for relation R.
 * hRelS     : host side array for relation S.
 * agrs      : arguments data structure needed for hash join.
 * cudaParam : data structure storing the cuda parameters
 */
int shared_memory_PT(relation_t *hRelR, relation_t *hRelS, args_t *args, cudaParameters_t *cudaParam) {

  std::cout << "Executing Function : shared_memory" << std::endl;

  relation_t *relR = (relation_t *) malloc(sizeof(relation_t)); //Device side array for relation R
  relation_t *relS = (relation_t *) malloc(sizeof(relation_t));; //Device side array for relation S

  relation_t *relRn = (relation_t *) malloc(sizeof(relation_t)); //Device side array for partitioned relation R
  relation_t *relSn = (relation_t *) malloc(sizeof(relation_t));; //Device side array for partitioned relation S

  relR->numTuples = hRelR->numTuples;
  relS->numTuples = hRelS->numTuples;

  data *out[2]; //GPU side output buffer
  int *globalPtr[2]; //The global pointer that is used to get the index of the output tuples.

  //allocating memory for output buffer
  cudaMalloc((void **) &out[0], relS->numTuples * sizeof(data));
  cudaMalloc((void **) &out[1], relS->numTuples * sizeof(data));

  //allocating memory for the global pointer
  cudaMalloc((void **) &globalPtr[0], sizeof(data));
  cudaMalloc((void **) &globalPtr[1], sizeof(data));

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
  cudaMalloc((void **) &rHisto, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMalloc((void **) &sHisto, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));

  cudaMalloc((void **) &rnHisto, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));
  cudaMalloc((void **) &snHisto, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));

  //setting the global pointer to 0
  cudaMemset(globalPtr[0], 0, sizeof(int));
  cudaMemset(globalPtr[1], 0, sizeof(int));

  //initializing all histogram entries to 0
  cudaMemset(rHisto, 0, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMemset(sHisto, 0, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMemset(rnHisto, 0, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));
  cudaMemset(snHisto, 0, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();
  check_cuda_error((char *) __FILE__, __LINE__);

  //copying Key of relation R to GPU for building the histogram
  cudaMemcpyAsync(relR->key, hRelR->key, relR->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[0]);
  cudaMemcpyAsync(relR->id, hRelR->id, relR->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[1]);

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();
  check_cuda_error((char *) __FILE__, __LINE__);

  //memTest<<<cudaParam->gridSize, cudaParam->blockSize>>>(relR->key, relR->id);

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //building histogram for relation R
  histogram_build_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int),
      cudaParam->streams[0] >> > (relR->key, relR->numTuples, args->pCountL1, rHisto);

  //getting the prefix sum of the level 1 histogram for relation R
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(rHisto),
                         thrust::device_pointer_cast(rHisto) + (args->pCountL1 * cudaParam->gridSize + 1),
                         thrust::device_pointer_cast(rHisto));

  //copying id of relation R to GPU
  //cudaMemcpyAsync(relR->id, hRelR->id, relR->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[1]);

  //making sure that all histogram build and data copy are complete.
  //cudaDeviceSynchronize();

  //re-ordering relation R. This is the first level of partitioning
  reorder_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int), cudaParam->streams[0] >>
      > (relR->key, relR->id, relR->numTuples, args->pCountL1, rHisto, relRn->key, relRn->id);

  //ending time measurement
  //cudaEventRecord(cudaParam->stop, cudaParam->streams[2]);

  cudaEventRecord(cudaParam->stop, cudaParam->streams[0]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  //displaying execution time
  std::cout << "Partition Stage Execution Time for Optimized Shared Memory: "
            << args->pCount << " "
            << cudaParam->gridSize << " "
            << cudaParam->blockSize << " "
            << cudaParam->time << std::endl;

  //checking for any errors during execution
  cudaDeviceSynchronize();
  check_cuda_error((char *) __FILE__, __LINE__);

  //debug code
  //displayGPUBuffer(relR->key, args->hOut[0], 10);

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

  cudaFree(out[0]);
  cudaFree(out[1]);
  cudaFree(globalPtr[0]);
  cudaFree(globalPtr[1]);

  return 0;
}

/**shared_memory(relation_t * , relation_t * , args_t *, cudaParameters_t )
	* Function to join small relations using shared memory. The function does not take advantage
	* of CUDA streams to excute CUDA tasks concurrently
	* hRelR     : host side array for relation R.
	* hRelS     : host side array for relation S.
	* agrs      : arguments data structure needed for hash join.
	* cudaParam : data structure storing the cuda parameters
	*/
int shared_memory_streams_disabled(relation_t *hRelR, relation_t *hRelS, args_t *args, cudaParameters_t *cudaParam) {

  relation_t *relR = (relation_t *) malloc(sizeof(relation_t)); //Device side array for relation R
  relation_t *relS = (relation_t *) malloc(sizeof(relation_t));; //Device side array for relation S

  relation_t *relRn = (relation_t *) malloc(sizeof(relation_t)); //Device side array for partitioned relation R
  relation_t *relSn = (relation_t *) malloc(sizeof(relation_t));; //Device side array for partitioned relation S

  relR->numTuples = hRelR->numTuples;
  relS->numTuples = hRelS->numTuples;

  data *out[2]; //GPU side output buffer
  int *globalPtr[2]; //The global pointer that is used to get the index of the output tuples.

  //allocating memory for output buffer
  cudaMalloc((void **) &out[0], relS->numTuples * sizeof(data));
  cudaMalloc((void **) &out[1], relS->numTuples * sizeof(data));

  //allocating memory for the global pointer
  cudaMalloc((void **) &globalPtr[0], sizeof(data));
  cudaMalloc((void **) &globalPtr[1], sizeof(data));

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
  cudaMalloc((void **) &rHisto, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMalloc((void **) &sHisto, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));

  cudaMalloc((void **) &rnHisto, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));
  cudaMalloc((void **) &snHisto, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));

  //setting the global pointer to 0
  cudaMemset(globalPtr[0], 0, sizeof(int));
  cudaMemset(globalPtr[1], 0, sizeof(int));

  //initializing all histogram entries to 0
  cudaMemset(rHisto, 0, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMemset(sHisto, 0, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMemset(rnHisto, 0, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));
  cudaMemset(snHisto, 0, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //copying Key of relation R to GPU for building the histogram
  cudaMemcpyAsync(relR->key, hRelR->key, relR->numTuples * sizeof(int), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  //building histogram for relation R
  histogram_build_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int),
      cudaParam->streams[0] >> > (relR->key, relR->numTuples, args->pCountL1, rHisto);

  //getting the prefix sum of the level 1 histogram for relation R
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(rHisto),
                         thrust::device_pointer_cast(rHisto) + (args->pCountL1 * cudaParam->gridSize + 1),
                         thrust::device_pointer_cast(rHisto));

  //copying id of relation R to GPU
  cudaMemcpyAsync(relR->id, hRelR->id, relR->numTuples * sizeof(int), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  //making sure that all histogram build and data copy are complete.
  cudaDeviceSynchronize();

  //re-ordering relation R. This is the first level of partitioning
  reorder_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int), cudaParam->streams[0] >>
      > (relR->key, relR->id, relR->numTuples, args->pCountL1, rHisto, relRn->key, relRn->id);

  //copying Key of relation S to GPU for building the histogram
  cudaMemcpyAsync(relS->key, hRelS->key, relS->numTuples * sizeof(int), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  //building histogram for second level of relation R partitioning
  histogram_build_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2
      * sizeof(int), cudaParam->streams[0] >>
      > (relRn->key, args->pCountL2, args->pCountL1, rnHisto, rHisto, cudaParam->gridSize);

  //building histogram for relation S
  histogram_build_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int),
      cudaParam->streams[0] >> > (relS->key, relS->numTuples, args->pCountL1, sHisto);

  //getting the prefix sum of the level 2 histogram for relation R
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(rnHisto),
                         thrust::device_pointer_cast(rnHisto) + (cudaParam->gridSizeMultiplier * args->pCount + 1),
                         thrust::device_pointer_cast(rnHisto));

  //getting the prefix sum of the level 1 histogram for relation S
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(sHisto),
                         thrust::device_pointer_cast(sHisto) + (args->pCountL1 * cudaParam->gridSize + 1),
                         thrust::device_pointer_cast(sHisto));

  //copying id of relation S to GPU
  cudaMemcpyAsync(relS->id, hRelS->id, relS->numTuples * sizeof(int), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  //re-ordering relation R. This is the second level of partitioning
  reorder_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2 * sizeof(int),
      cudaParam->streams[0] >>
          > (relRn->key, relRn->id, args->pCountL2, args->pCountL1, rnHisto, rHisto, relR->key, relR->id, cudaParam->gridSize);

  //making sure the data transfer of id values of relation S is complete before re-ordering the realtion.
  cudaStreamSynchronize(cudaParam->streams[0]);

  //re-ordering relation S. This is the first level of partitioning
  reorder_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int), cudaParam->streams[0] >>
      > (relS->key, relS->id, relS->numTuples, args->pCountL1, sHisto, relSn->key, relSn->id);

  //building histogram for second level of relation S partitioning
  histogram_build_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2
      * sizeof(int), cudaParam->streams[0] >>
      > (relSn->key, args->pCountL2, args->pCountL1, snHisto, sHisto, cudaParam->gridSize);

  //getting the prefix sum of the level 2 histogram for relation S
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(snHisto),
                         thrust::device_pointer_cast(snHisto) + (cudaParam->gridSizeMultiplier * args->pCount + 1),
                         thrust::device_pointer_cast(snHisto));

  //re-ordering relation S. This is the second level of partitioning
  reorder_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2 * sizeof(int),
      cudaParam->streams[0] >>
          > (relSn->key, relSn->id, args->pCountL2, args->pCountL1, snHisto, sHisto, relS->key, relS->id, cudaParam->gridSize);

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[0]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  //displaying execution time
  std::cout << "Partition Stage Execution Time for Optimized Shared Memory (w/o CUDA Streams): " << cudaParam->time
            << " ms" << std::endl;

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //probe kernel invocation. We assume that the data distribution is uniform.
  probe << < std::min(args->pCount / 2, MAX_GRID_SIZE), ceil((float) relS->numTuples / args->pCount), 2
      * ceil((float) relS->numTuples / args->pCount) * sizeof(int), cudaParam->streams[0] >>
      > (relR->key, relR->id, relS->key, relS->id, rnHisto, snHisto, args->pCount / 2, globalPtr[0], out[0], 0);

  //copying the results back to the CPU main memory. Assuming a 100% match rate.
  cudaMemcpyAsync(args->hOut[0], out[0], relS->numTuples * sizeof(int), cudaMemcpyDeviceToHost, cudaParam->streams[0]);

  //second probe kernel invocation
  probe << < std::min(args->pCount / 2, MAX_GRID_SIZE), ceil((float) relS->numTuples / args->pCount), 2
      * ceil((float) relS->numTuples / args->pCount) * sizeof(int), cudaParam->streams[0] >>
      > (relR->key, relR->id, relS->key, relS->id, rnHisto, snHisto, args->pCount, globalPtr[1], out[1], args->pCount
          / 2);

  //copying the second set of results back to the CPU main memory. Assuming a 100% match rate.
  cudaMemcpyAsync(args->hOut[0] + relS->numTuples,
                  out[1],
                  relS->numTuples * sizeof(int),
                  cudaMemcpyDeviceToHost,
                  cudaParam->streams[0]);

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[0]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  //checking for any errors during execution
  check_cuda_error((char *) __FILE__, __LINE__);

  //displaying execution time
  std::cout << "Join Stage Execution Time for Optimized Shared Memory (w/o CUDA Streams): " << cudaParam->time << " ms"
            << std::endl;

  //debug code
  //displayGPUBuffer(out[1], args->hOut[0], relR->numTuples);

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

  cudaFree(out[0]);
  cudaFree(out[1]);
  cudaFree(globalPtr[0]);
  cudaFree(globalPtr[1]);

  return 0;
}

/**global_memory(relation_t * , relation_t * , args_t *, cudaParameters_t )
	* Function to join small relations using GPU global memory. The function has additional
	* optimizations and tuning.
	* hRelR     : host side array for relation R.
	* hRelS     : host side array for relation S.
	* agrs      : arguments data structure needed for hash join.
	* cudaParam : data structure storing the cuda parameters
	*/
int global_memory(relation_t *hRelR, relation_t *hRelS, args_t *args, cudaParameters_t *cudaParam) {

  relation_t *relR = (relation_t *) malloc(sizeof(relation_t)); //Device side array for relation R
  relation_t *relS = (relation_t *) malloc(sizeof(relation_t));; //Device side array for relation S

  relation_t *relRn = (relation_t *) malloc(sizeof(relation_t)); //Device side array for partitioned relation R
  relation_t *relSn = (relation_t *) malloc(sizeof(relation_t));; //Device side array for partitioned relation S

  relR->numTuples = hRelR->numTuples;
  relS->numTuples = hRelS->numTuples;

  data *out[2]; //GPU side output buffer
  int *globalPtr[2]; //The global pointer that is used to get the index of the output tuples.

  //allocating memory for output buffer
  cudaMalloc((void **) &out[0], relS->numTuples * sizeof(data));
  cudaMalloc((void **) &out[1], relS->numTuples * sizeof(data));

  //allocating memory for the global pointer
  cudaMalloc((void **) &globalPtr[0], sizeof(data));
  cudaMalloc((void **) &globalPtr[1], sizeof(data));

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
  cudaMemset(globalPtr[0], 0, sizeof(int));
  cudaMemset(globalPtr[1], 0, sizeof(int));

  //initializing all histogram entries to 0
  cudaMemset(rHisto, 0, (args->pCount + 1) * sizeof(int));
  cudaMemset(sHisto, 0, (args->pCount + 1) * sizeof(int));
  cudaMemset(rnHisto, 0, (args->pCount + 1) * sizeof(int));
  cudaMemset(snHisto, 0, (args->pCount + 1) * sizeof(int));

  //setting kernel thread dimensions
  cudaParam->gridSize =
      args->pCountL2; //to avoid the histogram buffer overflow. the size of Histo is pCountL1 * pCountL2
  cudaParam->gridSizeMultiplier = GRID_SIZE_MULTIPLIER;

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //copying Key of relation R to GPU for building the histogram
  cudaMemcpyAsync(relR->key, hRelR->key, relR->numTuples * sizeof(int), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  //building histogram for relation R
  histogram_build_global <<< std::min((int) (relR->numTuples / cudaParam->blockSize), MAX_GRID_SIZE),
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
  histogram_build_global <<< std::min((int) (relS->numTuples / cudaParam->blockSize), MAX_GRID_SIZE),
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
  std::cout << "Partition Stage Execution Time for Optimized Global Memory: " << cudaParam->time << " ms" << std::endl;

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //probe kernel invocation. We assume that the data distribution is uniform.
  probe_global_memory << < std::min(args->pCount / 2, MAX_GRID_SIZE), ceil((float) relS->numTuples / args->pCount), 2
      * ceil((float) relS->numTuples / args->pCount) * sizeof(int), cudaParam->streams[0] >>
      > (relRn->key, relRn->id, relSn->key, relSn->id, rnHisto, snHisto, args->pCount / 2, globalPtr[0], out[0], 0);

  //copying the results back to the CPU main memory. Assuming a 100% match rate.
  cudaMemcpyAsync(args->hOut[0], out[0], relS->numTuples * sizeof(int), cudaMemcpyDeviceToHost, cudaParam->streams[0]);

  //probe kernel invocation. We assume that the data distribution is uniform.
  probe_global_memory << < std::min(args->pCount / 2, MAX_GRID_SIZE), ceil((float) relS->numTuples / args->pCount), 2
      * ceil((float) relS->numTuples / args->pCount) * sizeof(int), cudaParam->streams[1] >>
      > (relRn->key, relRn->id, relSn->key, relSn->id, rnHisto, snHisto, args->pCount, globalPtr[1], out[1],
          args->pCount / 2);

  //copying the results back to the CPU main memory. Assuming a 100% match rate.
  cudaMemcpyAsync(args->hOut[0] + relS->numTuples,
                  out[1],
                  relS->numTuples * sizeof(int),
                  cudaMemcpyDeviceToHost,
                  cudaParam->streams[1]);

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[1]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  //checking for any errors during execution
  check_cuda_error((char *) __FILE__, __LINE__);

  //displaying execution time
  std::cout << "Join Stage Execution Time for Optimized Global Memory: " << cudaParam->time << " ms" << std::endl;

  //debug code
  //displayGPUBuffer(out[1], args->hOut[0], relRn->numTuples);

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

  cudaFree(out[0]);
  cudaFree(out[1]);
  cudaFree(globalPtr[0]);
  cudaFree(globalPtr[1]);

  return 0;
}

/**shared_memory(relation_t * , relation_t * , args_t *, cudaParameters_t )
	* Function to join small relations using shared memory. The function takes advantage of
	* universal virtual addressing feature supported by CUDA.
	* hRelR     : host side array for relation R.
	* hRelS     : host side array for relation S.
	* agrs      : arguments data structure needed for hash join.
	* cudaParam : data structure storing the cuda parameters
	*/
int shared_memory_UVA(relation_t *hRelR, relation_t *hRelS, args_t *args, cudaParameters_t *cudaParam) {

  relation_t *relR = (relation_t *) malloc(sizeof(relation_t)); //Device side array for relation R
  relation_t *relS = (relation_t *) malloc(sizeof(relation_t));; //Device side array for relation S

  relation_t *relRn = (relation_t *) malloc(sizeof(relation_t)); //Device side array for partitioned relation R
  relation_t *relSn = (relation_t *) malloc(sizeof(relation_t));; //Device side array for partitioned relation S

  relR->numTuples = hRelR->numTuples;
  relS->numTuples = hRelS->numTuples;

  //data *out[2]; //GPU side output buffer

  //The global pointer that is used to get the index of the output tuples.
  int *globalPtr[2];

  //allocating memory for the global pointer
  cudaMalloc((void **) &globalPtr[0], sizeof(data));
  cudaMalloc((void **) &globalPtr[1], sizeof(data));

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
  cudaMalloc((void **) &rHisto, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMalloc((void **) &sHisto, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));

  cudaMalloc((void **) &rnHisto, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));
  cudaMalloc((void **) &snHisto, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));

  //setting the global pointer to 0
  cudaMemset(globalPtr[0], 0, sizeof(int));
  cudaMemset(globalPtr[1], 0, sizeof(int));

  //initializing all histogram entries to 0
  cudaMemset(rHisto, 0, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMemset(sHisto, 0, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMemset(rnHisto, 0, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));
  cudaMemset(snHisto, 0, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  cudaMemPrefetchAsync(hRelR->key, relR->numTuples * sizeof(data), 0, cudaParam->streams[0]);

  //building histogram for relation R
  histogram_build_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int),
      cudaParam->streams[0] >> > (hRelR->key, relR->numTuples, args->pCountL1, rHisto);

  //getting the prefix sum of the level 1 histogram for relation R
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(rHisto),
                         thrust::device_pointer_cast(rHisto) + (args->pCountL1 * cudaParam->gridSize + 1),
                         thrust::device_pointer_cast(rHisto));

  cudaMemPrefetchAsync(hRelR->id, relR->numTuples * sizeof(data), 0, cudaParam->streams[1]);

  //making sure that all histogram build and data copy are complete.
  cudaDeviceSynchronize();

  //re-ordering relation R. This is the first level of partitioning
  reorder_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int), cudaParam->streams[0] >>
      > (hRelR->key, hRelR->id, relR->numTuples, args->pCountL1, rHisto, relRn->key, relRn->id);

  cudaMemPrefetchAsync(hRelS->key, relR->numTuples * sizeof(data), 0, cudaParam->streams[2]);

  //building histogram for second level of relation R partitioning
  histogram_build_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2
      * sizeof(int), cudaParam->streams[0] >>
      > (relRn->key, args->pCountL2, args->pCountL1, rnHisto, rHisto, cudaParam->gridSize);

  //building histogram for relation S
  histogram_build_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int),
      cudaParam->streams[2] >> > (hRelS->key, relS->numTuples, args->pCountL1, sHisto);

  //getting the prefix sum of the level 2 histogram for relation R
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(rnHisto),
                         thrust::device_pointer_cast(rnHisto) + (cudaParam->gridSizeMultiplier * args->pCount + 1),
                         thrust::device_pointer_cast(rnHisto));

  //getting the prefix sum of the level 1 histogram for relation S
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[2]),
                         thrust::device_pointer_cast(sHisto),
                         thrust::device_pointer_cast(sHisto) + (args->pCountL1 * cudaParam->gridSize + 1),
                         thrust::device_pointer_cast(sHisto));

  cudaMemPrefetchAsync(hRelS->id, relR->numTuples * sizeof(data), 0, cudaParam->streams[3]);

  //re-ordering relation R. This is the second level of partitioning
  reorder_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2 * sizeof(int),
      cudaParam->streams[0] >>
          > (relRn->key, relRn->id, args->pCountL2, args->pCountL1, rnHisto, rHisto, relR->key, relR->id, cudaParam->gridSize);

  //making sure the data transfer of id values of relation S is complete before re-ordering the realtion.
  cudaStreamSynchronize(cudaParam->streams[3]);

  //re-ordering relation S. This is the first level of partitioning
  reorder_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int), cudaParam->streams[2] >>
      > (hRelS->key, hRelS->key, relS->numTuples, args->pCountL1, sHisto, relSn->key, relSn->id);

  //building histogram for second level of relation S partitioning
  histogram_build_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2
      * sizeof(int), cudaParam->streams[2] >>
      > (relSn->key, args->pCountL2, args->pCountL1, snHisto, sHisto, cudaParam->gridSize);

  //getting the prefix sum of the level 2 histogram for relation S
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[2]),
                         thrust::device_pointer_cast(snHisto),
                         thrust::device_pointer_cast(snHisto) + (cudaParam->gridSizeMultiplier * args->pCount + 1),
                         thrust::device_pointer_cast(snHisto));

  //re-ordering relation S. This is the second level of partitioning
  reorder_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2 * sizeof(int),
      cudaParam->streams[2] >>
          > (relSn->key, relSn->id, args->pCountL2, args->pCountL1, snHisto, sHisto, relS->key, relS->id, cudaParam->gridSize);

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[2]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  //displaying execution time
  std::cout << "Partition Stage Execution for Optimized UVA Join: " << cudaParam->time << " ms" << std::endl;

  //starting time measurement
  /*cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

		//probe kernel invocation. We assume that the data distribution is uniform.
		probe<<<std::min(args->pCount / 2, MAX_GRID_SIZE), ceil((float) relS->numTuples / args->pCount), 2 * ceil((float) relS->numTuples / args->pCount) * sizeof(data), cudaParam->streams[0]>>>(relR->key, relR->id, relS->key, relS->id, rnHisto, snHisto, args->pCount, globalPtr[0], args->hOut[0], 0);

		//ending time measurement
		cudaEventRecord(cudaParam->stop, cudaParam->streams[0]);

		//making sure all CUDA processes are completed before ending the time measurement
		cudaDeviceSynchronize();

		//measuring time
		cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

		//checking for any errors during execution
		check_cuda_error((char *)__FILE__, __LINE__);

		//displaying execution time
		std::cout << "Join Stage Execution Time " << cudaParam->time << " ms" << std::endl;*/

  //debug code
  //displayGPUBuffer(out[1], args->hOut[0], relR->numTuples);

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

  cudaFree(globalPtr[0]);
  cudaFree(globalPtr[1]);

  return 0;
}

/**shared_memory(relation_t * , relation_t * , args_t *, cudaParameters_t )
	* Function to join small relations using shared memory. The function does not use
	* CUDA streams.
	* hRelR     : host side array for relation R.
	* hRelS     : host side array for relation S.
	* agrs      : arguments data structure needed for hash join.
	* cudaParam : data structure storing the cuda parameters
	*/
int shared_memory_UVA_sd(relation_t *hRelR, relation_t *hRelS, args_t *args, cudaParameters_t *cudaParam) {

  relation_t *relR = (relation_t *) malloc(sizeof(relation_t)); //Device side array for relation R
  relation_t *relS = (relation_t *) malloc(sizeof(relation_t));; //Device side array for relation S

  relation_t *relRn = (relation_t *) malloc(sizeof(relation_t)); //Device side array for partitioned relation R
  relation_t *relSn = (relation_t *) malloc(sizeof(relation_t));; //Device side array for partitioned relation S

  relR->numTuples = hRelR->numTuples;
  relS->numTuples = hRelS->numTuples;

  //The global pointer that is used to get the index of the output tuples.
  int *globalPtr[2];

  //allocating memory for the global pointer
  cudaMalloc((void **) &globalPtr[0], sizeof(data));
  cudaMalloc((void **) &globalPtr[1], sizeof(data));

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
  cudaMalloc((void **) &rHisto, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMalloc((void **) &sHisto, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));

  cudaMalloc((void **) &rnHisto, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));
  cudaMalloc((void **) &snHisto, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));

  //setting the global pointer to 0
  cudaMemset(globalPtr[0], 0, sizeof(int));
  cudaMemset(globalPtr[1], 0, sizeof(int));

  //initializing all histogram entries to 0
  cudaMemset(rHisto, 0, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMemset(sHisto, 0, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMemset(rnHisto, 0, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));
  cudaMemset(snHisto, 0, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //building histogram for relation R
  histogram_build_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int),
      cudaParam->streams[0] >> > (hRelR->key, relR->numTuples, args->pCountL1, rHisto);

  //getting the prefix sum of the level 1 histogram for relation R
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(rHisto),
                         thrust::device_pointer_cast(rHisto) + (args->pCountL1 * cudaParam->gridSize + 1),
                         thrust::device_pointer_cast(rHisto));

  //making sure that all histogram build and data copy are complete.
  cudaDeviceSynchronize();

  //re-ordering relation R. This is the first level of partitioning
  reorder_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int), cudaParam->streams[0] >>
      > (hRelR->key, hRelR->id, relR->numTuples, args->pCountL1, rHisto, relRn->key, relRn->id);

  //building histogram for second level of relation R partitioning
  histogram_build_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2
      * sizeof(int), cudaParam->streams[0] >>
      > (relRn->key, args->pCountL2, args->pCountL1, rnHisto, rHisto, cudaParam->gridSize);

  //building histogram for relation S
  histogram_build_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int),
      cudaParam->streams[0] >> > (hRelS->key, relS->numTuples, args->pCountL1, sHisto);

  //getting the prefix sum of the level 2 histogram for relation R
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(rnHisto),
                         thrust::device_pointer_cast(rnHisto) + (cudaParam->gridSizeMultiplier * args->pCount + 1),
                         thrust::device_pointer_cast(rnHisto));

  //getting the prefix sum of the level 1 histogram for relation S
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(sHisto),
                         thrust::device_pointer_cast(sHisto) + (args->pCountL1 * cudaParam->gridSize + 1),
                         thrust::device_pointer_cast(sHisto));

  //re-ordering relation R. This is the second level of partitioning
  reorder_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2 * sizeof(int),
      cudaParam->streams[0] >>
          > (relRn->key, relRn->id, args->pCountL2, args->pCountL1, rnHisto, rHisto, relR->key, relR->id, cudaParam->gridSize);

  //making sure the data transfer of id values of relation S is complete before re-ordering the realtion.
  cudaStreamSynchronize(cudaParam->streams[0]);

  //re-ordering relation S. This is the first level of partitioning
  reorder_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int), cudaParam->streams[0] >>
      > (hRelS->key, hRelS->key, relS->numTuples, args->pCountL1, sHisto, relSn->key, relSn->id);

  //building histogram for second level of relation S partitioning
  histogram_build_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2
      * sizeof(int), cudaParam->streams[0] >>
      > (relSn->key, args->pCountL2, args->pCountL1, snHisto, sHisto, cudaParam->gridSize);

  //getting the prefix sum of the level 2 histogram for relation S
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(snHisto),
                         thrust::device_pointer_cast(snHisto) + (cudaParam->gridSizeMultiplier * args->pCount + 1),
                         thrust::device_pointer_cast(snHisto));

  //re-ordering relation S. This is the second level of partitioning
  reorder_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2 * sizeof(int),
      cudaParam->streams[0] >>
          > (relSn->key, relSn->id, args->pCountL2, args->pCountL1, snHisto, sHisto, relS->key, relS->id, cudaParam->gridSize);

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[0]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  //displaying execution time
  std::cout << "Partition Stage Execution for Optimized UVA Join: " << cudaParam->time << " ms" << std::endl;

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //probe kernel invocation. We assume that the data distribution is uniform.
  probe << < std::min(args->pCount / 2, MAX_GRID_SIZE), ceil((float) relS->numTuples / args->pCount), 2
      * ceil((float) relS->numTuples / args->pCount) * sizeof(int), cudaParam->streams[0] >>
      > (relR->key, relR->id, relS->key, relS->id, rnHisto, snHisto, args->pCount, globalPtr[0], args->hOut[0], 0);

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[0]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  //checking for any errors during execution
  check_cuda_error((char *) __FILE__, __LINE__);

  //displaying execution time
  std::cout << "Join Stage Execution Time " << cudaParam->time << " ms" << std::endl;

  //debug code
  //displayGPUBuffer(out[1], args->hOut[0], relR->numTuples);

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

  cudaFree(globalPtr[0]);
  cudaFree(globalPtr[1]);

  return 0;
}

/**shared_memory(relation_t * , relation_t * , args_t *, cudaParameters_t )
	* Function to join small relations using shared memory based on the 2008 implementation.
	* hRelR     : host side array for relation R.
	* hRelS     : host side array for relation S.
	* agrs      : arguments data structure needed for hash join.
	* cudaParam : data structure storing the cuda parameters
	*/
int shared_memory_2008(relation_t *hRelR, relation_t *hRelS, args_t *args, cudaParameters_t *cudaParam) {

  relation_t *relR = (relation_t *) malloc(sizeof(relation_t)); //Device side array for relation R
  relation_t *relS = (relation_t *) malloc(sizeof(relation_t));; //Device side array for relation S

  relation_t *relRn = (relation_t *) malloc(sizeof(relation_t)); //Device side array for partitioned relation R
  relation_t *relSn = (relation_t *) malloc(sizeof(relation_t));; //Device side array for partitioned relation S

  relR->numTuples = hRelR->numTuples;
  relS->numTuples = hRelS->numTuples;

  data *out[2]; //GPU side output buffer
  int *globalPtr[2]; //The global pointer that is used to get the index of the output tuples.

  //allocating memory for output buffer
  cudaMalloc((void **) &out[0], 2 * relS->numTuples * sizeof(data));
  cudaMalloc((void **) &out[1], 2 * relS->numTuples * sizeof(data));

  //allocating memory for the global pointer
  cudaMalloc((void **) &globalPtr[0], sizeof(data));
  cudaMalloc((void **) &globalPtr[1], sizeof(data));

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

  //setting kernel thread dimensions
  cudaParam->gridSize = 1024; //to avoid the histogram buffer overflow. the size of Histo is pCountL1 * pCountL2
  cudaParam->blockSize = 32 * 1024 / (4 * args->pCountL1);

  //allocating device side memory for histogram. An additional entry is required for the last partition.
  cudaMalloc((void **) &rHisto, (cudaParam->gridSize * cudaParam->blockSize * args->pCountL1 + 1) * sizeof(int));
  cudaMalloc((void **) &sHisto, (cudaParam->gridSize * cudaParam->blockSize * args->pCountL1 + 1) * sizeof(int));

  cudaMalloc((void **) &rnHisto,
             (cudaParam->gridSizeMultiplier * cudaParam->blockSize * args->pCount + 1) * sizeof(int));
  cudaMalloc((void **) &snHisto,
             (cudaParam->gridSizeMultiplier * cudaParam->blockSize * args->pCount + 1) * sizeof(int));

  //setting the global pointer to 0
  cudaMemset(globalPtr[0], 0, sizeof(int));
  cudaMemset(globalPtr[1], 0, sizeof(int));

  //initializing all histogram entries to 0
  cudaMemset(rHisto, 0, (cudaParam->gridSize * cudaParam->blockSize * args->pCountL1 + 1) * sizeof(int));
  cudaMemset(sHisto, 0, (cudaParam->gridSize * cudaParam->blockSize * args->pCountL1 + 1) * sizeof(int));
  cudaMemset(rnHisto, 0, (cudaParam->gridSizeMultiplier * cudaParam->blockSize * args->pCount + 1) * sizeof(int));
  cudaMemset(snHisto, 0, (cudaParam->gridSizeMultiplier * cudaParam->blockSize * args->pCount + 1) * sizeof(int));

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //copying Key of relation R to GPU for building the histogram
  cudaMemcpyAsync(relR->key, hRelR->key, relR->numTuples * sizeof(int), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  //building histogram for relation R
  histogram_build_L1_2008 << < cudaParam->gridSize, cudaParam->blockSize, cudaParam->blockSize * args->pCountL1
      * sizeof(int), cudaParam->streams[0] >> > (relR->key, relR->numTuples, args->pCountL1, rHisto);

  //getting the prefix sum of the level 1 histogram for relation R
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(rHisto),
                         thrust::device_pointer_cast(rHisto)
                             + (cudaParam->gridSize * cudaParam->blockSize * args->pCountL1 + 1),
                         thrust::device_pointer_cast(rHisto));

  //copying id of relation R to GPU
  cudaMemcpyAsync(relR->id, hRelR->id, relR->numTuples * sizeof(int), cudaMemcpyHostToDevice, cudaParam->streams[1]);

  //making sure that all histogram build and data copy are complete.
  cudaDeviceSynchronize();

  //re-ordering relation R. This is the first level of partitioning
  reorder_L1_2008 << < cudaParam->gridSize, cudaParam->blockSize, cudaParam->blockSize * args->pCountL1 * sizeof(int),
      cudaParam->streams[0] >> > (relR->key, relR->id, relR->numTuples, args->pCountL1, rHisto, relRn->key, relRn->id);

  //copying Key of relation S to GPU for building the histogram
  cudaMemcpyAsync(relS->key, hRelS->key, relS->numTuples * sizeof(int), cudaMemcpyHostToDevice, cudaParam->streams[2]);

  //building histogram for second level of relation R partitioning
  histogram_build_L2_2008 << < args->pCountL1, cudaParam->blockSize, cudaParam->blockSize * args->pCountL2
      * sizeof(int), cudaParam->streams[0] >>
      > (relRn->key, args->pCountL2, args->pCountL1, rnHisto, rHisto, cudaParam->gridSize, cudaParam->blockSize);

  //building histogram for relation S
  histogram_build_L1_2008 << < cudaParam->gridSize, cudaParam->blockSize, cudaParam->blockSize * args->pCountL1
      * sizeof(int), cudaParam->streams[2] >> > (relS->key, relS->numTuples, args->pCountL1, sHisto);

  //getting the prefix sum of the level 2 histogram for relation R
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(rnHisto),
                         thrust::device_pointer_cast(rnHisto) + (cudaParam->blockSize * args->pCount + 1),
                         thrust::device_pointer_cast(rnHisto));

  //getting the prefix sum of the level 1 histogram for relation S
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[2]),
                         thrust::device_pointer_cast(sHisto),
                         thrust::device_pointer_cast(sHisto)
                             + (cudaParam->gridSize * cudaParam->blockSize * args->pCountL1 + 1),
                         thrust::device_pointer_cast(sHisto));

  //copying id of relation S to GPU
  cudaMemcpyAsync(relS->id, hRelS->id, relS->numTuples * sizeof(int), cudaMemcpyHostToDevice, cudaParam->streams[3]);

  //re-ordering relation R. This is the second level of partitioning
  reorder_L2_2008 << < args->pCountL1, cudaParam->blockSize, cudaParam->blockSize * args->pCountL2 * sizeof(int),
      cudaParam->streams[0] >>
          > (relRn->key, relRn->id, args->pCountL2, args->pCountL1, rnHisto, rHisto, relR->key, relR->id, cudaParam->gridSize, cudaParam->blockSize);

  //making sure the data transfer of id values of relation S is complete before re-ordering the realtion.
  cudaStreamSynchronize(cudaParam->streams[3]);

  //re-ordering relation S. This is the first level of partitioning
  reorder_L1_2008 << < cudaParam->gridSize, cudaParam->blockSize, cudaParam->blockSize * args->pCountL1 * sizeof(int),
      cudaParam->streams[2] >> > (relS->key, relS->id, relS->numTuples, args->pCountL1, sHisto, relSn->key, relSn->id);

  //building histogram for second level of relation S partitioning
  histogram_build_L2_2008 << < args->pCountL1, cudaParam->blockSize, cudaParam->blockSize * args->pCountL2
      * sizeof(int), cudaParam->streams[2] >>
      > (relSn->key, args->pCountL2, args->pCountL1, snHisto, sHisto, cudaParam->gridSize, cudaParam->blockSize);

  //getting the prefix sum of the level 2 histogram for relation S
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[2]),
                         thrust::device_pointer_cast(snHisto),
                         thrust::device_pointer_cast(snHisto) + (cudaParam->blockSize * args->pCount + 1),
                         thrust::device_pointer_cast(snHisto));

  //re-ordering relation S. This is the second level of partitioning
  reorder_L2_2008 << < args->pCountL1, cudaParam->blockSize, cudaParam->blockSize * args->pCountL2 * sizeof(int),
      cudaParam->streams[2] >>
          > (relSn->key, relSn->id, args->pCountL2, args->pCountL1, snHisto, sHisto, relS->key, relS->id, cudaParam->gridSize, cudaParam->blockSize);

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[2]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  //displaying execution time
  std::cout << "Partition Stage Execution Time for Optimized Shared Memory: " << cudaParam->time << " ms" << std::endl;

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //probe kernel invocation. We assume that the data distribution is uniform.
  /*probe_2008<<<std::min(args->pCount / 2, MAX_GRID_SIZE), ceil((float) relS->numTuples / args->pCount), 2 * ceil((float) relS->numTuples / args->pCount) * sizeof(int), cudaParam->streams[0]>>>(relR->key, relR->id, relS->key, relS->id, rnHisto, snHisto, args->pCount / 2, globalPtr[0], out[0], 0, cudaParam->blockSize);

		//copying the results back to the CPU main memory. Assuming a 100% match rate.
		cudaMemcpyAsync(args->hOut[0], out[0], relS->numTuples * sizeof(int), cudaMemcpyDeviceToHost, cudaParam->streams[0]);

		//second probe kernel invocation
		probe_2008<<<std::min(args->pCount / 2, MAX_GRID_SIZE), ceil((float) relS->numTuples / args->pCount), 2 * ceil((float) relS->numTuples / args->pCount) * sizeof(int), cudaParam->streams[1]>>>(relR->key, relR->id, relS->key, relS->id, rnHisto, snHisto, args->pCount, globalPtr[1], out[1], args->pCount / 2, cudaParam->blockSize);

		//copying the second set of results back to the CPU main memory. Assuming a 100% match rate.
		cudaMemcpyAsync(args->hOut[0] + relS->numTuples, out[1], relS->numTuples * sizeof(int), cudaMemcpyDeviceToHost, cudaParam->streams[1]);*/

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[1]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  //checking for any errors during execution
  check_cuda_error((char *) __FILE__, __LINE__);

  //displaying execution time
  std::cout << "Join Stage Execution Time " << cudaParam->time << " ms" << std::endl;

  //debug code
  //displayGPUBuffer(out[0], args->hOut[0], relR->numTuples);

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

  cudaFree(out[0]);
  cudaFree(out[1]);
  cudaFree(globalPtr[0]);
  cudaFree(globalPtr[1]);

  return 0;
}

/**shared_memory(relation_t * , relation_t * , args_t *, cudaParameters_t )
	* Function to join small relations using shared memory. The function takes advantage of
	* universal virtual addressing feature supported by CUDA.
	* hRelR     : host side array for relation R.
	* hRelS     : host side array for relation S.
	* agrs      : arguments data structure needed for hash join.
	* cudaParam : data structure storing the cuda parameters
	*/
int shared_memory_UVA1(relation_t *hRelR,
                       relation_t *hRelS,
                       args_t *args,
                       cudaParameters_t *cudaParam,
                       relation_t *hRelRn,
                       relation_t *hRelSn) {
  relation_t *relR = (relation_t *) malloc(sizeof(relation_t)); //Device side array for relation R
  relation_t *relS = (relation_t *) malloc(sizeof(relation_t));; //Device side array for relation S

  relation_t *relRn = (relation_t *) malloc(sizeof(relation_t)); //Device side array for partitioned relation R
  relation_t *relSn = (relation_t *) malloc(sizeof(relation_t));; //Device side array for partitioned relation S

  relR->numTuples = hRelR->numTuples;
  relS->numTuples = hRelS->numTuples;

  //The global pointer that is used to get the index of the output tuples.
  int *globalPtr[2];

  //allocating memory for the global pointer
  cudaMalloc((void **) &globalPtr[0], sizeof(data));
  cudaMalloc((void **) &globalPtr[1], sizeof(data));

  //allocating device memory for storing input data
  /*cudaMalloc((void **)&relR->id, relR->numTuples * sizeof(data));
		cudaMalloc((void **)&relR->key, relR->numTuples * sizeof(data));
		cudaMalloc((void **)&relS->id, relS->numTuples * sizeof(data));
		cudaMalloc((void **)&relS->key, relS->numTuples * sizeof(data));

		//allocating device memory for storing partitioned data
		relRn->numTuples = relR->numTuples;
		relSn->numTuples = relS->numTuples;

		cudaMalloc((void **)&relRn->id, relRn->numTuples * sizeof(data));
		cudaMalloc((void **)&relRn->key, relRn->numTuples * sizeof(data));
		cudaMalloc((void **)&relSn->id, relSn->numTuples * sizeof(data));
		cudaMalloc((void **)&relSn->key, relSn->numTuples * sizeof(data));*/

  //declaring device side histogram data
  int *rHisto, *sHisto;
  int *rnHisto, *snHisto; //To allow for histogram update during re-order.

  //allocating device side memory for histogram. An additional entry is required for the last partition.
  cudaMalloc((void **) &rHisto, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMalloc((void **) &sHisto, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));

  cudaMalloc((void **) &rnHisto, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));
  cudaMalloc((void **) &snHisto, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));

  //setting the global pointer to 0
  cudaMemset(globalPtr[0], 0, sizeof(int));
  cudaMemset(globalPtr[1], 0, sizeof(int));

  //initializing all histogram entries to 0
  cudaMemset(rHisto, 0, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMemset(sHisto, 0, (cudaParam->gridSize * args->pCountL1 + 1) * sizeof(int));
  cudaMemset(rnHisto, 0, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));
  cudaMemset(snHisto, 0, (cudaParam->gridSizeMultiplier * args->pCount + 1) * sizeof(int));

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //building histogram for relation R
  histogram_build_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int),
      cudaParam->streams[0] >> > (hRelR->key, relR->numTuples, args->pCountL1, rHisto);

  //getting the prefix sum of the level 1 histogram for relation R
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(rHisto),
                         thrust::device_pointer_cast(rHisto) + (args->pCountL1 * cudaParam->gridSize + 1),
                         thrust::device_pointer_cast(rHisto));

  //making sure that all histogram build and data copy are complete.
  cudaDeviceSynchronize();

  //re-ordering relation R. This is the first level of partitioning
  reorder_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int), cudaParam->streams[0] >>
      > (hRelR->key, hRelR->id, relR->numTuples, args->pCountL1, rHisto, hRelRn->key, hRelRn->id);

  //building histogram for second level of relation R partitioning
  histogram_build_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2
      * sizeof(int), cudaParam->streams[0] >>
      > (hRelRn->key, args->pCountL2, args->pCountL1, rnHisto, rHisto, cudaParam->gridSize);

  //building histogram for relation S
  histogram_build_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int),
      cudaParam->streams[2] >> > (hRelS->key, relS->numTuples, args->pCountL1, sHisto);

  //getting the prefix sum of the level 2 histogram for relation R
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[0]),
                         thrust::device_pointer_cast(rnHisto),
                         thrust::device_pointer_cast(rnHisto) + (cudaParam->gridSizeMultiplier * args->pCount + 1),
                         thrust::device_pointer_cast(rnHisto));

  //getting the prefix sum of the level 1 histogram for relation S
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[2]),
                         thrust::device_pointer_cast(sHisto),
                         thrust::device_pointer_cast(sHisto) + (args->pCountL1 * cudaParam->gridSize + 1),
                         thrust::device_pointer_cast(sHisto));

  //re-ordering relation R. This is the second level of partitioning
  reorder_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2 * sizeof(int),
      cudaParam->streams[0] >>
          > (hRelRn->key, hRelRn->id, args->pCountL2, args->pCountL1, rnHisto, rHisto, hRelR->key, hRelR->id, cudaParam->gridSize);

  //making sure the data transfer of id values of relation S is complete before re-ordering the realtion.
  cudaStreamSynchronize(cudaParam->streams[3]);

  //re-ordering relation S. This is the first level of partitioning
  reorder_L1 << < cudaParam->gridSize, cudaParam->blockSize, args->pCountL1 * sizeof(int), cudaParam->streams[2] >>
      > (hRelS->key, hRelS->key, relS->numTuples, args->pCountL1, sHisto, hRelSn->key, hRelSn->id);

  //building histogram for second level of relation S partitioning
  histogram_build_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2
      * sizeof(int), cudaParam->streams[2] >>
      > (hRelSn->key, args->pCountL2, args->pCountL1, snHisto, sHisto, cudaParam->gridSize);

  //getting the prefix sum of the level 2 histogram for relation S
  thrust::exclusive_scan(thrust::cuda::par.on(cudaParam->streams[2]),
                         thrust::device_pointer_cast(snHisto),
                         thrust::device_pointer_cast(snHisto) + (cudaParam->gridSizeMultiplier * args->pCount + 1),
                         thrust::device_pointer_cast(snHisto));

  //re-ordering relation S. This is the second level of partitioning
  reorder_L2 << < cudaParam->gridSizeMultiplier * args->pCountL1, cudaParam->blockSize, args->pCountL2 * sizeof(int),
      cudaParam->streams[2] >>
          > (hRelSn->key, hRelSn->id, args->pCountL2, args->pCountL1, snHisto, sHisto, hRelS->key, hRelS->id, cudaParam->gridSize);

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[2]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  //displaying execution time
  std::cout << "Partition Stage Execution for Optimized UVA Join: " << cudaParam->time << " ms" << std::endl;

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //probe kernel invocation. We assume that the data distribution is uniform.
  probe << < std::min(args->pCount / 2, MAX_GRID_SIZE), ceil((float) relS->numTuples / args->pCount), 2
      * ceil((float) relS->numTuples / args->pCount) * sizeof(data), cudaParam->streams[0] >>
      > (hRelR->key, hRelR->id, hRelS->key, hRelS->id, rnHisto, snHisto, args->pCount, globalPtr[0], args->hOut[0], 0);

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[0]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  //checking for any errors during execution
  check_cuda_error((char *) __FILE__, __LINE__);

  //displaying execution time
  std::cout << "Join Stage Execution Time " << cudaParam->time << " ms" << std::endl;

  //debug code
  //displayGPUBuffer(out[1], args->hOut[0], relR->numTuples);

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

  cudaFree(globalPtr[0]);
  cudaFree(globalPtr[1]);

  return 0;
}

int UVA_benchmark1(data *input1, data *input2, int len, cudaParameters_t *cudaParam) {

  data *output1;
  data *output2;
  cudaMalloc((void **) &output1, len * sizeof(data));
  cudaMalloc((void **) &output2, len * sizeof(data));

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  UVA_benchmark << < cudaParam->gridSize, cudaParam->blockSize >> > (input1, output1);
  //UVA_benchmark<<<1024,1024>>>(input2, output2, len);

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[0]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  std::cout << "Time1: " << cudaParam->time << " ms" << std::endl;

  return 0;
}

int UVA_benchmark2(data *input, data *input0, int len, cudaParameters_t *cudaParam) {

  data *output1;
  data *output2;
  data *input1;
  data *input2;
  cudaMalloc((void **) &output1, len * sizeof(data));
  cudaMalloc((void **) &output2, len * sizeof(data));
  cudaMalloc((void **) &input1, len * sizeof(data));
  cudaMalloc((void **) &input2, len * sizeof(data));

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  cudaMemcpyAsync(input1, input, len * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[0]);
  /*cudaMemcpyAsync(input1, input, len * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[0]);
		UVA_benchmark<<<1024,1024, 0, cudaParam->streams[0]>>>(input1, output1, len);
		cudaMemcpyAsync(input2, input0, len * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[1]);
		UVA_benchmark<<<1024,1024, 0, cudaParam->streams[0]>>>(input2, output2, len);*/

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[0]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  std::cout << "Time2 " << cudaParam->time << " ms" << std::endl;

  return 0;
}

int simple_hash_join(relation_t *hRelR, relation_t *hRelS, args_t *args, cudaParameters_t *cudaParam) {

  relation_t *relR = (relation_t *) malloc(sizeof(relation_t)); //Device side array for relation R
  relation_t *relS = (relation_t *) malloc(sizeof(relation_t));; //Device side array for relation S

  relation_t *relRn = (relation_t *) malloc(sizeof(relation_t)); //Device side array for partitioned relation R
  relation_t *relSn = (relation_t *) malloc(sizeof(relation_t));; //Device side array for partitioned relation S

  relR->numTuples = hRelR->numTuples;
  relS->numTuples = hRelS->numTuples;

  data *out[2]; //GPU side output buffer
  int *globalPtr[2]; //The global pointer that is used to get the index of the output tuples.

  //allocating memory for output buffer
  cudaMalloc((void **) &out[0], 2 * relS->numTuples * sizeof(data));
  cudaMalloc((void **) &out[1], 2 * relS->numTuples * sizeof(data));

  //allocating memory for the global pointer
  cudaMalloc((void **) &globalPtr[0], sizeof(data));
  cudaMalloc((void **) &globalPtr[1], sizeof(data));

  //allocating device memory for storing input data
  cudaMalloc((void **) &relR->id, relR->numTuples * sizeof(data));
  cudaMalloc((void **) &relR->key, relR->numTuples * sizeof(data));
  cudaMalloc((void **) &relS->id, relS->numTuples * sizeof(data));
  cudaMalloc((void **) &relS->key, relS->numTuples * sizeof(data));

  //allocating device memory for storing partitioned data
  relRn->numTuples = relR->numTuples;
  relSn->numTuples = relS->numTuples;

  cudaMalloc((void **) &relRn->id, 2 * (relRn->numTuples + 2 * args->pCount) * sizeof(data));

  //setting the global pointer to 0
  cudaMemset(globalPtr[0], 0, sizeof(int));
  cudaMemset(globalPtr[1], 0, sizeof(int));

  //initializing all histogram entries to 0
  cudaMemset(relRn->id, 0, 2 * (relRn->numTuples + 2 * args->pCount) * sizeof(data));

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //copying Key of relation R to GPU for building the histogram
  cudaMemcpyAsync(relR->key, hRelR->key, relR->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  cudaMemcpyAsync(relR->id, hRelR->id, relR->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[1]);

  cudaDeviceSynchronize();

  build_kernel << < cudaParam->gridSize, cudaParam->blockSize, 0, cudaParam->streams[0] >> > (relR->id, relR->key, relRn->id, relR->numTuples, args->pCount);

  //copying Key of relation R to GPU for building the histogram
  cudaMemcpyAsync(relS->key, hRelS->key, relS->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[1]);

  cudaMemcpyAsync(relS->id, hRelS->id, relS->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[2]);

  cudaDeviceSynchronize();

  probe_kernel_sm << < cudaParam->gridSize, cudaParam->blockSize, 0, cudaParam->streams[0] >>
      > (relRn->id, relS->id, relS->key, out[0], relR->numTuples, relS->numTuples / 2, args->pCount, globalPtr[0]);

  cudaMemcpyAsync(args->hOut[0], out[0], relS->numTuples * sizeof(data), cudaMemcpyDeviceToHost, cudaParam->streams[0]);

  probe_kernel_sm << < cudaParam->gridSize, cudaParam->blockSize, 0, cudaParam->streams[1] >>
      > (relRn->id, relS->id + relS->numTuples / 2, relS->key + relS->numTuples / 2, out[1], relR->numTuples,
          relS->numTuples / 2, args->pCount, globalPtr[1]);

  cudaMemcpyAsync(args->hOut[0], out[1], relS->numTuples * sizeof(data), cudaMemcpyDeviceToHost, cudaParam->streams[1]);

  cudaDeviceSynchronize();

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[1]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  check_cuda_error((char *) __FILE__, __LINE__);

  std::cout << "Simple Hash Join Time Probe Stage: " << cudaParam->time << " ms" << std::endl;

  //displayGPUBuffer(out[0], args->hOut[0], 100);

  return 0;
}

int simple_hash_join_GM(relation_t *hRelR, relation_t *hRelS, args_t *args, cudaParameters_t *cudaParam) {

  relation_t *relR = (relation_t *) malloc(sizeof(relation_t)); //Device side array for relation R
  relation_t *relS = (relation_t *) malloc(sizeof(relation_t));; //Device side array for relation S

  relation_t *relRn = (relation_t *) malloc(sizeof(relation_t)); //Device side array for partitioned relation R
  relation_t *relSn = (relation_t *) malloc(sizeof(relation_t));; //Device side array for partitioned relation S

  relR->numTuples = hRelR->numTuples;
  relS->numTuples = hRelS->numTuples;

  data *out[2]; //GPU side output buffer
  int *globalPtr[2]; //The global pointer that is used to get the index of the output tuples.

  //allocating memory for output buffer
  cudaMalloc((void **) &out[0], 2 * relS->numTuples * sizeof(data));
  cudaMalloc((void **) &out[1], 2 * relS->numTuples * sizeof(data));

  //allocating memory for the global pointer
  cudaMalloc((void **) &globalPtr[0], sizeof(data));
  cudaMalloc((void **) &globalPtr[1], sizeof(data));

  //allocating device memory for storing input data
  cudaMalloc((void **) &relR->id, relR->numTuples * sizeof(data));
  cudaMalloc((void **) &relR->key, relR->numTuples * sizeof(data));
  cudaMalloc((void **) &relS->id, relS->numTuples * sizeof(data));
  cudaMalloc((void **) &relS->key, relS->numTuples * sizeof(data));

  //allocating device memory for storing partitioned data
  relRn->numTuples = relR->numTuples;
  relSn->numTuples = relS->numTuples;

  cudaMalloc((void **) &relRn->id, 2 * (relRn->numTuples + 2 * args->pCount) * sizeof(data));

  //setting the global pointer to 0
  cudaMemset(globalPtr[0], 0, sizeof(int));
  cudaMemset(globalPtr[1], 0, sizeof(int));

  //initializing all histogram entries to 0
  cudaMemset(relRn->id, 0, 2 * (relRn->numTuples + 2 * args->pCount) * sizeof(data));

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //copying Key of relation R to GPU for building the histogram
  cudaMemcpyAsync(relR->key, hRelR->key, relR->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  cudaMemcpyAsync(relR->id, hRelR->id, relR->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[1]);

  cudaDeviceSynchronize();

  build_kernel << < cudaParam->gridSize, cudaParam->blockSize, 0, cudaParam->streams[0] >> > (relR->id, relR->key, relRn->id, relR->numTuples, args->pCount);

  //copying Key of relation R to GPU for building the histogram
  cudaMemcpyAsync(relS->key, hRelS->key, relS->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[1]);

  cudaMemcpyAsync(relS->id, hRelS->id, relS->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[2]);

  cudaDeviceSynchronize();

  probe_kernel << < cudaParam->gridSize, cudaParam->blockSize, 0, cudaParam->streams[0] >>
      > (relRn->id, relS->id, relS->key, out[0], relR->numTuples, relS->numTuples / 2, args->pCount, globalPtr[0]);

  cudaMemcpyAsync(args->hOut[0], out[0], relS->numTuples * sizeof(data), cudaMemcpyDeviceToHost, cudaParam->streams[0]);

  probe_kernel << < cudaParam->gridSize, cudaParam->blockSize, 0, cudaParam->streams[1] >>
      > (relRn->id, relS->id + relS->numTuples / 2, relS->key + relS->numTuples / 2, out[1], relR->numTuples,
          relS->numTuples / 2, args->pCount, globalPtr[1]);

  cudaMemcpyAsync(args->hOut[0], out[1], relS->numTuples * sizeof(data), cudaMemcpyDeviceToHost, cudaParam->streams[1]);

  cudaDeviceSynchronize();

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[1]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  check_cuda_error((char *) __FILE__, __LINE__);

  std::cout << "Simple Hash Join Time Probe Stage: " << cudaParam->time << " ms" << std::endl;

  //displayGPUBuffer(out[0], args->hOut[0], 100);

  return 0;
}

int simple_hash_join_SD(relation_t *hRelR, relation_t *hRelS, args_t *args, cudaParameters_t *cudaParam) {

  relation_t *relR = (relation_t *) malloc(sizeof(relation_t)); //Device side array for relation R
  relation_t *relS = (relation_t *) malloc(sizeof(relation_t));; //Device side array for relation S

  relation_t *relRn = (relation_t *) malloc(sizeof(relation_t)); //Device side array for partitioned relation R
  relation_t *relSn = (relation_t *) malloc(sizeof(relation_t));; //Device side array for partitioned relation S

  relR->numTuples = hRelR->numTuples;
  relS->numTuples = hRelS->numTuples;

  data *out[2]; //GPU side output buffer
  int *globalPtr[2]; //The global pointer that is used to get the index of the output tuples.

  //allocating memory for output buffer
  cudaMalloc((void **) &out[0], 2 * relS->numTuples * sizeof(data));
  cudaMalloc((void **) &out[1], 2 * relS->numTuples * sizeof(data));

  //allocating memory for the global pointer
  cudaMalloc((void **) &globalPtr[0], sizeof(data));
  cudaMalloc((void **) &globalPtr[1], sizeof(data));

  //allocating device memory for storing input data
  cudaMalloc((void **) &relR->id, relR->numTuples * sizeof(data));
  cudaMalloc((void **) &relR->key, relR->numTuples * sizeof(data));
  cudaMalloc((void **) &relS->id, relS->numTuples * sizeof(data));
  cudaMalloc((void **) &relS->key, relS->numTuples * sizeof(data));

  //allocating device memory for storing partitioned data
  relRn->numTuples = relR->numTuples;
  relSn->numTuples = relS->numTuples;

  cudaMalloc((void **) &relRn->id, 2 * (relRn->numTuples + 2 * args->pCount) * sizeof(data));

  //setting the global pointer to 0
  cudaMemset(globalPtr[0], 0, sizeof(int));
  cudaMemset(globalPtr[1], 0, sizeof(int));

  //initializing all histogram entries to 0
  cudaMemset(relRn->id, 0, 2 * (relRn->numTuples + 2 * args->pCount) * sizeof(data));

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  //copying Key of relation R to GPU for building the histogram
  cudaMemcpyAsync(relR->key, hRelR->key, relR->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  cudaMemcpyAsync(relR->id, hRelR->id, relR->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  cudaDeviceSynchronize();

  build_kernel << < cudaParam->gridSize, cudaParam->blockSize, 0, cudaParam->streams[0] >> > (relR->id, relR->key, relRn->id, relR->numTuples, args->pCount);

  //copying Key of relation R to GPU for building the histogram
  cudaMemcpyAsync(relS->key, hRelS->key, relS->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  cudaMemcpyAsync(relS->id, hRelS->id, relS->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  cudaDeviceSynchronize();

  probe_kernel_sm << < cudaParam->gridSize, cudaParam->blockSize, 0, cudaParam->streams[0] >>
      > (relRn->id, relS->id, relS->key, out[0], relR->numTuples, relS->numTuples, args->pCount, globalPtr[0]);

  cudaMemcpyAsync(args->hOut[0],
                  out[0],
                  2 * relS->numTuples * sizeof(data),
                  cudaMemcpyDeviceToHost,
                  cudaParam->streams[0]);

  cudaDeviceSynchronize();

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[0]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  check_cuda_error((char *) __FILE__, __LINE__);

  std::cout << "Simple Hash Join Time Probe Stage: " << cudaParam->time << " ms" << std::endl;

  //displayGPUBuffer(out[0], args->hOut[0], 100);

  return 0;
}

int simple_hash_join_SD_PT(relation_t *hRelR, relation_t *hRelS, args_t *args, cudaParameters_t *cudaParam) {

  relation_t *relR = (relation_t *) malloc(sizeof(relation_t)); //Device side array for relation R
  relation_t *relS = (relation_t *) malloc(sizeof(relation_t));; //Device side array for relation S

  relation_t *relRn = (relation_t *) malloc(sizeof(relation_t)); //Device side array for partitioned relation R
  relation_t *relSn = (relation_t *) malloc(sizeof(relation_t));; //Device side array for partitioned relation S

  relR->numTuples = hRelR->numTuples;
  relS->numTuples = hRelS->numTuples;

  data *out[2]; //GPU side output buffer
  int *globalPtr[2]; //The global pointer that is used to get the index of the output tuples.

  //allocating memory for output buffer
  cudaMalloc((void **) &out[0], 2 * relS->numTuples * sizeof(data));
  cudaMalloc((void **) &out[1], 2 * relS->numTuples * sizeof(data));

  //allocating memory for the global pointer
  cudaMalloc((void **) &globalPtr[0], sizeof(data));
  cudaMalloc((void **) &globalPtr[1], sizeof(data));

  //allocating device memory for storing input data
  cudaMalloc((void **) &relR->id, relR->numTuples * sizeof(data));
  cudaMalloc((void **) &relR->key, relR->numTuples * sizeof(data));
  cudaMalloc((void **) &relS->id, relS->numTuples * sizeof(data));
  cudaMalloc((void **) &relS->key, relS->numTuples * sizeof(data));

  //allocating device memory for storing partitioned data
  relRn->numTuples = relR->numTuples;
  relSn->numTuples = relS->numTuples;

  cudaMalloc((void **) &relRn->id, 2 * (relRn->numTuples + 2 * args->pCount) * sizeof(data));

  //setting the global pointer to 0
  cudaMemset(globalPtr[0], 0, sizeof(int));
  cudaMemset(globalPtr[1], 0, sizeof(int));

  //initializing all histogram entries to 0
  cudaMemset(relRn->id, 0, 2 * (relRn->numTuples + 2 * args->pCount) * sizeof(data));

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();

  //copying Key of relation R to GPU for building the histogram
  cudaMemcpyAsync(relR->key, hRelR->key, relR->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  cudaMemcpyAsync(relR->id, hRelR->id, relR->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  cudaDeviceSynchronize();

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  build_kernel << < cudaParam->gridSize, cudaParam->blockSize, 0, cudaParam->streams[0] >>> (relR->id, relR->key, relRn->id, relR->numTuples, args->pCount);

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[0]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  check_cuda_error((char *) __FILE__, __LINE__);

  std::cout << "Simple Hash Join Time Partition Stage: "
            << args->pCount << " "
            << cudaParam->blockSize << " "
            << cudaParam->gridSize << " "
            << cudaParam->time << std::endl;

  //displayGPUBuffer(out[0], args->hOut[0], 100);

  return 0;
}
}

namespace eth {

__global__ void build_kernel_compressed(data *rTableID,
                                        data *rHashTable,
                                        int rTupleNum,
                                        int rHashTableBucketNum,
                                        uint32_t shiftBits) {
  int numWorkItems = gridDim.x * blockDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int key, hash, count;
  int hashBucketSize = rTupleNum / rHashTableBucketNum; //number of tuples inserted into each bucket

  while (tid < rTupleNum) {
    //phase 1
    key = rTableID[tid]; //get the key of one tuple
    hash = key >> shiftBits % rHashTableBucketNum;

    //phase 2
    if ((count = atomicAdd(&rHashTable[hash * hashBucketSize], 1)) < hashBucketSize) {
      rHashTable[hash * hashBucketSize + count] = key;
    }

    tid += numWorkItems;
  }
}

int simple_hash_join_compressed(relation_t *hRelR,
                                relation_t *hRelS,
                                args_t *args,
                                uint32_t shiftBits,
                                uint32_t keyShift) {
  cudaParameters_t * cudaParam = (cudaParameters_t *) malloc(sizeof(cudaParameters_t));

  relation_t *relR = (relation_t *) malloc(sizeof(relation_t)); //Device side array for relation R
  relation_t *relS = (relation_t *) malloc(sizeof(relation_t));; //Device side array for relation S

  relation_t *relRn = (relation_t *) malloc(sizeof(relation_t)); //Device side array for partitioned relation R
  relation_t *relSn = (relation_t *) malloc(sizeof(relation_t));; //Device side array for partitioned relation S

  relR->numTuples = hRelR->numTuples;
  relS->numTuples = hRelS->numTuples;

  data *out[2]; //GPU side output buffer
  int *globalPtr[2]; //The global pointer that is used to get the index of the output tuples.

  //allocating memory for output buffer
  cudaMalloc((void **) &out[0], 2 * relS->numTuples * sizeof(data));
  cudaMalloc((void **) &out[1], 2 * relS->numTuples * sizeof(data));

  //allocating memory for the global pointer
  cudaMalloc((void **) &globalPtr[0], sizeof(data));
  cudaMalloc((void **) &globalPtr[1], sizeof(data));

  //allocating device memory for storing input data
  cudaMalloc((void **) &relR->id, relR->numTuples * sizeof(data));
  cudaMalloc((void **) &relS->id, relS->numTuples * sizeof(data));

  //allocating device memory for storing partitioned data
  relRn->numTuples = relR->numTuples;
  relSn->numTuples = relS->numTuples;

  cudaMalloc((void **) &relRn->id, 2 * (relRn->numTuples + 2 * args->pCount) * sizeof(data));

  //setting the global pointer to 0
  cudaMemset(globalPtr[0], 0, sizeof(int));
  cudaMemset(globalPtr[1], 0, sizeof(int));

  //initializing all histogram entries to 0
  cudaMemset(relRn->id, 0, 2 * (relRn->numTuples + 2 * args->pCount) * sizeof(data));

  //makign sure all cuda instruction before this point are completed before starting the time measurement
  cudaDeviceSynchronize();

  //starting time measurement
  cudaEventRecord(cudaParam->start, cudaParam->streams[0]);

  cudaMemcpyAsync(relR->id, hRelR->id, relR->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  cudaDeviceSynchronize();

  build_kernel_compressed << < cudaParam->gridSize, cudaParam->blockSize, 0, cudaParam->streams[0] >> >
      (relR->id, relRn->id, relR->numTuples, args->pCount, shiftBits);

  cudaMemcpyAsync(relS->id, hRelS->id, relS->numTuples * sizeof(data), cudaMemcpyHostToDevice, cudaParam->streams[0]);

  cudaDeviceSynchronize();

  probe_kernel_compressed << < cudaParam->gridSize, cudaParam->blockSize, 0, cudaParam->streams[0] >> >
      (relRn->id,
          relS->id,
          relR->numTuples,
          relS->numTuples,
          args->pCount, globalPtr[0], shiftBits, keyShift);

  cudaMemcpyAsync(args->hOut[0],
                  out[0],
                  2 * relS->numTuples * sizeof(data),
                  cudaMemcpyDeviceToHost,
                  cudaParam->streams[0]);

  cudaDeviceSynchronize();

  //ending time measurement
  cudaEventRecord(cudaParam->stop, cudaParam->streams[0]);

  //making sure all CUDA processes are completed before ending the time measurement
  cudaDeviceSynchronize();

  //measuring time
  cudaEventElapsedTime(&cudaParam->time, cudaParam->start, cudaParam->stop);

  check_cuda_error((char *) __FILE__, __LINE__);

  std::cout << "Simple Hash Join Time Probe Stage: " << cudaParam->time << " ms" << std::endl;

  //displayGPUBuffer(out[0], args->hOut[0], 100);

  return 0;
}

}
}