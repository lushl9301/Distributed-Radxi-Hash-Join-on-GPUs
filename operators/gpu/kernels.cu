#include "../../data/data.hpp"
#include "../../utils/gpu/cuda_parameters.hpp"
namespace SD {

__global__ void memTest(data *in, data *out) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid + 1] = in[tid + 1];
}

/**histogram_build_L1(data * , data , int *)
* The function builds the histogram based on the key values of the relation. This
* histogram is later used to partition the relation using the re-order kernel.
* key           : The array containing the key values used for building the histogram.
* len           : Size of the relation for which the histogram is being built.
* hashKey       : The hash value based on which the data is partitioned.
* globalHisto   : The histogram data structure in global memory.
*/
__global__ void histogram_build_L1(data *key, long len, data hashKey, int *globalHisto) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  //getting thread id and work item count
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int numWorkItems = gridDim.x * blockDim.x;

  int hashValue; //variable for storing the hash value of each tuple

  //initializing all histogram entries in shared memory to 0. Otherwise there could be left over values from another thread block.
  for (int i = threadIdx.x; i < hashKey; i += blockDim.x) {
    sharedHisto[i] = 0;
  }

  //barrier
  __syncthreads();

  //building the histogram in shared memory
  for (long i = tid; i < len; i += numWorkItems) {
    hashValue = key[i] % hashKey;
    atomicAdd(&sharedHisto[hashValue], 1);
  }

  //barrier
  __syncthreads();

  //writing the histogram back into the global memory
  for (int i = threadIdx.x; i < hashKey; i += blockDim.x) {
    globalHisto[i * gridDim.x + blockIdx.x] = sharedHisto[i];
  }

}

/**reorder_L1(data *, data *, long , data , int *, data *, data *)
* This function re-orders a relation based on the histogram built using 'histogram_build_L1' kernel.
* key           : The key values of the relation to be re-ordered.
* id            : The id values of the relation to be re-ordered.
* len           : Size of the relation which is being re-ordered.
* hashKey       : The hash value based on which the data is re-ordered.
* globalHisto   : The histogram data structure in the global memory.
* keyOut        : The array storing the re-ordered key values.
* idOut         : The array storing the re-ordered id values.
*/
__global__ void reorder_L1(data *key, data *id, long len, data hashKey, int *globalHisto, data *keyOut, data *idOut) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  //getting thread id and work item count
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int numWorkItems = gridDim.x * blockDim.x;

  int hashValue; //variable for storing the hash value of each tuple
  int pos; //variable for storing the destination for re-ordering

  //reading the histogram data from global memory
  for (int i = threadIdx.x; i < hashKey; i += blockDim.x) {
    sharedHisto[i] = globalHisto[i * gridDim.x + blockIdx.x];
  }

  //barrier
  __syncthreads();

  //re-ordering the data
  for (long i = tid; i < len; i += numWorkItems) {
    hashValue = key[i] % hashKey;
    pos = atomicAdd(&sharedHisto[hashValue], 1); //getting the destination position
    keyOut[pos] = key[i];
    idOut[pos] = id[i];
  }
}

/**histogram_build_L2(data *, data *, long , int , int *, int *)
* This function builds the histogram for second level of partitioning.
* key           : The array containing the key values which has already been partitioned by reorder_L1.
* hashKey       : The hash value based on which the data will be partitioned in the second pass.
* hashKeyL1     : The hash value based on which the data was partitioned in the first pass.
* globalHisto   : The histogram data structure for the second partitioning pass, located in the global memory.
* globalHistoL1 : The histogram data structure generated during level 1 partitioning, also located in global memory.
*/
__global__ void histogram_build_L2(data *key, int hashKey, int hashKeyL1, int *globalHisto, int *globalHistoL1) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  int hashValue; //variable for storing the hash value of each tuple
  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage

  //getting the start and end index of the partition to be processed by the current thread block
  startIndex = globalHistoL1[blockIdx.x * gridDim.x];
  endIndex = globalHistoL1[(blockIdx.x + 1) * gridDim.x];

  //initializing all histogram entries in shared memory to 0. Otherwise there could be left over values from another thread block.
  for (int i = threadIdx.x; i < hashKey; i += blockDim.x) {
    sharedHisto[i] = 0;
  }

  //barrier
  __syncthreads();

  //building the histogram in shared memory
  for (long i = threadIdx.x + startIndex; i < endIndex; i += blockDim.x) {
    hashValue = (key[i] / hashKeyL1) % hashKey;
    atomicAdd(&sharedHisto[hashValue], 1);
  }

  //barrier
  __syncthreads();

  //writing the histogram back into the global memory
  for (int i = threadIdx.x; i < hashKey; i += blockDim.x) {
    globalHisto[blockIdx.x * hashKey + i] = sharedHisto[i];
  }
}

/**reorder_L2(data *, data *, data , data , int *, int *, data *, data *)
* This function re-orders a relation based on the histogram built using 'histogram_build_L2' kernel.
* key           : The array containing the key values which has already been partitioned by reorder_L1.
* id            : The array containing the id values which has already been partitioned by reorder_L1.
* hashKey       : The hash value based on which the data will be partitioned in the second pass.
* hashKeyL1     : The hash value based on which the data was partitioned in the first pass.
* globalHisto   : The histogram data structure for the second partitioning pass, located in the global memory.
* globalHistoL1 : The histogram data structure generated during level 1 partitioning, also located in global memory.
* keyOut        : The array storing the re-ordered key values.
* idOut         : The array storing the re-ordered id values.
*/
__global__ void reorder_L2(data *key,
                           data *id,
                           int hashKey,
                           int hashKeyL1,
                           int *globalHisto,
                           int *globalHistoL1,
                           data *keyOut,
                           data *idOut) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  int hashValue; //variable for storing the hash value of each tuple
  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage
  int pos; //variable for storing the destination for re-ordering

  //getting the start and end index of the partition to be processed by the current thread block
  startIndex = globalHistoL1[blockIdx.x * gridDim.x];
  endIndex = globalHistoL1[(blockIdx.x + 1) * gridDim.x];

  //reading the histogram data from global memory
  for (int i = threadIdx.x; i < hashKey; i += blockDim.x) {
    sharedHisto[i] = globalHisto[blockIdx.x * hashKey + i];
  }

  //barrier
  __syncthreads();

  //re-ordering the data
  for (long i = threadIdx.x + startIndex; i < endIndex; i += blockDim.x) {
    hashValue = (key[i] / hashKeyL1) % hashKey;
    pos = atomicAdd(&sharedHisto[hashValue], 1);
    keyOut[pos] = key[i];
    idOut[pos] = id[i];
  }

}

/**probe(data *, data *, data *, data *, int *, int *, data *)
* The function performs the actual join operation by joining partitions with the same hash value.
* rKey      : The array containing the key values of partitioned relation R.
* rID       : The array containing the id values of partitioned relation R.
* sKey      : The array containing the key values of partitioned relation S.
* sID       : The array containing id key values of partitioned relation S.
* rHisto    : The histogram of relation R.
* sHisto    : The histogram of relation S.
* pCount    : The total number of partitions of each realtion.
* globalPtr : The global pointer that is used to get the index of the output tuple.
* output    : The array used for storing the output of the probe operation.
*/
__global__ void probe(data *rKey,
                      data *rId,
                      data *sKey,
                      data *sId,
                      int *rHisto,
                      int *sHisto,
                      int pCount,
                      int *globalPtr,
                      data *output) {

  //allocating shared memory for storing each partition of relation R
  extern __shared__ data
  sharedPartitionR[];

  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage
  int pos; //variable for storing the destination of the output tuple

  for (int pid = blockIdx.x; pid < pCount; pid += gridDim.x) {

    //getting the start and end index of the relation R partition
    startIndex = rHisto[pid];
    endIndex = rHisto[pid + 1];

    //loading the relation R partition into shared memory
    for (int i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x) {
      sharedPartitionR[i - startIndex] = rKey[i];
      sharedPartitionR[(endIndex - startIndex) + i - startIndex] = rId[i];
    }

    //barrier
    __syncthreads();

    //probing the R partition using the S partition
    for (int i = sHisto[pid] + threadIdx.x; i < sHisto[pid + 1]; i += blockDim.x) {

      for (int j = 0; j < endIndex - startIndex; j++) {

        if (sKey[i] == sharedPartitionR[j]) {
          pos = atomicAdd(globalPtr, 1);
          output[2 * pos] = sharedPartitionR[(endIndex - startIndex) + j];
          output[2 * pos + 1] = sId[i];
        }
      }
    }
  }

}

/**histogram_build_global(data * , data , int *)
* The function builds the histogram based on the key values of the relation using GPU global
* memory. This histogram is later used to partition the relation using the re-order kernel.
* key           : The array containing the key values used for building the histogram.
* len           : Size of the relation for which the histogram is being built.
* hashKey       : The hash value based on which the data is partitioned.
* histo   : The histogram data structure in global memory.
*/
__global__ void histogram_build_global(data *key, long len, data hashKey, int *histo) {

  //getting thread id and work item count
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int numWorkItems = gridDim.x * blockDim.x;

  int hashValue; //variable for storing the hash value of each tuple

  //building the histogram in global memory
  for (long i = tid; i < len; i += numWorkItems) {
    hashValue = key[i] % hashKey;
    atomicAdd(&histo[hashValue], 1);
  }

}

/**reorder_global(data *, data *, long , data , int *, data *, data *)
* This function re-orders a relation based on the histogram built using 'histogram_build_global' kernel.
* key           : The key values of the relation to be re-ordered.
* id            : The id values of the relation to be re-ordered.
* len           : Size of the relation which is being re-ordered.
* hashKey       : The hash value based on which the data is re-ordered.
* histo   : The histogram data structure in the global memory.
* keyOut        : The array storing the re-ordered key values.
* idOut         : The array storing the re-ordered id values.
*/
__global__ void reorder_global(data *key, data *id, long len, data hashKey, int *histo, data *keyOut, data *idOut) {

  //getting thread id and work item count
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int numWorkItems = gridDim.x * blockDim.x;

  int hashValue; //variable for storing the hash value of each tuple
  int pos; //variable for storing the destination for re-ordering

  //re-ordering the data
  for (long i = tid; i < len; i += numWorkItems) {
    hashValue = key[i] % hashKey;
    pos = atomicAdd(&histo[hashValue], 1); //getting the destination position
    keyOut[pos] = key[i];
    idOut[pos] = id[i];
  }
}

/**probe_match_rate(data *, data *, data *, data *, int *, int *, data *)
* The function performs the probe operation for data sets with high match rate. In this kernel the
* block size should be greater than or equal to the size of the partition with maximum number of tuples.
* rKey      : The array containing the key values of partitioned relation R.
* rID       : The array containing the id values of partitioned relation R.
* sKey      : The array containing the key values of partitioned relation S.
* sID       : The array containing id key values of partitioned relation S.
* rHisto    : The histogram of relation R.
* sHisto    : The histogram of relation S.
* pCount    : The total number of partitions of each realtion.
* globalPtr : The global pointer that is used to get the index of the output tuple.
* output    : The array used for storing the output of the probe operation.
* pFlag     : A flag used to denote wheter a given partition has completed the probe operation.
*/
__global__ void probe_match_rate(data *rKey,
                                 data *rId,
                                 data *sKey,
                                 data *sId,
                                 int *rHisto,
                                 int *sHisto,
                                 int pCount,
                                 int *globalPtr,
                                 data *output,
                                 int *pFlag) {

  //allocating shared memory for storing each partition of relation R
  extern __shared__ data
  sharedPartitionR[];

  //pointer storing the index of each output tuple within the outputs generated by a block of threads
  __shared__ int sharedPtr;

  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage
  int pos, posLocal; //variables for storing the destination of the output tuple
  int mCount = 0; //Number of matches for each S relation Tuple. mCount starts at 1 for the R table tuple
  int pSize; //Size of the aprtition being processed
  int sKeyVal, sIdVal; //sKey and sId value for each thread;

  int matches[MAX_MATCH_RATE]; //Matched R relation values.

  for (int pid = blockIdx.x; pid < pCount; pid += gridDim.x) {

    //checking if this partition has already been processed.
    if (pFlag[pid] != -1) {

      //setting mCount as 0 for the current partition
      mCount = 0;

      //setting the shared pointer value to 0
      sharedPtr = 0;

      //getting the start and end index of the relation R partition
      startIndex = rHisto[pid];
      endIndex = rHisto[pid + 1];
      pSize = endIndex - startIndex;

      //loading the relation R partition into shared memory
      for (int i = threadIdx.x; i < pSize; i += blockDim.x) {
        sharedPartitionR[i] = rKey[i + startIndex];
        sharedPartitionR[pSize + i] = rId[i + startIndex];
      }

      //barrier
      __syncthreads();

      if (threadIdx.x < sHisto[pid + 1] - sHisto[pid]) {

        //getting the sKey value for the thread
        sKeyVal = sKey[sHisto[pid] + threadIdx.x];
        sIdVal = sId[sHisto[pid] + threadIdx.x];

        //probing the R partition using the S partition. Each element in S partition is processed by a single thread in the block. So max block size is 1024.
        for (int j = 0; j < endIndex - startIndex; j++) {

          if (sKeyVal == sharedPartitionR[j]) {
            matches[mCount / 2] = sharedPartitionR[pSize + j];
            mCount += 2;
          }
        }

        //calculating the number of macthes in the current partition
        posLocal = atomicAdd(&sharedPtr, mCount);

        //barrier
        __syncthreads();

        if (threadIdx.x == 0) {

          //checking if there is space in the global buffer for writing the output
          sharedPtr = atomicAdd(globalPtr, sharedPtr);
          if (pos < MAX_OUTPUT_SIZE) {
            pFlag[pid] = -1;
          }
        }

        //barrier
        __syncthreads();

        //writing the output to the global buffer if there is space available in the buffer
        if (sharedPtr < MAX_OUTPUT_SIZE) {

          for (int i = 0; i < mCount; i += 2) {
            output[sharedPtr + posLocal + i] = sIdVal;
            output[sharedPtr + posLocal + i + 1] = matches[i];
          }
        }
      }
    }
  }

}

/**probe_count(data *, data *, data *, data *, int *, int *, data *)
* The function simply counts the number of probe outputs. This makes it possible to
* handle cases with high match rate in the traditional way.
* rKey      : The array containing the key values of partitioned relation R.
* sKey      : The array containing the key values of partitioned relation S.
* rHisto    : The histogram of relation R.
* sHisto    : The histogram of relation S.
* pCount    : The total number of partitions of each realtion.
* output    : The array used for storing the number of matches in each partition.
*/
__global__ void probe_count(data *rKey, data *sKey, int *rHisto, int *sHisto, int pCount, data *output) {

  //allocating shared memory for storing each partition of relation R
  extern __shared__ data
  sharedPartitionR[];

  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage

  for (int pid = blockIdx.x; pid < pCount; pid += gridDim.x) {

    //getting the start and end index of the relation R partition
    startIndex = rHisto[pid];
    endIndex = rHisto[pid + 1];

    __shared__ int sharedPtr;

    //loading the relation R partition into shared memory
    for (int i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x) {
      sharedPartitionR[i - startIndex] = rKey[i];
    }

    //barrier
    __syncthreads();

    //probing the R partition using the S partition
    for (int i = sHisto[pid] + threadIdx.x; i < sHisto[pid + 1]; i += blockDim.x) {

      for (int j = 0; j < endIndex - startIndex; j++) {

        if (sKey[i] == sharedPartitionR[j]) {
          atomicAdd(&sharedPtr, 1);
        }
      }
    }

    //writing the number of matches identified in the current partition to global memory.
    output[pid] = sharedPtr;
  }

}

/**histogram_build_L1(data * , data , int *)
* The function builds the histogram based on the 2008 implementation. This
* histogram is later used to partition the relation using the re-order kernel.
* key           : The array containing the key values used for building the histogram.
* len           : Size of the relation for which the histogram is being built.
* hashKey       : The hash value based on which the data is partitioned.
* globalHisto   : The histogram data structure in global memory.
*/
__global__ void histogram_build_L1_2008(data *key, long len, data hashKey, int *globalHisto) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  //getting thread id and work item count
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int numWorkItems = gridDim.x * blockDim.x;

  int hashValue; //variable for storing the hash value of each tuple

  //initializing all histogram entries in shared memory to 0. Otherwise there could be left over values from another thread block.
  for (int i = threadIdx.x; i < hashKey * blockDim.x; i += blockDim.x) {
    sharedHisto[i] = 0;
  }

  //barrier
  __syncthreads();

  //building the histogram in shared memory
  for (long i = tid; i < len; i += numWorkItems) {
    hashValue = key[i] % hashKey;
    sharedHisto[hashValue * blockDim.x + threadIdx.x] += 1;
  }

  //barrier
  __syncthreads();

  //writing the histogram back into the global memory
  for (int i = threadIdx.x; i < hashKey * blockDim.x; i += blockDim.x) {
    globalHisto[(i / blockDim.x) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + (i % blockDim.x)] =
        sharedHisto[i];
  }

}

/**reorder_L1(data *, data *, long , data , int *, data *, data *)
* This function re-orders a relation based on the 2008 implementation.
* key           : The key values of the relation to be re-ordered.
* id            : The id values of the relation to be re-ordered.
* len           : Size of the relation which is being re-ordered.
* hashKey       : The hash value based on which the data is re-ordered.
* globalHisto   : The histogram data structure in the global memory.
* keyOut        : The array storing the re-ordered key values.
* idOut         : The array storing the re-ordered id values.
*/
__global__ void reorder_L1_2008(data *key,
                                data *id,
                                long len,
                                data hashKey,
                                int *globalHisto,
                                data *keyOut,
                                data *idOut) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  //getting thread id and work item count
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int numWorkItems = gridDim.x * blockDim.x;

  int hashValue; //variable for storing the hash value of each tuple
  int pos; //variable for storing the destination for re-ordering

  //reading the histogram data from global memory
  for (int i = threadIdx.x; i < hashKey * blockDim.x; i += blockDim.x) {
    sharedHisto[i] =
        globalHisto[(i / blockDim.x) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + (i % blockDim.x)];
  }

  //barrier
  __syncthreads();

  //re-ordering the data
  for (long i = tid; i < len; i += numWorkItems) {
    hashValue = key[i] % hashKey;
    pos = sharedHisto[hashValue * blockDim.x + threadIdx.x]; //getting the destination position
    sharedHisto[hashValue * blockDim.x + threadIdx.x]++;
    keyOut[pos] = key[i];
    idOut[pos] = id[i];
  }
}
}

//namspace containing all kernels related to small data sets
namespace MD {

}

//namspace containing all kernels related to small data sets
namespace LD {

/**histogram_build_L1(data * , data , int *)
* The function builds the histogram based on the key values of the relation. This
* histogram is later used to partition the relation using the re-order kernel.
* key           : The array containing the key values used for building the histogram.
* len           : Size of the relation for which the histogram is being built.
* hashKey       : The hash value based on which the data is partitioned.
* globalHisto   : The histogram data structure in global memory.
*/
__global__ void histogram_build_L1(data *key, long len, data hashKey, int *globalHisto) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  //getting thread id and work item count
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int numWorkItems = gridDim.x * blockDim.x;

  int hashValue; //variable for storing the hash value of each tuple

  //initializing all histogram entries in shared memory to 0. Otherwise there could be left over values from another thread block.
  for (int i = threadIdx.x; i < hashKey; i += blockDim.x) {
    sharedHisto[i] = 0;
  }

  //barrier
  __syncthreads();

  //building the histogram in shared memory
  for (long i = tid; i < len; i += numWorkItems) {
    hashValue = key[i] % hashKey;
    atomicAdd(&sharedHisto[hashValue], 1);
  }

  //barrier
  __syncthreads();

  //writing the histogram back into the global memory
  for (int i = threadIdx.x; i < hashKey; i += blockDim.x) {
    //atomicAdd(&globalHisto[i * gridDim.x + blockIdx.x], sharedHisto[i]);
    globalHisto[i * gridDim.x + blockIdx.x] = sharedHisto[i];
  }

}

/**reorder_L1(data *, data *, long , data , int *, data *, data *)
* This function re-orders a relation based on the histogram built using 'histogram_build_L1' kernel.
* key           : The key values of the relation to be re-ordered.
* id            : The id values of the relation to be re-ordered.
* len           : Size of the relation which is being re-ordered.
* hashKey       : The hash value based on which the data is re-ordered.
* globalHisto   : The histogram data structure in the global memory.
* keyOut        : The array storing the re-ordered key values.
* idOut         : The array storing the re-ordered id values.
*/
__global__ void reorder_L1(data *key, data *id, long len, data hashKey, int *globalHisto, data *keyOut, data *idOut) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  //getting thread id and work item count
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int numWorkItems = gridDim.x * blockDim.x;

  int hashValue; //variable for storing the hash value of each tuple
  int pos; //variable for storing the destination for re-ordering

  //reading the histogram data from global memory
  for (int i = threadIdx.x; i < hashKey; i += blockDim.x) {
    sharedHisto[i] = globalHisto[i * gridDim.x + blockIdx.x];
  }

  //barrier
  __syncthreads();

  //re-ordering the data
  for (long i = tid; i < len; i += numWorkItems) {
    hashValue = key[i] % hashKey;
    pos = atomicAdd(&sharedHisto[hashValue], 1); //getting the destination position
    keyOut[pos] = key[i];
    idOut[pos] = id[i];
  }
}

/**histogram_build_L2(data *, data *, long , int , int *, int *)
* This function builds the histogram for second level of partitioning.
* key           : The array containing the key values which has already been partitioned by reorder_L1.
* hashKey       : The hash value based on which the data will be partitioned in the second pass.
* hashKeyL1     : The hash value based on which the data was partitioned in the first pass.
* globalHisto   : The histogram data structure for the second partitioning pass, located in the global memory.
* globalHistoL1 : The histogram data structure generated during level 1 partitioning, also located in global memory.
* iterCount     : Iteration count of the current kernel call.
*/
__global__ void histogram_build_L2(data *key,
                                   int hashKey,
                                   int hashKeyL1,
                                   int *globalHisto,
                                   int *globalHistoL1,
                                   int gridDimL1,
                                   int iterCount,
                                   int maxIterCount) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  int hashValue; //variable for storing the hash value of each tuple
  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage

  int startPosCorrection = globalHistoL1[0];

  //getting the start and end index of the partition to be processed by the current thread block
  startIndex = globalHistoL1[(blockIdx.x / GRID_SIZE_MULTIPLIER) * gridDimL1] - startPosCorrection;
  endIndex = globalHistoL1[(blockIdx.x / GRID_SIZE_MULTIPLIER + 1) * gridDimL1] - startPosCorrection;

  //initializing all histogram entries in shared memory to 0. Otherwise there could be left over values from another thread block.
  for (int i = threadIdx.x; i < hashKey; i += blockDim.x) {
    sharedHisto[i] = 0;
  }

  //barrier
  __syncthreads();

  //building the histogram in shared memory
  for (long i = (blockIdx.x % GRID_SIZE_MULTIPLIER) * blockDim.x + threadIdx.x + startIndex; i < endIndex;
       i += GRID_SIZE_MULTIPLIER * blockDim.x) {
    hashValue = (key[i] / hashKeyL1) % hashKey;
    atomicAdd(&sharedHisto[hashValue], 1);
  }

  //barrier
  __syncthreads();

  //writing the histogram back into the global memory
  for (int i = threadIdx.x; i < hashKey; i += blockDim.x) {
    globalHisto[maxIterCount
        * (((int) (blockIdx.x / GRID_SIZE_MULTIPLIER)) * hashKey * GRID_SIZE_MULTIPLIER + (i * GRID_SIZE_MULTIPLIER)
            + (blockIdx.x % GRID_SIZE_MULTIPLIER)) + iterCount] = sharedHisto[i];
  }
}

/**reorder_L2(data *, data *, data , data , int *, int *, data *, data *)
* This function re-orders a relation based on the histogram built using 'histogram_build_L2' kernel.
* key           : The array containing the key values which has already been partitioned by reorder_L1.
* id            : The array containing the id values which has already been partitioned by reorder_L1.
* hashKey       : The hash value based on which the data will be partitioned in the second pass.
* hashKeyL1     : The hash value based on which the data was partitioned in the first pass.
* globalHisto   : The histogram data structure for the second partitioning pass, located in the global memory.
* globalHistoL1 : The histogram data structure generated during level 1 partitioning, also located in global memory.
* keyOut        : The array storing the re-ordered key values.
* idOut         : The array storing the re-ordered id values.
* iterCount     : Iteration count of the current kernel call.
*/
__global__ void reorder_L2(data *key,
                           data *id,
                           int hashKey,
                           int hashKeyL1,
                           int *globalHisto,
                           int *globalHistoL1,
                           data *keyOut,
                           data *idOut,
                           int gridDimL1,
                           int iterCount,
                           int maxIterCount) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  int hashValue; //variable for storing the hash value of each tuple
  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage
  int pos; //variable for storing the destination for re-ordering

  int startPosCorrection = globalHistoL1[0];

  //getting the start and end index of the partition to be processed by the current thread block
  startIndex = globalHistoL1[(blockIdx.x / GRID_SIZE_MULTIPLIER) * gridDimL1] - startPosCorrection;
  endIndex = globalHistoL1[(blockIdx.x / GRID_SIZE_MULTIPLIER + 1) * gridDimL1] - startPosCorrection;

  //reading the histogram data from global memory
  for (int i = threadIdx.x; i < hashKey; i += blockDim.x) {
    sharedHisto[i] = globalHisto[maxIterCount
        * (((int) (blockIdx.x / GRID_SIZE_MULTIPLIER)) * hashKey * GRID_SIZE_MULTIPLIER + (i * GRID_SIZE_MULTIPLIER)
            + (blockIdx.x % GRID_SIZE_MULTIPLIER)) + iterCount];
  }

  //barrier
  __syncthreads();

  //re-ordering the data
  for (long i = (blockIdx.x % GRID_SIZE_MULTIPLIER) * blockDim.x + threadIdx.x + startIndex; i < endIndex;
       i += GRID_SIZE_MULTIPLIER * blockDim.x) {
    hashValue = (key[i] / hashKeyL1) % hashKey;
    pos = atomicAdd(&sharedHisto[hashValue], 1);
    keyOut[pos] = key[i];
    idOut[pos] = id[i];
  }

}

/**probe(data *, data *, data *, data *, int *, int *, data *)
* The function performs the actual join operation by joining partitions with the same hash value.
* The number of threads in a block must be greater than or equalt to the number of tuples in the
* largest partition. We also assume a 100% match rate for this kernel.
* rKey      : The array containing the key values of partitioned relation R.
* rID       : The array containing the id values of partitioned relation R.
* sKey      : The array containing the key values of partitioned relation S.
* sID       : The array containing id key values of partitioned relation S.
* rHisto    : The histogram of relation R.
* sHisto    : The histogram of relation S.
* pCount    : The total number of partitions of each realtion.
* globalPtr : The global pointer that is used to get the index of the output tuple.
* output    : The array used for storing the output of the probe operation.
*/
__global__ void probe(data *rKey,
                      data *rId,
                      data *sKey,
                      data *sId,
                      int *rHisto,
                      int *sHisto,
                      int pCount,
                      int *globalPtr,
                      data *output,
                      int pidStartIndex,
                      int iterCount) {

  //allocating shared memory for storing each partition of relation R
  extern __shared__ data
  sharedPartitionR[];

  //pointer storing the index of each output tuple within the outputs generated by a block of threads
  __shared__ int sharedPtr;

  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage
  int posLocal = 0; //variables for storing the destination of the output tuple
  int sKeyVal; //realtion S Key value for each thread
  int sIdVal; //realtion S Id value for each thread

  int matchedRValue; //Matched R relation values.

  for (int pid = pidStartIndex + blockIdx.x; pid < pCount; pid += gridDim.x) {

    //getting the start and end index of the relation R partition
    startIndex = rHisto[iterCount * pid * GRID_SIZE_MULTIPLIER];
    endIndex = rHisto[iterCount * (pid + 1) * GRID_SIZE_MULTIPLIER];

    //loading the relation R partition into shared memory

    for (int i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x) {
      sharedPartitionR[i - startIndex] = rKey[i];
      sharedPartitionR[(endIndex - startIndex) + i - startIndex] = rId[i];
    }

    sharedPtr = 0;

    //barrier
    __syncthreads();

    if (threadIdx.x
        < sHisto[iterCount * (pid + 1) * GRID_SIZE_MULTIPLIER] - sHisto[iterCount * pid * GRID_SIZE_MULTIPLIER]) {

      sKeyVal = sKey[sHisto[iterCount * pid * GRID_SIZE_MULTIPLIER] + threadIdx.x];
      sIdVal = sId[sHisto[iterCount * pid * GRID_SIZE_MULTIPLIER] + threadIdx.x];

      //probing the R partition using the S partition
      for (int j = 0; j < endIndex - startIndex; j++) {

        if (sKeyVal == sharedPartitionR[j]) {
          posLocal = atomicAdd(&sharedPtr, 2);
          matchedRValue = sharedPartitionR[endIndex - startIndex + j];
        }
      }

      //barrier
      __syncthreads();

      if (threadIdx.x == 0) {

        //checking if there is space in the global buffer for writing the output
        sharedPtr = atomicAdd(globalPtr, sharedPtr);
      }

      //barrier
      __syncthreads();

      output[sharedPtr + posLocal] = sIdVal;
      output[sharedPtr + posLocal + 1] = matchedRValue;
    }

  }

}

}