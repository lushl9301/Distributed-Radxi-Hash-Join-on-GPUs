#include "../../data/data.hpp"
#include "../../utils/gpu/cuda_parameters.hpp"

namespace SD {
namespace OPT {
namespace TLD {

/**histogram_build_L1(data * , data , int *)
* The function builds the histogram based on the key values of the relation. This
* histogram is later used to partition the relation using the re-order kernel.
* key           : The array containing the key values used for building the histogram.
* len           : Size of the relation for which the histogram is being built.
* hashKey       : The hash value based on which the data is partitioned.
* globalHisto   : The histogram data structure in global memory.
*/
__global__ void histogram_build_L1_tile(data *key, long len, data hashKey, int *globalHisto) {

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
    atomicAdd(&globalHisto[i * gridDim.x + blockIdx.x], sharedHisto[i]);
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
__global__ void reorder_L1_tile(data *key,
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
__global__ void reorder_L1_tile_opt(data *key,
                                    data *id,
                                    long len,
                                    data hashKey,
                                    int *globalHisto,
                                    data *keyOut,
                                    data *idOut,
                                    int *rHistoFinal,
                                    int index) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  //getting thread id and work item count
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int numWorkItems = gridDim.x * blockDim.x;

  int si = rHistoFinal[index];

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
    keyOut[si + pos] = key[i];
    idOut[si + pos] = id[i];
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
__global__ void histogram_build_L2_tile(data *key,
                                        int hashKeyL2,
                                        int hashKeyL1,
                                        int *globalHisto,
                                        int *globalHistoL1,
                                        int gridDimL1) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  int hashValue; //variable for storing the hash value of each tuple
  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage

  //getting the start and end index of the partition to be processed by the current thread block
  startIndex = globalHistoL1[blockIdx.x * gridDimL1];
  endIndex = globalHistoL1[(blockIdx.x + 1) * gridDimL1];

  //initializing all histogram entries in shared memory to 0. Otherwise there could be left over values from another thread block.
  for (int i = threadIdx.x; i < hashKeyL2; i += blockDim.x) {
    sharedHisto[i] = 0;
  }

  //barrier
  __syncthreads();

  //building the histogram in shared memory
  for (long i = threadIdx.x + startIndex; i < endIndex; i += blockDim.x) {
    hashValue = (key[i] / hashKeyL1) % hashKeyL2;
    atomicAdd(&sharedHisto[hashValue], 1);
  }

  //barrier
  __syncthreads();

  //writing the histogram back into the global memory
  for (int i = threadIdx.x; i < hashKeyL2; i += blockDim.x) {
    globalHisto[blockIdx.x * hashKeyL2 + i] = sharedHisto[i];
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
__global__ void histogram_build_L2_tile_opt(data *key,
                                            int hashKeyL2,
                                            int hashKeyL1,
                                            int *globalHisto,
                                            int *globalHistoL1,
                                            int gridDimL1) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  int hashValue; //variable for storing the hash value of each tuple
  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage

  //getting the start and end index of the partition to be processed by the current thread block
  startIndex = globalHistoL1[blockIdx.x * gridDimL1];
  endIndex = globalHistoL1[(blockIdx.x + 1) * gridDimL1];

  //initializing all histogram entries in shared memory to 0. Otherwise there could be left over values from another thread block.
  for (int i = threadIdx.x; i < hashKeyL2; i += blockDim.x) {
    sharedHisto[i] = 0;
  }

  //barrier
  __syncthreads();

  //building the histogram in shared memory
  for (long i = threadIdx.x + startIndex; i < endIndex; i += blockDim.x) {
    hashValue = (key[i] / hashKeyL1) % hashKeyL2;
    atomicAdd(&sharedHisto[hashValue], 1);
  }

  //barrier
  __syncthreads();

  //writing the histogram back into the global memory
  for (int i = threadIdx.x; i < hashKeyL2; i += blockDim.x) {
    atomicAdd(&globalHisto[blockIdx.x * hashKeyL2 + i], sharedHisto[i]);
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
__global__ void reorder_L2_tile(data *key,
                                data *id,
                                int hashKeyL2,
                                int hashKeyL1,
                                int *globalHisto,
                                int *globalHistoL1,
                                data *keyOut,
                                data *idOut,
                                int gridDimL1) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  int hashValue; //variable for storing the hash value of each tuple
  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage
  int pos; //variable for storing the destination for re-ordering

  //getting the start and end index of the partition to be processed by the current thread block
  startIndex = globalHistoL1[blockIdx.x * gridDimL1];
  endIndex = globalHistoL1[(blockIdx.x + 1) * gridDimL1];

  //reading the histogram data from global memory
  for (int i = threadIdx.x; i < hashKeyL2; i += blockDim.x) {
    sharedHisto[i] = globalHisto[blockIdx.x * hashKeyL2 + i];
  }

  //barrier
  __syncthreads();

  //re-ordering the data
  for (long i = threadIdx.x + startIndex; i < endIndex; i += blockDim.x) {
    hashValue = (key[i] / hashKeyL1) % hashKeyL2;
    pos = atomicAdd(&sharedHisto[hashValue], 1);
    keyOut[pos] = key[i];
    idOut[pos] = id[i];
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
__global__ void reorder_L2_tile_opt(data *key,
                                    data *id,
                                    int hashKeyL2,
                                    int hashKeyL1,
                                    int *globalHisto,
                                    int *globalHistoL1,
                                    data *keyOut,
                                    data *idOut,
                                    int gridDimL1) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  int hashValue; //variable for storing the hash value of each tuple
  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage
  int pos; //variable for storing the destination for re-ordering

  //getting the start and end index of the partition to be processed by the current thread block
  startIndex = globalHistoL1[blockIdx.x * gridDimL1];
  endIndex = globalHistoL1[(blockIdx.x + 1) * gridDimL1];

  //reading the histogram data from global memory
  for (int i = threadIdx.x; i < hashKeyL2; i += blockDim.x) {
    sharedHisto[i] = globalHisto[blockIdx.x * hashKeyL2 + i];
  }

  //barrier
  __syncthreads();

  //re-ordering the data
  for (long i = threadIdx.x + startIndex; i < endIndex; i += blockDim.x) {
    hashValue = (key[i] / hashKeyL1) % hashKeyL2;
    pos = atomicAdd(&sharedHisto[hashValue], 1);
    keyOut[pos] = key[i];
    idOut[pos] = id[i];
  }

  //barrier
  __syncthreads();

  //writing the histogram back into the global memory
  for (int i = threadIdx.x; i < hashKeyL2; i += blockDim.x) {
    globalHisto[blockIdx.x * hashKeyL2 + i] = sharedHisto[i];
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
__global__ void histogram_build_L2_tile_multiplied(data *key,
                                                   int hashKey,
                                                   int hashKeyL1,
                                                   int *globalHisto,
                                                   int *globalHistoL1,
                                                   int gridDimL1) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  int hashValue; //variable for storing the hash value of each tuple
  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage

  //getting the start and end index of the partition to be processed by the current thread block
  startIndex = globalHistoL1[(blockIdx.x / GRID_SIZE_MULTIPLIER) * gridDimL1];
  endIndex = globalHistoL1[(blockIdx.x / GRID_SIZE_MULTIPLIER + 1) * gridDimL1];

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
    globalHisto[((int) (blockIdx.x / GRID_SIZE_MULTIPLIER)) * hashKey * GRID_SIZE_MULTIPLIER
        + (i * GRID_SIZE_MULTIPLIER) + (blockIdx.x % GRID_SIZE_MULTIPLIER)] = sharedHisto[i];
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
__global__ void reorder_L2_tile_multiplied(data *key,
                                           data *id,
                                           int hashKey,
                                           int hashKeyL1,
                                           int *globalHisto,
                                           int *globalHistoL1,
                                           data *keyOut,
                                           data *idOut,
                                           int gridDimL1) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  int hashValue; //variable for storing the hash value of each tuple
  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage
  int pos; //variable for storing the destination for re-ordering

  //getting the start and end index of the partition to be processed by the current thread block
  startIndex = globalHistoL1[(blockIdx.x / GRID_SIZE_MULTIPLIER) * gridDimL1];
  endIndex = globalHistoL1[(blockIdx.x / GRID_SIZE_MULTIPLIER + 1) * gridDimL1];

  //reading the histogram data from global memory
  for (int i = threadIdx.x; i < hashKey; i += blockDim.x) {
    sharedHisto[i] = globalHisto[((int) (blockIdx.x / GRID_SIZE_MULTIPLIER)) * hashKey * GRID_SIZE_MULTIPLIER
        + (i * GRID_SIZE_MULTIPLIER) + (blockIdx.x % GRID_SIZE_MULTIPLIER)];
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
__global__ void probe_tile(data *rKey,
                           data *rId,
                           data *sKey,
                           data *sId,
                           int *rHisto,
                           int *sHisto,
                           int pCount,
                           int *globalPtr,
                           data *output,
                           int
                           pidStartIndex) {

  //allocating shared memory for storing each partition of relation R
  extern __shared__ data
  sharedPartitionR[];

  //pointer storing the index of each output tuple within the outputs generated by a block of threads
  __shared__ int sharedPtr;

  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage
  int posLocal; //variables for storing the destination of the output tuple
  data
  sKeyVal; //realtion S Key value for each thread
  data
  sIdVal; //realtion S Id value for each thread

  int matchedRValue; //Matched R relation values.

  for (int pid = pidStartIndex + blockIdx.x; pid < pCount; pid += gridDim.x) {

    //getting the start and end index of the relation R partition
    startIndex = rHisto[pid];
    endIndex = rHisto[(pid + 1)];

    //loading the relation R partition into shared memory
    for (int i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x) {
      sharedPartitionR[i - startIndex] = rKey[i];
      sharedPartitionR[(endIndex - startIndex) + i - startIndex] = rId[i];
    }

    sharedPtr = 0;

    //barrier
    __syncthreads();

    if (threadIdx.x < sHisto[(pid + 1)] - sHisto[pid]) {

      sKeyVal = sKey[sHisto[pid] + threadIdx.x];
      sIdVal = sId[sHisto[pid] + threadIdx.x];

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

namespace RCD {
/**histogram_build_L1(data * , data , int *)
* The function builds the histogram based on the key values of the relation. This
* histogram is later used to partition the relation using the re-order kernel.
* key           : The array containing the key values used for building the histogram.
* len           : Size of the relation for which the histogram is being built.
* hashKey       : The hash value based on which the data is partitioned.
* globalHisto   : The histogram data structure in global memory.
*/
__global__ void histogram_build_L1_tile_rcd(Record *records, long len, data hashKey, int *globalHisto) {

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
    hashValue = records[i].key % hashKey;
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
__global__ void reorder_L1_tile_opt_rcd(Record *in,
                                        long len,
                                        data hashKey,
                                        int *globalHisto,
                                        Record *out,
                                        int startIndex) {

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
    hashValue = in[i].key % hashKey;
    pos = atomicAdd(&sharedHisto[hashValue], 1); //getting the destination position
    out[startIndex + pos] = in[i];
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
__global__ void histogram_build_L2_tile_opt_rcd(Record *records,
                                                int hashKeyL2,
                                                int hashKeyL1,
                                                int *globalHisto,
                                                int *globalHistoL1,
                                                int gridDimL1) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  int hashValue; //variable for storing the hash value of each tuple
  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage

  //getting the start and end index of the partition to be processed by the current thread block
  startIndex = globalHistoL1[blockIdx.x * gridDimL1];
  endIndex = globalHistoL1[(blockIdx.x + 1) * gridDimL1];

  //initializing all histogram entries in shared memory to 0. Otherwise there could be left over values from another thread block.
  for (int i = threadIdx.x; i < hashKeyL2; i += blockDim.x) {
    sharedHisto[i] = 0;
  }

  //barrier
  __syncthreads();

  //building the histogram in shared memory
  for (long i = threadIdx.x + startIndex; i < endIndex; i += blockDim.x) {
    hashValue = (records[i].key / hashKeyL1) % hashKeyL2;
    atomicAdd(&sharedHisto[hashValue], 1);
  }

  //barrier
  __syncthreads();

  //writing the histogram back into the global memory
  for (int i = threadIdx.x; i < hashKeyL2; i += blockDim.x) {
    atomicAdd(&globalHisto[blockIdx.x * hashKeyL2 + i], sharedHisto[i]);
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
__global__ void reorder_L2_tile_opt_rcd(Record *in,
                                        int hashKeyL2,
                                        int hashKeyL1,
                                        int *globalHisto,
                                        int *globalHistoL1,
                                        Record *out,
                                        int gridDimL1) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  int hashValue; //variable for storing the hash value of each tuple
  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage
  int pos; //variable for storing the destination for re-ordering

  //getting the start and end index of the partition to be processed by the current thread block
  startIndex = globalHistoL1[blockIdx.x * gridDimL1];
  endIndex = globalHistoL1[(blockIdx.x + 1) * gridDimL1];

  //reading the histogram data from global memory
  for (int i = threadIdx.x; i < hashKeyL2; i += blockDim.x) {
    sharedHisto[i] = globalHisto[blockIdx.x * hashKeyL2 + i];
  }

  //barrier
  __syncthreads();

  //re-ordering the data
  for (long i = threadIdx.x + startIndex; i < endIndex; i += blockDim.x) {
    hashValue = (in[i].key / hashKeyL1) % hashKeyL2;
    pos = atomicAdd(&sharedHisto[hashValue], 1);
    out[pos] = in[i];
  }

  //barrier
  __syncthreads();

  //writing the histogram back into the global memory
  for (int i = threadIdx.x; i < hashKeyL2; i += blockDim.x) {
    globalHisto[blockIdx.x * hashKeyL2 + i] = sharedHisto[i];
  }

}

}
}
}
}