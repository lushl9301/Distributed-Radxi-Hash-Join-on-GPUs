#include "../../data/data.hpp"
#include "../../utils/gpu/cuda_parameters.hpp"

namespace SD {

//namepsace containing the optimized functions
namespace OPT {

/**histogram_build_L2(data *, data *, long , int , int *, int *)
* This function builds the histogram for second level of partitioning.
* key           : The array containing the key values which has already been partitioned by reorder_L1.
* hashKey       : The hash value based on which the data will be partitioned in the second pass.
* hashKeyL1     : The hash value based on which the data was partitioned in the first pass.
* globalHisto   : The histogram data structure for the second partitioning pass, located in the global memory.
* globalHistoL1 : The histogram data structure generated during level 1 partitioning, also located in global memory.
*/
__global__ void histogram_build_L2(data *key,
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
__global__ void reorder_L2(data *key,
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
__global__ void probe(data *rKey,
                      data *rId,
                      data *sKey,
                      data *sId,
                      int *rHisto,
                      int *sHisto,
                      int pCount,
                      int *globalPtr,
                      data *output,
                      int pidStartIndex) {

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
    startIndex = rHisto[pid * GRID_SIZE_MULTIPLIER];
    endIndex = rHisto[(pid + 1) * GRID_SIZE_MULTIPLIER];

    //loading the relation R partition into shared memory
    for (int i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x) {
      sharedPartitionR[i - startIndex] = rKey[i];
      sharedPartitionR[(endIndex - startIndex) + i - startIndex] = rId[i];
    }

    sharedPtr = 0;

    //barrier
    __syncthreads();

    if (threadIdx.x < sHisto[(pid + 1) * GRID_SIZE_MULTIPLIER] - sHisto[pid * GRID_SIZE_MULTIPLIER]) {

      sKeyVal = sKey[sHisto[pid * GRID_SIZE_MULTIPLIER] + threadIdx.x];
      sIdVal = sId[sHisto[pid * GRID_SIZE_MULTIPLIER] + threadIdx.x];

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

/**probe_global_memory(data *, data *, data *, data *, int *, int *, data *)
* Optimized probe for global memory join.
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
__global__ void probe_global_memory(data *rKey,
                                    data *rId,
                                    data *sKey,
                                    data *sId,
                                    int *rHisto,
                                    int *sHisto,
                                    int pCount,
                                    int *globalPtr,
                                    data *output,
                                    int pidStartIndex) {

  //allocating shared memory for storing each partition of relation R
  extern __shared__ data
  sharedPartitionR[];

  //pointer storing the index of each output tuple within the outputs generated by a block of threads
  __shared__ int sharedPtr;

  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage
  int posLocal; //variables for storing the destination of the output tuple
  int sKeyVal; //realtion S Key value for each thread
  int sIdVal; //realtion S Id value for each thread

  int matchedRValue; //Matched R relation values.

  for (int pid = pidStartIndex + blockIdx.x; pid < pCount; pid += gridDim.x) {

    //getting the start and end index of the relation R partition
    startIndex = rHisto[pid];
    endIndex = rHisto[pid + 1];

    //loading the relation R partition into shared memory
    for (int i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x) {
      sharedPartitionR[i - startIndex] = rKey[i];
      sharedPartitionR[(endIndex - startIndex) + i - startIndex] = rId[i];
    }

    sharedPtr = 0;

    //barrier
    __syncthreads();

    if (threadIdx.x < sHisto[pid + 1] - sHisto[pid]) {

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

/**skew_detect(int * , int *, int , int ).
* Kernel to detect skew in S relation partitions. It generates an array that
* specifies the number of blocks required to process a skewed partition.
* histo            : Histogram for relation S.
* pCount           : Partition count for each relation.
* skewThreshold    : The threshold value after which the partition requires additional blocks.
* skewFlag         : The skewFlag array which denotes the number of blocks required to process a skewed partition.
*/
__global__ void skew_detect(int *histo, int pCount, int skewThreshold, int *skewFlag) {

  //getting thread id and work item count
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int numWorkItems = gridDim.x * blockDim.x;

  for (long i = tid; i < pCount; i += numWorkItems) {
    skewFlag[i] =
        ((int) ((histo[(i + 1) * GRID_SIZE_MULTIPLIER] - histo[i * GRID_SIZE_MULTIPLIER]) / skewThreshold)) + 1;
  }
}

/**generate_extended_histogram(int * , int *, int , int ).
* This kernel generates a mapping between blocIds and the partition ID the corresponding
* blockId needs to process. We assume that only relation S has skew.
* pCount            : Partition count for each relation.
* skewFlag          : The skewFlag array generated by skew_detect kernel.
* mapping           : The mapping between blockId and partition ID.
* mappingSecondary  : The secondary mapping array to find idenxes within a partition.
*/
__global__ void generate_block_mapping(int pCount, int *skewFlag, int *mapping, int *mappingSecondary) {

  //getting thread id and work item count
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int numWorkItems = gridDim.x * blockDim.x;

  //the index variable for the mapping array
  int mapIndex;

  //the count variable for secondary mapping array.
  int count;

  for (long i = tid; i < pCount; i += numWorkItems) {
    mapIndex = skewFlag[i];
    count = 0;
    while (mapIndex < skewFlag[i + 1]) {
      mapping[mapIndex] = i;
      mappingSecondary[mapIndex] = count;
      mapIndex++;
      count++;
    }

  }
}

/**probe_skew(data *, data *, data *, data *, int *, int *, data *)
* Probe kernel for skewed data sets. We assume a 100% match rate for the kernel.
* rKey              : The array containing the key values of partitioned relation R.
* rID               : The array containing the id values of partitioned relation R.
* sKey              : The array containing the key values of partitioned relation S.
* sID               : The array containing id key values of partitioned relation S.
* rHisto            : The histogram of relation R.
* sHisto            : The histogram of relation S.
* pCount            : The total number of partitions of each realtion.
* globalPtr         : The global pointer that is used to get the index of the output tuple.
* output            : The array used for storing the output of the probe operation.
* skewFlagStartIndex: The index into the skew flag array.
* mapping           : The mapping between blockId and partition ID.
* mappingSecondary  : The secondary mapping array to find idenxes within a partition.
* skewFlag          : The skewFlag array generated by skew_detect kernel.
* skewFlagLen       : Length of the skewFlag array
* skewThreshold     : Threshold for detecting data skew.
*/
__global__ void probe_skew(data *rKey,
                           data *rId,
                           data *sKey,
                           data *sId,
                           int *rHisto,
                           int *sHisto,
                           int *globalPtr,
                           data *output,
                           int skewFlagStartIndex,
                           int *mapping,
                           int *mappingSecondary,
                           int *skewFlag,
                           int skewFlagLen,
                           int skewThreshold) {

  //allocating shared memory for storing each partition of relation R
  extern __shared__ data
  sharedPartitionR[];

  //pointer storing the index of each output tuple within the outputs generated by a block of threads
  __shared__ int sharedPtr;

  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage
  long sStartIndex,
      sEndIndex; //used to store the start index or end index of each partition generated in the previous stage for relation S
  int posLocal; //variables for storing the destination of the output tuple
  int sKeyVal; //realtion S Key value for each thread
  int sIdVal; //realtion S Id value for each thread

  int matchedRValue; //Matched R relation values.

  //getting mappingLen and mappingStartIndex
  int mappingLen = skewFlag[skewFlagLen];
  int mappingStartIndex = skewFlag[skewFlagStartIndex];

  for (int i = blockIdx.x + mappingStartIndex; i < mappingLen; i += gridDim.x) {

    //getting the start and end index of the relation R partition
    startIndex = rHisto[mapping[i] * GRID_SIZE_MULTIPLIER];
    endIndex = rHisto[(mapping[i] + 1) * GRID_SIZE_MULTIPLIER];

    //loading the relation R partition into shared memory
    for (int j = startIndex + threadIdx.x; j < endIndex; j += blockDim.x) {
      sharedPartitionR[j - startIndex] = rKey[j];
      sharedPartitionR[(endIndex - startIndex) + j - startIndex] = rId[j];
    }

    sharedPtr = 0;

    //barrier
    __syncthreads();

    sStartIndex = sHisto[mapping[i] * GRID_SIZE_MULTIPLIER] + mappingSecondary[i] * skewThreshold;
    sEndIndex = sHisto[(mapping[i] + 1) * GRID_SIZE_MULTIPLIER];

    if (mappingSecondary[i + 1]) {
      sEndIndex = sStartIndex + skewThreshold;
    }

    if (threadIdx.x < sEndIndex - sStartIndex) {

      posLocal = 0;
      sKeyVal = sKey[sStartIndex + threadIdx.x];
      sIdVal = sId[sStartIndex + threadIdx.x];

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

/**skew_detect_pth(int * , int *, int , int ).
* Kernel to detect skew in S relation partitions. It generates an array that
* specifies the number of blocks required to process a skewed partition.
* histo            : Histogram for relation S.
* pCount           : Partition count for each relation.
* skewThreshold    : The threshold value after which the partition requires additional blocks.
* skewFlag         : The skewFlag array which denotes the number of blocks required to process a skewed partition.
*/
__global__ void skew_detect_pth(int *histo, int pCount, int skewThreshold, int *skewFlag) {

  //getting thread id and work item count
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int numWorkItems = gridDim.x * blockDim.x;

  for (long i = tid; i < pCount; i += numWorkItems) {
    skewFlag[i] = ((int) ((histo[(i + 1) * GRID_SIZE_MULTIPLIER] - histo[i * GRID_SIZE_MULTIPLIER]) / skewThreshold));
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
__global__ void probe_skew_pth_small(data *rKey,
                                     data *rId,
                                     data *sKey,
                                     data *sId,
                                     int *rHisto,
                                     int *sHisto,
                                     int pCount,
                                     int *globalPtr,
                                     data *output,
                                     int pidStartIndex,
                                     int *skewFlag) {

  //allocating shared memory for storing each partition of relation R
  extern __shared__ data
  sharedPartitionR[];

  //pointer storing the index of each output tuple within the outputs generated by a block of threads
  __shared__ int sharedPtr;

  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage
  int posLocal; //variables for storing the destination of the output tuple
  int sKeyVal; //realtion S Key value for each thread
  int sIdVal; //realtion S Id value for each thread

  int matchedRValue; //Matched R relation values.

  for (int pid = pidStartIndex + blockIdx.x; pid < pCount; pid += gridDim.x) {

    if (skewFlag[pid] == 0) {

      //getting the start and end index of the relation R partition
      startIndex = rHisto[pid * GRID_SIZE_MULTIPLIER];
      endIndex = rHisto[(pid + 1) * GRID_SIZE_MULTIPLIER];

      //loading the relation R partition into shared memory
      for (int i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x) {
        sharedPartitionR[i - startIndex] = rKey[i];
        sharedPartitionR[(endIndex - startIndex) + i - startIndex] = rId[i];
      }

      sharedPtr = 0;
      posLocal = -1;

      //barrier
      __syncthreads();

      if (threadIdx.x < sHisto[(pid + 1) * GRID_SIZE_MULTIPLIER] - sHisto[pid * GRID_SIZE_MULTIPLIER]) {

        sKeyVal = sKey[sHisto[pid * GRID_SIZE_MULTIPLIER] + threadIdx.x];
        sIdVal = sId[sHisto[pid * GRID_SIZE_MULTIPLIER] + threadIdx.x];

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

        if (posLocal > -1) {
          output[sharedPtr + posLocal] = sIdVal;
          output[sharedPtr + posLocal + 1] = matchedRValue;
        }
      }
    }

  }

}

/**probe_pth(data *, data *, data *, data *, int *, int *, data *)
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
* skewThreshold     : Threshold for detecting data skew.
*/
__global__ void probe_skew_pth_large(data *rKey,
                                     data *rId,
                                     data *sKey,
                                     data *sId,
                                     int *rHisto,
                                     int *sHisto,
                                     int *skewFlag,
                                     int *globalPtr,
                                     data *output,
                                     int pid,
                                     int skewThreshold) {

  //allocating shared memory for storing each partition of relation R
  extern __shared__ data
  sharedPartitionR[];

  //pointer storing the index of each output tuple within the outputs generated by a block of threads
  __shared__ int sharedPtr;

  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage
  int posLocal; //variables for storing the destination of the output tuple
  int sKeyVal; //realtion S Key value for each thread
  int sIdVal; //realtion S Id value for each thread
  int tempEndIndex;

  int matchedRValue; //Matched R relation values.

  //getting the start and end index of the relation R partition
  startIndex = rHisto[pid * GRID_SIZE_MULTIPLIER] + blockIdx.x * skewThreshold;
  endIndex = rHisto[(pid + 1) * GRID_SIZE_MULTIPLIER];
  tempEndIndex = startIndex + skewThreshold;

  if (tempEndIndex <= endIndex) {

    //loading the relation R partition into shared memory
    for (int i = startIndex + threadIdx.x; i < tempEndIndex; i += blockDim.x) {
      sharedPartitionR[i - startIndex] = rKey[i];
      sharedPartitionR[(tempEndIndex - startIndex) + i - startIndex] = rId[i];
    }

    sharedPtr = 0;
    posLocal = -1;

    //barrier
    __syncthreads();

    if (threadIdx.x < sHisto[(pid + 1) * GRID_SIZE_MULTIPLIER] - sHisto[pid * GRID_SIZE_MULTIPLIER]) {

      sKeyVal = sKey[sHisto[pid * GRID_SIZE_MULTIPLIER] + threadIdx.x];
      sIdVal = sId[sHisto[pid * GRID_SIZE_MULTIPLIER] + threadIdx.x];

      //probing the R partition using the S partition
      for (int j = 0; j < tempEndIndex - startIndex; j++) {

        if (sKeyVal == sharedPartitionR[j]) {
          posLocal = atomicAdd(&sharedPtr, 2);
          matchedRValue = sharedPartitionR[tempEndIndex - startIndex + j];
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

      if (posLocal > -1) {
        output[sharedPtr + posLocal] = sIdVal;
        output[sharedPtr + posLocal + 1] = matchedRValue;
      }
    }

  }

}

/**probe_match_rate(data *, data *, data *, data *, int *, int *, data *)
* The function performs the probe operation for data sets with high match rate. In this kernel the
* block size should be greater than or equal to the size of the partition with maximum number of tuples.
* rKey          : The array containing the key values of partitioned relation R.
* rID           : The array containing the id values of partitioned relation R.
* sKey          : The array containing the key values of partitioned relation S.
* sID           : The array containing id key values of partitioned relation S.
* rHisto        : The histogram of relation R.
* sHisto        : The histogram of relation S.
* pCount        : The total number of partitions of each realtion.
* globalPtr     : The global pointer that is used to get the index of the output tuple.
* output        : The array used for storing the output of the probe operation.
* pFlag         : A flag used to denote wheter a given partition has completed the probe operation.
* pidStartIndex : Start index of the histogram array.
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
                                 int *pFlag,
                                 int pidStartIndex) {

  //allocating shared memory for storing each partition of relation R
  extern __shared__ data
  sharedPartitionR[];

  //pointer storing the index of each output tuple within the outputs generated by a block of threads
  __shared__ int sharedPtr;

  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage
  int posLocal; //variables for storing the destination of the output tuple
  int mCount = 0; //Number of matches for each S relation Tuple. mCount starts at 1 for the R table tuple
  int pSize; //Size of the aprtition being processed
  int sKeyVal, sIdVal; //sKey and sId value for each thread;

  int matches[MAX_MATCH_RATE]; //Matched R relation values.

  for (int pid = blockIdx.x + pidStartIndex; pid < pCount; pid += gridDim.x) {

    //checking if this partition has already been processed.
    if (pFlag[pid] != -1) {

      //setting mCount as 0 for the current partition
      mCount = 0;

      //setting the shared pointer value to 0
      sharedPtr = 0;

      //getting the start and end index of the relation R partition
      startIndex = rHisto[pid * GRID_SIZE_MULTIPLIER];
      endIndex = rHisto[(pid + 1) * GRID_SIZE_MULTIPLIER];
      pSize = endIndex - startIndex;

      //loading the relation R partition into shared memory
      for (int i = threadIdx.x; i < pSize; i += blockDim.x) {
        sharedPartitionR[i] = rKey[i + startIndex];
        sharedPartitionR[pSize + i] = rId[i + startIndex];
      }

      //barrier
      __syncthreads();

      if (threadIdx.x < sHisto[(pid + 1) * GRID_SIZE_MULTIPLIER] - sHisto[pid * GRID_SIZE_MULTIPLIER]) {

        //getting the sKey value for the thread
        sKeyVal = sKey[sHisto[pid * GRID_SIZE_MULTIPLIER] + threadIdx.x];
        sIdVal = sId[sHisto[pid * GRID_SIZE_MULTIPLIER] + threadIdx.x];

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
          if (sharedPtr <= MAX_OUTPUT_SIZE) {
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

  int sKeyVal; //realtion S Key value for each thread

  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage

  for (int pid = blockIdx.x; pid < pCount; pid += gridDim.x) {

    //getting the start and end index of the relation R partition
    startIndex = rHisto[pid * GRID_SIZE_MULTIPLIER];
    endIndex = rHisto[(pid + 1) * GRID_SIZE_MULTIPLIER];

    __shared__ int sharedPtr;

    //loading the relation R partition into shared memory
    for (int i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x) {
      sharedPartitionR[i - startIndex] = rKey[i];
    }

    sharedPtr = 0;

    //barrier
    __syncthreads();

    if (threadIdx.x < sHisto[(pid + 1) * GRID_SIZE_MULTIPLIER] - sHisto[pid * GRID_SIZE_MULTIPLIER]) {

      sKeyVal = sKey[sHisto[pid * GRID_SIZE_MULTIPLIER] + threadIdx.x];

      //probing the R partition using the S partition
      for (int j = 0; j < endIndex - startIndex; j++) {

        if (sKeyVal == sharedPartitionR[j]) {
          atomicAdd(&sharedPtr, 1);
        }
      }
    }

    //barrier
    __syncthreads();

    //writing the number of matches identified in the current partition to global memory.
    output[pid] = sharedPtr;
  }

}

/**probe_skew_dynamic(data *, data *, data *, data *, int , int , int , int , int *, int *, int)
* The child kernel for the baseline implementtaion that handles data sets with skew.
* rKey          : The array containing the key values of partitioned relation R.
* rID           : The array containing the id values of partitioned relation R.
* sKey          : The array containing the key values of partitioned relation S.
* sID           : The array containing id key values of partitioned relation S.
* rStartIndex   : Start index of the part of the R relation that needs to be processed by this kernel.
* rEndIndex     : End index of the part of the R relation that needs to be processed by this kernel.
* sStartIndex   : Start index of the part of the S relation that needs to be processed by this kernel.
* sEndIndex     : End index of the part of the S relation that needs to be processed by this kernel.
* globalPtr     : The global pointer that is used to get the index of the output tuple.
* output        : The array used for storing the output of the probe operation.
* skewThreshold : Threshold for detecting data skew.
*/
__global__ void probe_skew_dynamic(data *rKey,
                                   data *rId,
                                   data *sKey,
                                   data *sId,
                                   int rStartIndex,
                                   int rEndIndex,
                                   int sStartIndex,
                                   int sEndIndex,
                                   int *globalPtr,
                                   int *output,
                                   int skewThreshold) {

  //allocating shared memory for storing each partition of relation R
  extern __shared__ data
  sharedPartitionR[];

  //pointer storing the index of each output tuple within the outputs generated by a block of threads
  __shared__ int sharedPtr;

  int posLocal; //variables for storing the destination of the output tuple
  int sKeyVal; //realtion S Key value for each thread
  int sIdVal; //realtion S Id value for each thread

  int matchedRValue; //Matched R relation values.

  //loading the relation R partition into shared memory
  for (int i = threadIdx.x; i < rEndIndex - rStartIndex; i += blockDim.x) {
    sharedPartitionR[i] = rKey[i];
    sharedPartitionR[(rEndIndex - rStartIndex) + i] = rId[i];
  }

  sharedPtr = 0;

  //barrier
  __syncthreads();

  //checking if there is skew in the partition
  if (sEndIndex - sStartIndex > skewThreshold) {
    if (threadIdx.x == 0) {
      //launching child kernel
      probe_skew_dynamic << < 1, sEndIndex - sStartIndex - skewThreshold, 2 * (rEndIndex - rStartIndex) * sizeof(int) >>
          > (rKey + rStartIndex, rId + rStartIndex, sKey + sStartIndex + skewThreshold, sId + sStartIndex
              + skewThreshold, rStartIndex, rEndIndex, sStartIndex
              + skewThreshold, sEndIndex, globalPtr, output, skewThreshold);
    }
    sEndIndex = sStartIndex + skewThreshold;
  }

  if (threadIdx.x < sEndIndex - sStartIndex) {

    posLocal = 0;

    sKeyVal = sKey[threadIdx.x];
    sIdVal = sId[threadIdx.x];

    //probing the R partition using the S partition
    for (int j = 0; j < rEndIndex - rStartIndex; j++) {

      if (sKeyVal == sharedPartitionR[j]) {
        posLocal = atomicAdd(&sharedPtr, 2);
        matchedRValue = sharedPartitionR[rEndIndex - rStartIndex + j];
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

/**probe_skew_baseline(data *, data *, data *, data *, int *, int *, data *)
* The function performs the actual join operation by joining partitions with the same hash value.
* The number of threads in a block must be greater than or equalt to the number of tuples in the
* largest partition. We also assume a 100% match rate for this kernel.
* rKey          : The array containing the key values of partitioned relation R.
* rID           : The array containing the id values of partitioned relation R.
* sKey          : The array containing the key values of partitioned relation S.
* sID           : The array containing id key values of partitioned relation S.
* rHisto        : The histogram of relation R.
* sHisto        : The histogram of relation S.
* pCount        : The total number of partitions of each realtion.
* globalPtr     : The global pointer that is used to get the index of the output tuple.
* output        : The array used for storing the output of the probe operation.
* skewThreshold : Threshold for detecting data skew.
*/
/*__global__ void probe_skew_baseline(data *rKey, data *rId, data *sKey, data *sId, int *rHisto, int *sHisto, int pCount, int *globalPtr, data *output, int pidStartIndex, int skewThreshold){

    //allocating shared memory for storing each partition of relation R
    extern __shared__ data sharedPartitionR[];

    //pointer storing the index of each output tuple within the outputs generated by a block of threads
    __shared__ int sharedPtr;

    long startIndex, endIndex; //used to store the start index or end index of each partition generated in the previous stage
    long sStartIndex, sEndIndex; //used to store the start index or end index of each partition generated in the previous stage for relation S
    int posLocal; //variables for storing the destination of the output tuple
    int sKeyVal; //realtion S Key value for each thread
    int sIdVal; //realtion S Id value for each thread

    int matchedRValue; //Matched R relation values.

    for(int pid = pidStartIndex + blockIdx.x; pid < pCount; pid += gridDim.x){

        //getting the start and end index of the relation R partition
        startIndex = rHisto[pid * GRID_SIZE_MULTIPLIER];
        endIndex = rHisto[(pid + 1) * GRID_SIZE_MULTIPLIER];

        //loading the relation R partition into shared memory
        for(int i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x){
            sharedPartitionR[i - startIndex] = rKey[i];
            sharedPartitionR[(endIndex - startIndex) + i - startIndex] = rId[i];
        }

        sharedPtr = 0;

        //barrier
        __syncthreads();

        sStartIndex = sHisto[pid * GRID_SIZE_MULTIPLIER];
        sEndIndex = sHisto[(pid + 1) * GRID_SIZE_MULTIPLIER];

        //checking if there is skew in the partition
        if(sEndIndex - sStartIndex > skewThreshold){
            if(threadIdx.x == 0){
                //launching child kernel
                probe_skew_dynamic<<<1, sEndIndex - sStartIndex - skewThreshold, 2 * (endIndex - startIndex) * sizeof(data)>>>(rKey + startIndex, rId + startIndex, sKey + sStartIndex + skewThreshold, sId + sStartIndex + skewThreshold, startIndex, endIndex, sStartIndex + skewThreshold, sEndIndex, globalPtr, output, skewThreshold);
            }
            sEndIndex = sStartIndex + skewThreshold;
        }

        if(threadIdx.x < sEndIndex - sStartIndex){

            posLocal = 0;

            sKeyVal = sKey[sStartIndex + threadIdx.x];
            sIdVal = sId[sStartIndex + threadIdx.x];

            //probing the R partition using the S partition
            for(int j = 0; j < endIndex - startIndex; j++){

                if(sKeyVal == sharedPartitionR[j]){
                    posLocal = atomicAdd(&sharedPtr, 2);
                    matchedRValue = sharedPartitionR[endIndex - startIndex + j];
                }
            }

            //barrier
            __syncthreads();

            if(threadIdx.x == 0){

                //checking if there is space in the global buffer for writing the output
                sharedPtr = atomicAdd(globalPtr, sharedPtr);
            }

            //barrier
            __syncthreads();

            output[sharedPtr + posLocal] = sIdVal;
            output[sharedPtr + posLocal + 1] = matchedRValue;
        }

    }

}*/

/**histogram_build_L2_2008(data *, data *, long , int , int *, int *)
* This function builds the histogram for second level of partitioning based on the 2008 implementation.
* key           : The array containing the key values which has already been partitioned by reorder_L1.
* hashKey       : The hash value based on which the data will be partitioned in the second pass.
* hashKeyL1     : The hash value based on which the data was partitioned in the first pass.
* globalHisto   : The histogram data structure for the second partitioning pass, located in the global memory.
* globalHistoL1 : The histogram data structure generated during level 1 partitioning, also located in global memory.
*/
__global__ void histogram_build_L2_2008(data *key,
                                        int hashKey,
                                        int hashKeyL1,
                                        int *globalHisto,
                                        int *globalHistoL1,
                                        int gridDimL1,
                                        int blockDimL1) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  int hashValue; //variable for storing the hash value of each tuple
  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage

  //getting the start and end index of the partition to be processed by the current thread block
  startIndex = globalHistoL1[(blockIdx.x / GRID_SIZE_MULTIPLIER) * gridDimL1 * blockDimL1];
  endIndex = globalHistoL1[((blockIdx.x / GRID_SIZE_MULTIPLIER) + 1) * gridDimL1 * blockDimL1];

  //initializing all histogram entries in shared memory to 0. Otherwise there could be left over values from another thread block.
  for (int i = threadIdx.x; i < hashKey * blockDim.x; i += blockDim.x) {
    sharedHisto[i] = 0;
  }

  //barrier
  __syncthreads();

  //building the histogram in shared memory
  for (long i = (blockIdx.x % GRID_SIZE_MULTIPLIER) * blockDim.x + threadIdx.x + startIndex; i < endIndex;
       i += GRID_SIZE_MULTIPLIER * blockDim.x) {
    hashValue = (key[i] / hashKeyL1) % hashKey;
    sharedHisto[hashValue * blockDim.x + threadIdx.x] += 1;
  }

  //barrier
  __syncthreads();

  //writing the histogram back into the global memory
  for (int i = threadIdx.x; i < hashKey * blockDim.x; i += blockDim.x) {
    globalHisto[(int) (blockIdx.x / GRID_SIZE_MULTIPLIER) + blockIdx.x * blockDim.x * hashKey * GRID_SIZE_MULTIPLIER
        + (i * GRID_SIZE_MULTIPLIER) + (blockIdx.x % GRID_SIZE_MULTIPLIER)] = sharedHisto[i];
  }
}

/**reorder_L2_2008(data *, data *, data , data , int *, int *, data *, data *)
* This function re-orders a relation based on the 2008 implementation.
* key           : The array containing the key values which has already been partitioned by reorder_L1.
* id            : The array containing the id values which has already been partitioned by reorder_L1.
* hashKey       : The hash value based on which the data will be partitioned in the second pass.
* hashKeyL1     : The hash value based on which the data was partitioned in the first pass.
* globalHisto   : The histogram data structure for the second partitioning pass, located in the global memory.
* globalHistoL1 : The histogram data structure generated during level 1 partitioning, also located in global memory.
* keyOut        : The array storing the re-ordered key values.
* idOut         : The array storing the re-ordered id values.
*/
__global__ void reorder_L2_2008(data *key,
                                data *id,
                                data hashKey,
                                data hashKeyL1,
                                int *globalHisto,
                                int *globalHistoL1,
                                data *keyOut,
                                data *idOut,
                                int gridDimL1,
                                int blockDimL1) {

  //allocating shared memory for storing the histogram
  extern __shared__ int sharedHisto[];

  int hashValue; //variable for storing the hash value of each tuple
  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage
  int pos; //variable for storing the destination for re-ordering

  //getting the start and end index of the partition to be processed by the current thread block
  startIndex = globalHistoL1[(blockIdx.x / GRID_SIZE_MULTIPLIER) * gridDimL1 * blockDimL1];
  endIndex = globalHistoL1[((blockIdx.x / GRID_SIZE_MULTIPLIER) + 1) * gridDimL1 * blockDimL1];

  //reading the histogram data from global memory
  for (int i = threadIdx.x; i < hashKey * blockDim.x; i += blockDim.x) {
    sharedHisto[i] =
        globalHisto[(int) (blockIdx.x / GRID_SIZE_MULTIPLIER) + blockIdx.x * blockDim.x * hashKey * GRID_SIZE_MULTIPLIER
            + (i * GRID_SIZE_MULTIPLIER) + (blockIdx.x % GRID_SIZE_MULTIPLIER)];
  }

  //barrier
  __syncthreads();

  //re-ordering the data
  for (long i = (blockIdx.x % GRID_SIZE_MULTIPLIER) * blockDim.x + threadIdx.x + startIndex; i < endIndex;
       i += GRID_SIZE_MULTIPLIER * blockDim.x) {
    hashValue = (key[i] / hashKeyL1) % hashKey;
    pos = sharedHisto[hashValue * blockDim.x + threadIdx.x];
    sharedHisto[hashValue * blockDim.x + threadIdx.x]++;
    keyOut[pos] = key[i];
    idOut[pos] = id[i];
  }

}

/**probe_2008(data *, data *, data *, data *, int *, int *, data *)
* The function performs the actual join operation by joining partitions with the same hash value.
* This is based on the 2008 implementation.
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
__global__ void probe_2008(data *rKey,
                           data *rId,
                           data *sKey,
                           data *sId,
                           int *rHisto,
                           int *sHisto,
                           int pCount,
                           int *globalPtr,
                           data *output,
                           int pidStartIndex,
                           int blockDimL2) {

  //allocating shared memory for storing each partition of relation R
  extern __shared__ data
  sharedPartitionR[];

  //pointer storing the index of each output tuple within the outputs generated by a block of threads
  __shared__ int sharedPtr;

  long startIndex,
      endIndex; //used to store the start index or end index of each partition generated in the previous stage
  int posLocal; //variables for storing the destination of the output tuple
  int sKeyVal; //realtion S Key value for each thread
  int sIdVal; //realtion S Id value for each thread

  int matchedRValue; //Matched R relation values.

  for (int pid = pidStartIndex + blockIdx.x; pid < pCount; pid += gridDim.x) {

    //getting the start and end index of the relation R partition
    startIndex = rHisto[pid * blockDimL2 * GRID_SIZE_MULTIPLIER];
    endIndex = rHisto[(pid + 1) * blockDimL2 * GRID_SIZE_MULTIPLIER];

    //loading the relation R partition into shared memory
    for (int i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x) {
      sharedPartitionR[i - startIndex] = rKey[i];
      sharedPartitionR[(endIndex - startIndex) + i - startIndex] = rId[i];
    }

    sharedPtr = 0;

    //barrier
    __syncthreads();

    if (threadIdx.x
        < sHisto[(pid + 1) * blockDimL2 * GRID_SIZE_MULTIPLIER] - sHisto[pid * blockDimL2 * GRID_SIZE_MULTIPLIER]) {

      sKeyVal = sKey[sHisto[pid * blockDimL2 * GRID_SIZE_MULTIPLIER] + threadIdx.x];
      sIdVal = sId[sHisto[pid * blockDimL2 * GRID_SIZE_MULTIPLIER] + threadIdx.x];

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

__global__ void UVA_benchmark(data *input, data *output) {
  //int tid = blockIdx.x * blockDim.x + threadIdx.x;
  //int numWorkItems = gridDim.x * blockDim.x;

  //for(int i = tid; i < len; i += numWorkItems) {
  output[blockIdx.x * blockDim.x + threadIdx.x] = input[blockIdx.x * blockDim.x + threadIdx.x];
  //}
}

__global__ void build_kernel(data *rTableID,
                             data *rTableKey,
                             data *rHashTable,
                             int rTupleNum,
                             int rHashTableBucketNum) {
  int numWorkItems = gridDim.x * blockDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int key, val, hash, count;
  int hashBucketSize = 2 * (rTupleNum / rHashTableBucketNum + 2); //number of tuples inserted into each bucket

  while (tid < rTupleNum) {
    //phase 1
    key = rTableID[tid]; //get the key of one tuple
    val = rTableKey[tid]; //get the value of one tuple
    hash = key % rHashTableBucketNum;

    //phase 2
    if ((count = atomicAdd(&rHashTable[hash * hashBucketSize], 2)) < hashBucketSize) {
      rHashTable[hash * hashBucketSize + 2 + count + 0] = key;
      rHashTable[hash * hashBucketSize + 2 + count + 1] = val;
    }

    tid += numWorkItems;
  }
}

__global__ void probe_kernel(data *rHashTable,
                             data *sID,
                             data *sKey,
                             data *matchedTable,
                             int rTupleNum,
                             int sTupleNum,
                             int rHashTableBucketNum,
                             int *globalPtr) {
  uint numWorkItems = gridDim.x * blockDim.x;
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;

  int key, val, hash, count, matchedNum;
  int hashBucketSize = 2 * (rTupleNum / rHashTableBucketNum + 2);

  while (tid < sTupleNum) {
    //get one tuple from S table
    key = sID[tid];
    val = sKey[tid];

    //since hash value calculation consumes only tens ms, so GPU will finish it first
    hash = key % rHashTableBucketNum;

    //find out matched tuples in hash table for R table
    count = 0;
    while (count < hashBucketSize) {
      if (rHashTable[hash * hashBucketSize + 2 + count] == key) {
        matchedNum = atomicAdd(globalPtr, 1);
        matchedTable[matchedNum * 2] = val;
        matchedTable[matchedNum * 2 + 1] = rHashTable[hash * hashBucketSize + 2 + count + 1];
      }
      count += 2;
    }

    tid += numWorkItems;
  }
}

__global__ void probe_kernel_sm(data *rHashTable,
                                data *sID,
                                data *sKey,
                                data *matchedTable,
                                int rTupleNum,
                                int sTupleNum,
                                int rHashTableBucketNum,
                                int *globalPtr) {
  uint numWorkItems = gridDim.x * blockDim.x;
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;

  int key, val, hash, count;
  //int matchedNum;
  int hashBucketSize = 2 * (rTupleNum / rHashTableBucketNum + 2);
  int posLocal; //variables for storing the destination of the output tuple

  //pointer storing the index of each output tuple within the outputs generated by a block of threads
  __shared__ int sharedPtr;

  int matchedRValue; //Matched R relation values.

  while (tid < sTupleNum) {
    //get one tuple from S table
    key = sID[tid];
    val = sKey[tid];

    //since hash value calculation consumes only tens ms, so GPU will finish it first
    hash = key % rHashTableBucketNum;

    //find out matched tuples in hash table for R table
    count = 0;

    sharedPtr = 0;

    //barrier
    __syncthreads();

    while (count < hashBucketSize) {
      if (rHashTable[hash * hashBucketSize + 2 + count] == key) {
        posLocal = atomicAdd(&sharedPtr, 2);
        matchedRValue = rHashTable[hash * hashBucketSize + 2 + count + 1];

      }
      count += 2;
    }

    //barrier
    __syncthreads();

    if (threadIdx.x == 0) {

      //checking if there is space in the global buffer for writing the output
      sharedPtr = atomicAdd(globalPtr, sharedPtr);
    }

    //barrier
    __syncthreads();

    matchedTable[sharedPtr + posLocal] = val;
    matchedTable[sharedPtr + posLocal + 1] = matchedRValue;

    tid += numWorkItems;
  }
}

}
}

namespace LD {
namespace OPT {
/**histogram_build_L2(data *, data *, long , int , int *, int *)
    * This function builds the histogram for second level of partitioning.
    * key           : The array containing the key values which has already been partitioned by reorder_L1.
    * hashKey       : The hash value based on which the data will be partitioned in the second pass.
    * hashKeyL1     : The hash value based on which the data was partitioned in the first pass.
    * globalHisto   : The histogram data structure for the second partitioning pass, located in the global memory.
    * globalHistoL1 : The histogram data structure generated during level 1 partitioning, also located in global memory.
    */
__global__ void histogram_build_L2(data *key,
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
__global__ void reorder_L2(data *key,
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
__global__ void probe(data *rKey,
                      data *rId,
                      data *sKey,
                      data *sId,
                      int *rHisto,
                      int *sHisto,
                      int pCount,
                      int *globalPtr,
                      data *output,
                      int pidStartIndex) {

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
    startIndex = rHisto[pid * GRID_SIZE_MULTIPLIER];
    endIndex = rHisto[(pid + 1) * GRID_SIZE_MULTIPLIER];

    //loading the relation R partition into shared memory
    for (int i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x) {
      sharedPartitionR[i - startIndex] = rKey[i];
      sharedPartitionR[(endIndex - startIndex) + i - startIndex] = rId[i];
    }

    sharedPtr = 0;

    //barrier
    __syncthreads();

    if (threadIdx.x < sHisto[(pid + 1) * GRID_SIZE_MULTIPLIER] - sHisto[pid * GRID_SIZE_MULTIPLIER]) {

      sKeyVal = sKey[sHisto[pid * GRID_SIZE_MULTIPLIER] + threadIdx.x];
      sIdVal = sId[sHisto[pid * GRID_SIZE_MULTIPLIER] + threadIdx.x];

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
}