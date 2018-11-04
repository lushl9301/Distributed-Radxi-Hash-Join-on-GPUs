#include <cstdlib>
#include <iostream>
#include "GPUWrapper.hpp"
#include "../../data/data.hpp"
#include "../../core/Configuration.h"
#include "../../operators/gpu/small_data.cuh"

namespace gpu {

#define NEXT_POW_2(V)                           \
    do {                                        \
        V--;                                    \
        V |= V >> 1;                            \
        V |= V >> 2;                            \
        V |= V >> 4;                            \
        V |= V >> 8;                            \
        V |= V >> 16;                           \
        V++;                                    \
    } while(0)

#define HASH_BIT_MODULO(KEY, MASK, NBITS) (((KEY) & (MASK)) >> (NBITS))

void gpu::GPUWrapper::execute() {
  auto *R = (relation_t *) malloc(sizeof(relation_t));
  auto *S = (relation_t *) malloc(sizeof(relation_t));
  R->numTuples = this->innerPartitionSize;
  S->numTuples = this->outerPartitionSize;
  cudaMallocManaged((void **) &R->id, R->numTuples * sizeof(data));
  cudaMemcpy(R->id, this->innerPartition, sizeof(R->numTuples) * sizeof(data), cudaMemcpyHostToDevice);
  cudaMallocManaged((void **) &S->id, S->numTuples * sizeof(data));
  cudaMemcpy(S->id, this->innerPartition, sizeof(S->numTuples) * sizeof(data), cudaMemcpyHostToDevice);

  uint32_t const keyShift = hpcjoin::core::Configuration::NETWORK_PARTITIONING_FANOUT +
      hpcjoin::core::Configuration::PAYLOAD_BITS;
  uint32_t const shiftBits = keyShift + hpcjoin::core::Configuration::LOCAL_PARTITIONING_FANOUT;
  uint64_t N = this->innerPartitionSize;
  NEXT_POW_2(N);
  args_t *args = (struct args_t *)malloc(sizeof(struct args_t));

  args->hRelRn = (relation_t *)malloc(sizeof(relation_t));
  args->hRelSn = (relation_t *)malloc(sizeof(relation_t));
  args->pCount = 1;
  args->matchRate = 0; // not in use
  args->zFactor = 0; // not in use
  args->hRelRn->numTuples = R->numTuples;
  args->hRelSn->numTuples = S->numTuples;

  cudaError_t error = cudaGetLastError();
  std::cout << cudaGetErrorString(error) << std::endl;

  SD::eth::simple_hash_join_compressed(R,
                                       S,
                                       args,
                                       shiftBits,
                                       keyShift);

}
task_type_t GPUWrapper::getType() {
  return TASK_BUILD_PROBE;
}
}