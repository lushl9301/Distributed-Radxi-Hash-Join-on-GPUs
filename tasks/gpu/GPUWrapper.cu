#include <cstdlib>
#include <iostream>
#include "GPUWrapper.hpp"
#include "../../data/data.hpp"
#include "../../core/Configuration.h"
#include "../../operators/gpu/eth.cuh"

namespace gpu {
GPUWrapper::GPUWrapper(int32_t id,
                       float *time,
                       std::uint64_t innerPartitionSize,
                       hpcjoin::data::CompressedTuple *innerPartition,
                       std::uint64_t outerPartitionSize,
                       hpcjoin::data::CompressedTuple *outerPartition) {
  //cudaSetDevice(id);
  // cudaSetDevice(0);
  this->id = id;
  this->time = time;
  this->innerPartitionSize = innerPartitionSize;
  this->innerPartition = innerPartition;

  this->outerPartitionSize = outerPartitionSize;
  this->outerPartition = outerPartition;
}

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
  uint32_t const keyShift = hpcjoin::core::Configuration::NETWORK_PARTITIONING_FANOUT +
      hpcjoin::core::Configuration::PAYLOAD_BITS;
  uint32_t const shiftBits = keyShift + hpcjoin::core::Configuration::LOCAL_PARTITIONING_FANOUT;
  uint64_t N = this->innerPartitionSize;
  NEXT_POW_2(N);
  args_t *args = (struct args_t *) malloc(sizeof(struct args_t));

  args->pCount = 1;
  args->matchRate = 0; // not in use
  args->zFactor = 0; // not in use

//  cudaError_t error = cudaGetLastError();
//  std::cout << cudaGetErrorString(error) << std::endl;

  SD::eth::simple_hash_join_eth(this->innerPartition,
                                this->outerPartition,
                                this->innerPartitionSize,
                                this->outerPartitionSize,
                                args,
                                shiftBits,
                                keyShift,
                                id,
                                time);


}
task_type_t GPUWrapper::getType() {
  return TASK_BUILD_PROBE;
}
}