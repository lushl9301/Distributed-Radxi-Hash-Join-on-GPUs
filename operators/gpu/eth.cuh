#include <cuda.h>
#include <cuda_runtime.h>
#include "../../data/CompressedTuple.h"
#include "../../data/data.hpp"
#include "../../utils/gpu/cuda_parameters.hpp"

namespace SD {

namespace eth {

__global__ void probe_kernel_eth(hpcjoin::data::CompressedTuple *rHashTable,
                                        hpcjoin::data::CompressedTuple *sID,
                                        int rTupleNum,
                                        int sTupleNum,
                                        int rHashTableBucketNum,
                                        int *globalPtr,
                                        uint32_t shiftBits,
                                        uint32_t keyShift);

__global__ void build_kernel_eth(hpcjoin::data::CompressedTuple *rTableID,
                                 hpcjoin::data::CompressedTuple *rHashTable,
                                 int rTupleNum,
                                 int rHashTableBucketNum,
                                 uint32_t shiftBits);

int simple_hash_join_eth(hpcjoin::data::CompressedTuple *hRelR,
                         hpcjoin::data::CompressedTuple *hRelS,
                         uint64_t rtuples,
                         uint64_t stuples,
                         args_t *args,
                         uint32_t shiftBits,
                         uint32_t keyShift);

}

}