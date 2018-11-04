#include "../../data/data.hpp"
#include "../../utils/gpu/cuda_parameters.hpp"
#include "kernels.cuh"

namespace SD {

int shared_memory(relation_t *, relation_t *, args_t *, cudaParameters_t *);

int global_memory(relation_t *, relation_t *, args_t *, cudaParameters_t *);

int high_match_rate(relation_t *, relation_t *, args_t *, cudaParameters_t *);

namespace OPT {

int shared_memory(relation_t *, relation_t *, args_t *, cudaParameters_t *);
int shared_memory_PT(relation_t *, relation_t *, args_t *, cudaParameters_t *);

namespace TLD {
int shared_memory_tiled(relation_t *, relation_t *, args_t *, cudaParameters_t *);

int shared_memory_tiled_opt(relation_t *, relation_t *, args_t *, cudaParameters_t *);

namespace RCD {
int shared_memory_tiled_rcd(relation_t *, relation_t *, args_t *, cudaParameters_t *);
}
}

int shared_memory_2008(relation_t *, relation_t *, args_t *, cudaParameters_t *);

int shared_memory_streams_disabled(relation_t *, relation_t *, args_t *, cudaParameters_t *);

int global_memory(relation_t *, relation_t *, args_t *, cudaParameters_t *);

int shared_memory_skew(relation_t *, relation_t *, args_t *, cudaParameters_t *);

int shared_memory_skew_pth(relation_t *, relation_t *, args_t *, cudaParameters_t *);

int shared_memory_UVA(relation_t *, relation_t *, args_t *, cudaParameters_t *);
int shared_memory_UVA_PT(relation_t *, relation_t *, args_t *, cudaParameters_t *);

int shared_memory_UVA_sd(relation_t *, relation_t *, args_t *, cudaParameters_t *);

int high_match_rate(relation_t *, relation_t *, args_t *, cudaParameters_t *);

int high_match_rate_worst_case_estimate(relation_t *, relation_t *, args_t *, cudaParameters_t *);

int high_match_rate_baseline(relation_t *, relation_t *, args_t *, cudaParameters_t *);

int shared_memory_skew_baseline(relation_t *, relation_t *, args_t *, cudaParameters_t *);

int shared_memory_UVA1(relation_t *, relation_t *, args_t *, cudaParameters_t *, relation_t *, relation_t *);

int UVA_benchmark1(data *, data *, int, cudaParameters_t *);
int UVA_benchmark2(data *, data *, int, cudaParameters_t *);

int simple_hash_join(relation_t *, relation_t *, args_t *, cudaParameters_t *);
int simple_hash_join_GM(relation_t *, relation_t *, args_t *, cudaParameters_t *);
int simple_hash_join_SD(relation_t *, relation_t *, args_t *, cudaParameters_t *);
int simple_hash_join_SD_PT(relation_t *, relation_t *, args_t *, cudaParameters_t *);

}

namespace eth {

__global__ void build_kernel_compressed(data *rTableTuple,
                                        data *rHashTable,
                                        int rTupleNum,
                                        int rHashTableBucketNum,
                                        uint32_t shiftBits);
int simple_hash_join_compressed(relation_t *hRelR,
                                relation_t *hRelS,
                                args_t *args,
                                uint32_t shiftBits,
                                uint32_t keyShift);

}
}

int um_benchmark();