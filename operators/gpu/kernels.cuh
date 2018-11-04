#include "../../data/data.hpp"

namespace SD {

__global__ void histogram_build_L1(data *, long, data, int *);

__global__ void memTest(data *in, data *out);

__global__ void reorder_L1(data *, data *, long, data, int *, data *, data *);

__global__ void histogram_build_L2(data *, int, int, int *, int *);

__global__ void reorder_L2(data *, data *, int, int, int *, int *, data *, data *);

__global__ void probe(data *, data *, data *, data *, int *, int *, int, int *, data *);

__global__ void histogram_build_global(data *, long, data, int *);

__global__ void reorder_global(data *, data *, long, data, int *, data *, data *);

__global__ void probe_match_rate(data *, data *, data *, data *, int *, int *, int, int *, data *, int *);

__global__ void histogram_build_L1_2008(data *, long, data, int *);

__global__ void reorder_L1_2008(data *, data *, long, data, int *, data *, data *);

namespace OPT {

__global__ void histogram_build_L2(data *, int, int, int *, int *, int);

__global__ void reorder_L2(data *, data *, int, int, int *, int *, data *, data *, int);

__global__ void skew_detect(int *, int, int, int *);

__global__ void skew_detect_pth(int *, int, int, int *);

__global__ void generate_block_mapping(int, int *, int *, int *);

__global__ void probe(data *, data *, data *, data *, int *, int *, int, int *, data *, int);

__global__ void probe_global_memory(data *, data *, data *, data *, int *, int *, int, int *, data *, int);

__global__ void probe_skew(data *,
                           data *,
                           data *,
                           data *,
                           int *,
                           int *,
                           int *,
                           data *,
                           int,
                           int *,
                           int *,
                           int *,
                           int,
                           int);

__global__ void probe_skew_pth_large(data *, data *, data *, data *, int *, int *, int *, int *, data *, int, int);

__global__ void probe_skew_pth_small(data *, data *, data *, data *, int *, int *, int, int *, data *, int, int *);

__global__ void probe_match_rate(data *, data *, data *, data *, int *, int *, int, int *, data *, int *, int);

__global__ void probe_count(data *, data *, int *, int *, int, data *);

__global__ void probe_skew_baseline(data *, data *, data *, data *, int *, int *, int, int *, data *, int, int);

__global__ void histogram_build_L2_2008(data *, int, int, int *, int *, int, int);

__global__ void reorder_L2_2008(data *, data *, data, data, int *, int *, data *, data *, int, int);

__global__ void probe_2008(data *, data *, data *, data *, int *, int *, int, int *, data *, int, int);

__global__ void UVA_benchmark(data *, data *);

__global__ void build_kernel(data *, data *, data *, int, int);

__global__ void probe_kernel(data *, data *, data *, data *, int, int, int, int *);

__global__ void probe_kernel_sm(data *, data *, data *, data *, int, int, int, int *);

namespace TLD {

__global__ void histogram_build_L1_tile(data *, long, data, int *);

__global__ void reorder_L1_tile(data *, data *, long, data, int *, data *, data *);

__global__ void reorder_L1_tile_opt(data *, data *, long, data, int *, data *, data *, int *, int);

__global__ void histogram_build_L2_tile(data *, int, int, int *, int *, int);

__global__ void histogram_build_L2_tile_opt(data *, int, int, int *, int *, int);

__global__ void histogram_build_L2_tile_multiplied(data *, int, int, int *, int *, int);

__global__ void reorder_L2_tile_multiplied(data *, data *, int, int, int *, int *, data *, data *, int);

__global__ void reorder_L2_tile(data *, data *, int, int, int *, int *, data *, data *, int);

__global__ void reorder_L2_tile_opt(data *, data *, int, int, int *, int *, data *, data *, int);

__global__ void probe_tile(data *, data *, data *, data *, int *, int *, int, int *, data *, int);

namespace RCD {
__global__ void histogram_build_L1_tile_rcd(Record *, long, data, int *);

__global__ void reorder_L1_tile_opt_rcd(Record *, long, data, int *, Record *, int);

__global__ void histogram_build_L2_tile_opt_rcd(Record *, int, int, int *, int *, int);

__global__ void reorder_L2_tile_opt_rcd(Record *, int, int, int *, int *, Record *, int);

}
}

}
}

//namspace containing all kernels related to small data sets
namespace MD {

}

//namspace containing all kernels related to small data sets
namespace LD {
__global__ void histogram_build_L1(data *, long, data, int *);

__global__ void reorder_L1(data *, data *, long, data, int *, data *, data *);

__global__ void histogram_build_L2(data *, int, int, int *, int *, int, int, int);

__global__ void reorder_L2(data *, data *, int, int, int *, int *, data *, data *, int, int, int);

__global__ void probe(data *, data *, data *, data *, int *, int *, int, int *, data *, int, int);

namespace OPT {

__global__ void histogram_build_L2(data *, int, int, int *, int *, int);

__global__ void reorder_L2(data *, data *, int, int, int *, int *, data *, data *, int);

__global__ void probe(data *, data *, data *, data *, int *, int *, int, int *, data *, int);
}
}
namespace SD {
namespace eth {

__global__ void probe_kernel_compressed(data *rHashTable,
                                        data *sID,
                                        int rTupleNum,
                                        int sTupleNum,
                                        int rHashTableBucketNum,
                                        int *globalPtr,
                                        uint32_t shiftBits,
                                        uint32_t keyShift);

}
}