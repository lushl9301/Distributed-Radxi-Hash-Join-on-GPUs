#ifndef GPU_WRAPPER_HPP
#define GPU_WRAPPER_HPP

#include <cstdint>
#include "../../data/CompressedTuple.h"
namespace gpu {

class GPUWrapper {
 public:
  GPUWrapper() {
    ;
  }

  int BuildProbe(std::uint64_t innerPartitionSize, hpcjoin::data::CompressedTuple *innerPartition,
                 std::uint64_t outerPartitionSize, hpcjoin::data::CompressedTuple *outerPartition);
 private:
  // not really anything

};

}

#endif