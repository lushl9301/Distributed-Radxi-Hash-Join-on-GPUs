#ifndef GPU_WRAPPER_HPP
#define GPU_WRAPPER_HPP

#include <cstdint>
#include "../../data/CompressedTuple.h"
#include "../Task.h"

namespace gpu {

class GPUWrapper : public hpcjoin::tasks::Task {
 public:
  GPUWrapper(int32_t id,
             std::uint64_t innerPartitionSize,
             hpcjoin::data::CompressedTuple *innerPartition,
             std::uint64_t outerPartitionSize,
             hpcjoin::data::CompressedTuple *outerPartition);

  void execute();

  task_type_t getType();

 private:

  // not really anything

  uint64_t innerPartitionSize;
  hpcjoin::data::CompressedTuple *innerPartition;
  uint64_t outerPartitionSize;
  hpcjoin::data::CompressedTuple *outerPartition;

};

}

#endif