#include <cstdlib>
#include "GPUWrapper.hpp"
#include "../../data/data.hpp"

namespace gpu {
// empty
void gpu::GPUWrapper::execute() {
  auto *R = (relation_t *) malloc(sizeof(relation_t));

}
task_type_t GPUWrapper::getType() {
  return TASK_BUILD_PROBE;
}
}