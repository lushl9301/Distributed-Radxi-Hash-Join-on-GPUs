/**
 * @author  Claude Barthels <claudeb@inf.ethz.ch>
 * (c) 2016, ETH Zurich, Systems Group
 *
 */

#include "HashJoin.h"

#include <stdlib.h>

#include "../data/Window.h"
#include "../core/Configuration.h"
#include "../tasks/HistogramComputation.h"
#include "../tasks/NetworkPartitioning.h"
#include "../tasks/LocalPartitioning.h"
#include "../tasks/BuildProbe.h"
#include "../performance/Measurements.h"
#include "../utils/Debug.h"
#include "../memory/Pool.h"
#include "../data/CompressedTuple.h"
#include "../data/Relation.h"

namespace hpcjoin {
namespace operators {

uint64_t HashJoin::RESULT_COUNTER = 0;
std::queue<hpcjoin::tasks::Task *> HashJoin::TASK_QUEUE;

HashJoin::HashJoin(uint32_t numberOfNodes, uint32_t nodeId, hpcjoin::data::Relation *innerRelation,
                   hpcjoin::data::Relation *outerRelation) {

  this->nodeId = nodeId;
  this->numberOfNodes = numberOfNodes;
  this->innerRelation = innerRelation;
  this->outerRelation = outerRelation;

}

HashJoin::~HashJoin() {

}

void HashJoin::join() {
  using Measurements = hpcjoin::performance::Measurements;

  /**********************************************************************/

  MPI_Barrier(MPI_COMM_WORLD);
  Measurements::startJoin();

  /**********************************************************************/

  /**
   * Histogram computation
   */
  Measurements::startHistogramComputation();
  auto *histogramComputation = new hpcjoin::tasks::HistogramComputation(this->numberOfNodes,
                                                                        this->nodeId,
                                                                        this->innerRelation,
                                                                        this->outerRelation);
  histogramComputation->execute();
  Measurements::stopHistogramComputation();
  JOIN_MEM_DEBUG("Histogram phase completed");

  /**********************************************************************/

  /**
   * Window allocation
   */

  Measurements::startWindowAllocation();
  auto *innerWindow = new hpcjoin::data::Window(this->numberOfNodes,
                                                this->nodeId,
                                                histogramComputation->getAssignment(),
                                                histogramComputation->getInnerRelationLocalHistogram(),
                                                histogramComputation->getInnerRelationGlobalHistogram(),
                                                histogramComputation->getInnerRelationBaseOffsets(),
                                                histogramComputation->getInnerRelationWriteOffsets());

  auto *outerWindow = new hpcjoin::data::Window(this->numberOfNodes,
                                                this->nodeId,
                                                histogramComputation->getAssignment(),
                                                histogramComputation->getOuterRelationLocalHistogram(),
                                                histogramComputation->getOuterRelationGlobalHistogram(),
                                                histogramComputation->getOuterRelationBaseOffsets(),
                                                histogramComputation->getOuterRelationWriteOffsets());
  Measurements::stopWindowAllocation();
  JOIN_MEM_DEBUG("Window allocated");

  /**********************************************************************/

  /**
   * Network partitioning
   */

  Measurements::startNetworkPartitioning();
  auto *networkPartitioning = new hpcjoin::tasks::NetworkPartitioning(this->nodeId,
                                                                      this->innerRelation,
                                                                      this->outerRelation,
                                                                      innerWindow,
                                                                      outerWindow);
  networkPartitioning->execute();
  Measurements::stopNetworkPartitioning();
  JOIN_MEM_DEBUG("Network phase completed");

  // OPTIMIZATION Save memory as soon as possible
  //delete this->innerRelation;
  //delete this->outerRelation;
  JOIN_MEM_DEBUG("Input relations deleted");

  /**********************************************************************/

  /**
   * Main synchronization
   */

  Measurements::startWaitingForNetworkCompletion();
  MPI_Barrier(MPI_COMM_WORLD);
  Measurements::stopWaitingForNetworkCompletion();

  /**********************************************************************/

  /**
   * Prepare transition
   */

  Measurements::startLocalProcessingPreparations();
  if (hpcjoin::core::Configuration::ENABLE_TWO_LEVEL_PARTITIONING) {
    //hpcjoin::memory::Pool::allocate((innerWindow->computeLocalWindowSize() + outerWindow->computeLocalWindowSize())*sizeof(hpcjoin::data::Tuple));
    hpcjoin::memory::Pool::reset();
  }
  // Create initial set of tasks
  uint32_t *assignment = histogramComputation->getAssignment();
  for (uint32_t p = 0; p < hpcjoin::core::Configuration::NETWORK_PARTITIONING_COUNT; ++p) {
    if (assignment[p] == this->nodeId) {
      hpcjoin::data::CompressedTuple *innerRelationPartition = innerWindow->getPartition(p);
      uint64_t innerRelationPartitionSize = innerWindow->getPartitionSize(p);
      hpcjoin::data::CompressedTuple *outerRelationPartition = outerWindow->getPartition(p);
      uint64_t outerRelationPartitionSize = outerWindow->getPartitionSize(p);

      if (hpcjoin::core::Configuration::ENABLE_TWO_LEVEL_PARTITIONING) {
        TASK_QUEUE.push(new hpcjoin::tasks::LocalPartitioning(innerRelationPartitionSize,
                                                              innerRelationPartition,
                                                              outerRelationPartitionSize,
                                                              outerRelationPartition));
      } else {
        TASK_QUEUE.push(new hpcjoin::tasks::BuildProbe(innerRelationPartitionSize,
                                                       innerRelationPartition,
                                                       outerRelationPartitionSize,
                                                       outerRelationPartition));
      }
    }
  }

  // Delete the network related computation
  delete histogramComputation;
  delete networkPartitioning;

  JOIN_MEM_DEBUG("Local phase prepared");

  Measurements::stopLocalProcessingPreparations();

  /**********************************************************************/

  /**
   * Local processing
   */

  // OPTIMIZATION Delete window as soon as possible
  bool windowsDeleted = false;

  // Execute tasks
  Measurements::startLocalProcessing();
  while (TASK_QUEUE.size() > 0) {

    hpcjoin::tasks::Task *task = TASK_QUEUE.front();
    TASK_QUEUE.pop();

    // OPTIMIZATION When second partitioning pass is completed, windows are no longer required
    if (hpcjoin::core::Configuration::ENABLE_TWO_LEVEL_PARTITIONING && windowsDeleted) {
      if (task->getType() == TASK_BUILD_PROBE) {
        delete innerWindow;
        delete outerWindow;
        windowsDeleted = true;
      }
    }

    task->execute();
    delete task;

  }
  Measurements::stopLocalProcessing();

  JOIN_MEM_DEBUG("Local phase completed");

  /**********************************************************************/

  Measurements::stopJoin();

  // OPTIMIZATION (see above)
  if (!windowsDeleted) {
    delete innerWindow;
    delete outerWindow;
  }

}

} /* namespace operators */
} /* namespace hpcjoin */
