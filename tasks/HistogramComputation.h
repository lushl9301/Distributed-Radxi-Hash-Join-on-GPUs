/**
 * @author  Claude Barthels <claudeb@inf.ethz.ch>
 * (c) 2016, ETH Zurich, Systems Group
 *
 */

#ifndef HPCJOIN_TASKS_HISTOGRAMCOMPUTATION_H_
#define HPCJOIN_TASKS_HISTOGRAMCOMPUTATION_H_

#include "Task.h"
#include "../data/Relation.h"
#include "../histograms/GlobalHistogram.h"
#include "../histograms/LocalHistogram.h"
#include "../histograms/AssignmentMap.h"
#include "../histograms/OffsetMap.h"

namespace hpcjoin {
namespace tasks {

class HistogramComputation : public Task {

 public:

  HistogramComputation(uint32_t numberOfNodes, uint32_t nodeId,
					   hpcjoin::data::Relation *innerRelation,
					   hpcjoin::data::Relation *outerRelation);

  ~HistogramComputation();

 public:

  void execute();

  task_type_t getType();

 protected:

  void computeLocalHistograms();

  void computeGlobalInformation();

 public:

  uint32_t *getAssignment();

  uint64_t *getInnerRelationLocalHistogram();

  uint64_t *getOuterRelationLocalHistogram();

  uint64_t *getInnerRelationGlobalHistogram();

  uint64_t *getOuterRelationGlobalHistogram();

  uint64_t *getInnerRelationBaseOffsets();

  uint64_t *getOuterRelationBaseOffsets();

  uint64_t *getInnerRelationWriteOffsets();

  uint64_t *getOuterRelationWriteOffsets();

 protected:

  uint32_t nodeId;
  uint32_t numberOfNodes;

  hpcjoin::data::Relation *innerRelation;
  hpcjoin::data::Relation *outerRelation;

  hpcjoin::histograms::LocalHistogram *innerRelationLocalHistogram;
  hpcjoin::histograms::LocalHistogram *outerRelationLocalHistogram;

  hpcjoin::histograms::GlobalHistogram *innerRelationGlobalHistogram;
  hpcjoin::histograms::GlobalHistogram *outerRelationGlobalHistogram;

  hpcjoin::histograms::AssignmentMap *assignment;

  hpcjoin::histograms::OffsetMap *innerOffsets;
  hpcjoin::histograms::OffsetMap *outerOffsets;

};

} /* namespace tasks */
} /* namespace hpcjoin */

#endif /* HPCJOIN_TASKS_HISTOGRAMCOMPUTATION_H_ */
