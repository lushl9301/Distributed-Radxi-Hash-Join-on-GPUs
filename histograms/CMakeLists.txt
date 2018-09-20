add_library(hj_histograms ../empty.cpp)

add_library(hj_histograms_assignmentMap AssignmentMap.cpp AssignmentMap.h)
add_library(hj_histograms_globalHistogram GlobalHistogram.cpp GlobalHistogram.h)
add_library(hj_histograms_localHistogram LocalHistogram.cpp LocalHistogram.h)
add_library(hj_histograms_offsetMap OffsetMap.cpp OffsetMap.h)

target_link_libraries(hj_histograms_globalHistogram
                      hj_histograms_localHistogram)
target_link_libraries(hj_histograms_assignmentMap
                      hj_histograms_globalHistogram)
target_link_libraries(hj_histograms_offsetMap
                      hj_histograms_assignmentMap
                      hj_histograms_globalHistogram
                      hj_histograms_localHistogram)

target_link_libraries(hj_histograms
                      hj_histograms_assignmentMap
                      hj_histograms_globalHistogram
                      hj_histograms_localHistogram
                      hj_histograms_offsetMap)