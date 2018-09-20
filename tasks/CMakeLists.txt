add_library(hj_tasks ../empty.cpp)
add_library(hj_tasks_task ../empty.cpp Task.h)
add_library(hj_tasks_buildProbe BuildProbe.cpp BuildProbe.h)
add_library(hj_tasks_histogramComputation HistogramComputation.cpp HistogramComputation.h)
add_library(hj_tasks_localPartitioning LocalPartitioning.cpp LocalPartitioning.h)
add_library(hj_tasks_networkPartitioning NetworkPartitioning.cpp NetworkPartitioning.h)

target_link_libraries(hj_tasks_histogramComputation
                      hj_histograms)
target_link_libraries(hj_tasks_localPartitioning
                      hj_tasks_buildProbe)
target_link_libraries(hj_tasks
                      hj_tasks_task
                      hj_tasks_buildProbe
                      hj_tasks_histogramComputation
                      hj_tasks_localPartitioning
                      hj_tasks_networkPartitioning)