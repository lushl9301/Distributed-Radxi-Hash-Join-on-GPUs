add_library(hj_data ../empty.cpp)

add_library(hj_data_tuple ../empty.cpp Tuple.h)
add_library(hj_data_compressedTuple ../empty.cpp CompressedTuple.h)
add_library(hj_data_window Window.cpp Window.h)
add_library(hj_data_relation Relation.cpp Relation.h)

target_link_libraries(hj_data_window
                      hj_data_compressedTuple)
target_link_libraries(hj_data_relation
                      hj_data_tuple)
target_link_libraries(hj_data
                      hj_data_tuple
                      hj_data_window
                      hj_data_relation
                      hj_data_compressedTuple)