add_subdirectory(gpu)
add_library(hj_utils ../empty.cpp)

add_library(hj_utils_debug ../empty.cpp Debug.h)
add_library(hj_utils_thread Thread.cpp Thread.h)

target_link_libraries(hj_utils
                      hj_utils_debug
                      hj_utils_thread)