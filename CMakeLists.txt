cmake_minimum_required(VERSION 3.0)

project(hashjoin)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
set(CMAKE_CXX_COMPILER /opt/bin/mpicxx)
set(CMAKE_C_COMPILER /opt/bin/mpicc)
set(MPI_CXX_COMPILER /opt/bin/mpicxx)
set(MPI_C_COMPILER /opt/bin/mpicc)
set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -mavx -lpthread -lpapi -D MEASUREMENT_DETAILS_HISTOGRAM -D MEASUREMENT_DETAILS_NETWORK -D MEASUREMENT_DETAILS_LOCALPART -D MEASUREMENT_DETAILS_LOCALBP")
set(MPI_CXX_FLAGS "-O3 -std=c++11 -mavx -lpthread -lpapi -D MEASUREMENT_DETAILS_HISTOGRAM -D MEASUREMENT_DETAILS_NETWORK -D MEASUREMENT_DETAILS_LOCALPART -D MEASUREMENT_DETAILS_LOCALBP")
include_directories("/usr/local/include")
#add_definitions("-D JOIN_DEBUG_PRINT")
#add_definitions("-D JOIN_DEBUG_PRINT -D JOIN_MEM_PRINT")

add_subdirectory(core)
add_subdirectory(data)
add_subdirectory(histograms)
add_subdirectory(memory)
add_subdirectory(operators)
add_subdirectory(performance)
add_subdirectory(tasks)
add_subdirectory(utils)
add_executable(program main.cpp)
target_link_libraries(program
                      hj_core
                      hj_utils
                      hj_data
                      hj_histograms
                      hj_memory
                      hj_operators
                      hj_performance
                      hj_tasks
                      ${MPI_LIBRARIES})
