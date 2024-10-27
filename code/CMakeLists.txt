cmake_minimum_required(VERSION 3.16)
project(tricount)
include(../../common.cmake)

add_executable(tricount tricount.cpp)
target_link_libraries(tricount Kokkos::kokkos)


